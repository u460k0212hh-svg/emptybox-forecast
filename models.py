"""예측 모델 모음. 모두 fit(train_df) → predict(df) 인터페이스를 따른다.

회귀 계열은 '예측 시점에 알 수 있는 값'만 쓰는 시차 변수를 입력으로 받는다.
시계열 계열은 공컨테이너 순유입의 과거값만(또는 외생변수 1개를 더해) 사용한다.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

from .data import FEATURES, TARGET

CLEAN = ["month", "laden_imb_lag1", "laden_imb_lag2", "empty_net_lag1", "empty_net_lag12", "scfi_lag1"]
CORE = ["month", "laden_imb_lag1", "empty_net_lag1", "empty_net_lag12"]


class Baseline:
    """아무 분석 없이 쓸 수 있는 기준: 전월값, 전년동월값, 학습기간 평균."""
    def __init__(self, kind: str):
        self.kind, self.name = kind, {"lag1": "기준: 전월값", "lag12": "기준: 전년동월값", "mean": "기준: 학습기간 평균"}[kind]
    def fit(self, tr):
        self.mu = tr[TARGET].mean(); return self
    def predict(self, df):
        if self.kind == "lag1": return df["empty_net_lag1"].to_numpy(float)
        if self.kind == "lag12": return df["empty_net_lag12"].to_numpy(float)
        return np.full(len(df), self.mu)


class Linear:
    def __init__(self, features=FEATURES, name="선형 예측(8변수, 포털과 동일)"):
        self.features, self.name = list(features), name
    def fit(self, tr):
        self.m = LinearRegression().fit(tr[self.features], tr[TARGET]); return self
    def predict(self, df):
        return self.m.predict(df[self.features])
    @property
    def coef(self):
        return dict(zip(self.features, self.m.coef_))


class SeasonalRidge:
    """개선 모델: 전월 적컨 불균형 + (선택) 최근 평균 수준 + 월별 계절효과(더미), 릿지로 안정화."""
    def __init__(self, level: str | None = "empty_net_rm6", alpha: float = 3.0, name=None):
        from sklearn.linear_model import Ridge
        self.level, self.alpha, self._Ridge = level, alpha, Ridge
        base = "계절 릿지: 전월 적컨 불균형" + (" + 최근6개월 평균" if level == "empty_net_rm6" else " + 최근12개월 평균" if level else "") + " + 월 효과"
        self.name = name or f"{base} (α={alpha:g})"
    def _X(self, df):
        X = pd.DataFrame({"laden_imb_lag1": df["laden_imb_lag1"].to_numpy(float)})
        if self.level: X[self.level] = df[self.level].to_numpy(float)
        for m in range(2, 13): X[f"m{m}"] = (df["month"].to_numpy() == m).astype(float)
        return X
    def fit(self, tr):
        self.m = self._Ridge(self.alpha).fit(self._X(tr), tr[TARGET]); return self
    def predict(self, df):
        return self.m.predict(self._X(df))
    @property
    def coef(self):
        names = ["laden_imb_lag1"] + ([self.level] if self.level else []) + [f"m{m}" for m in range(2, 13)]
        return {"intercept": float(self.m.intercept_), **dict(zip(names, map(float, self.m.coef_)))}


class Forest:
    def __init__(self, max_depth=None, min_samples_leaf=1, features=FEATURES, name=None, seed=42):
        self.kw = dict(n_estimators=100, random_state=seed, max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        self.features = list(features)
        self.name = name or ("랜덤포레스트(포털과 동일)" if max_depth is None else f"랜덤포레스트(깊이≤{max_depth}, 잎≥{min_samples_leaf})")
    def fit(self, tr):
        self.m = RandomForestRegressor(**self.kw).fit(tr[self.features], tr[TARGET]); return self
    def predict(self, df):
        return self.m.predict(df[self.features])
    @property
    def importance(self):
        return dict(zip(self.features, self.m.feature_importances_))


# ---------------------------------------------------------------- 시계열 모형 (조건부 제곱합, CSS 추정)
class SARIMAX_CSS:
    """(S)ARIMA(p,0,q)(P,0,0)_12 + 선택적 외생변수. 조건부 제곱합(CSS)으로 추정한다.

    y_t = c + Σφ_i y_{t-i} + Σ Φ_j y_{t-12j} + β·x_t + e_t + Σθ_k e_{t-k}
    예측은 1개월 앞(one-step) 예측이며, 예측 시점까지의 실제값으로 오차항을 갱신한다.
    statsmodels 없이 numpy/scipy로 구현해 계산 과정을 그대로 보여 줄 수 있게 했다.
    """
    def __init__(self, p=1, q=0, P=0, exog: str | None = None, name=None):
        self.p, self.q, self.P, self.exog = p, q, P, exog
        tag = f"ARIMA({p},0,{q})" + (f"({P},0,0)₁₂" if P else "")
        self.name = name or (("ARIMAX " if exog else "") + tag + (" + 전월 적컨 불균형" if exog else ""))

    def _resid(self, params, y, x):
        p, q, P = self.p, self.q, self.P
        c = params[0]; phi = params[1:1+p]; Phi = params[1+p:1+p+P]; theta = params[1+p+P:1+p+P+q]
        beta = params[1+p+P+q] if self.exog else 0.0
        start = max(p, 12*P)
        e = np.zeros(len(y)); yhat = np.full(len(y), np.nan)
        for t in range(start, len(y)):
            f = c + sum(phi[i]*y[t-1-i] for i in range(p)) + sum(Phi[j]*y[t-12*(j+1)] for j in range(P))
            f += sum(theta[k]*e[t-1-k] for k in range(q) if t-1-k >= 0)
            if self.exog: f += beta*x[t]
            yhat[t] = f; e[t] = y[t]-f
        return e, yhat, start

    def fit(self, tr, history=None):
        hist = history if history is not None else tr
        y = hist[TARGET].to_numpy(float); x = hist[self.exog].to_numpy(float) if self.exog else None
        k = 1 + self.p + self.P + self.q + (1 if self.exog else 0)
        x0 = np.zeros(k); x0[0] = y.mean()
        def sse(par):
            e, _, s = self._resid(par, y, x)
            if any(abs(v) >= 0.99 for v in par[1+self.p+self.P:1+self.p+self.P+self.q]): return 1e12
            return float((e[s:]**2).sum())
        r = minimize(sse, x0, method="Nelder-Mead", options=dict(maxiter=20000, xatol=1e-6, fatol=1e-6))
        self.params = r.x; return self

    def predict_series(self, full: pd.DataFrame) -> np.ndarray:
        """full(시간순) 전체에 대해 1개월 앞 예측값을 돌려준다. 각 시점 예측은 그 이전 실제값만 사용."""
        y = full[TARGET].to_numpy(float); x = full[self.exog].to_numpy(float) if self.exog else None
        _, yhat, _ = self._resid(self.params, y, x)
        return yhat


class HoltWinters:
    """가법 계절 지수평활(주기 12). 1개월 앞 예측."""
    name = "지수평활(Holt-Winters, 가법 계절)"
    def _run(self, par, y):
        a, b, g = par; m = 12
        L = y[:m].mean(); T = (y[m:2*m].mean()-y[:m].mean())/m; S = list(y[:m]-L)
        yhat = np.full(len(y), np.nan)
        for t in range(m, len(y)):
            f = L + T + S[t-m]; yhat[t] = f
            Lp = L; L = a*(y[t]-S[t-m]) + (1-a)*(L+T); T = b*(L-Lp) + (1-b)*T; S.append(g*(y[t]-L) + (1-g)*S[t-m])
        return yhat
    def fit(self, tr, history=None):
        y = (history if history is not None else tr)[TARGET].to_numpy(float)
        def sse(par):
            if any(v <= 0 or v >= 1 for v in par): return 1e12
            h = self._run(par, y); return float(np.nansum((y-h)[12:]**2))
        self.par = minimize(sse, [0.3, 0.05, 0.2], method="Nelder-Mead").x; return self
    def predict_series(self, full):
        return self._run(self.par, full[TARGET].to_numpy(float))
