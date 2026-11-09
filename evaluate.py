"""평가 도구: 고정 분할 평가, 매달 다시 학습하는 확장 윈도 평가, 부트스트랩 신뢰구간."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .data import TARGET

def mae(y, p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y)-np.asarray(p))**2)))
def r2(y, p):
    y, p = np.asarray(y), np.asarray(p); return float(1-((y-p)**2).sum()/((y-y.mean())**2).sum())


def is_series_model(m) -> bool:
    return hasattr(m, "predict_series")


def fixed_split(model, data: pd.DataFrame, history: pd.DataFrame | None = None):
    """포털과 같은 방식: 2018~2024로 한 번 학습, 2025.01~2026.08 예측."""
    tr, te = data[data.split != "test"], data[data.split == "test"]
    if is_series_model(model):
        full = history if history is not None else data
        model.fit(tr, history=full[full.ym < te.ym.min()])
        pred = pd.Series(model.predict_series(full), index=full.ym).loc[te.ym].to_numpy()
    else:
        model.fit(tr); pred = model.predict(te)
    return te.ym.to_numpy(), te[TARGET].to_numpy(float), np.asarray(pred, float)


def expanding_window(model, data: pd.DataFrame, start: int, history: pd.DataFrame | None = None):
    """매달 그 직전까지의 데이터로 다시 학습해 다음 달 하나를 예측한다(실제 운영 방식)."""
    out = []
    full = history if history is not None else data
    for ym in data.ym[data.ym >= start]:
        tr = data[data.ym < ym]; row = data[data.ym == ym]
        if is_series_model(model):
            model.fit(tr, history=full[full.ym < ym])
            p = pd.Series(model.predict_series(full[full.ym <= ym]), index=full.ym[full.ym <= ym]).loc[ym]
        else:
            model.fit(tr); p = model.predict(row)[0]
        out.append((ym, float(row[TARGET].iloc[0]), float(p)))
    o = pd.DataFrame(out, columns=["ym", "y", "pred"])
    return o.ym.to_numpy(), o.y.to_numpy(), o.pred.to_numpy()


def bootstrap_diff(y, p_a, p_b, n=5000, seed=0, block=3):
    """모델 A와 B의 MAE 차이(A-B)에 대한 이동 블록 부트스트랩 95% 구간."""
    rng = np.random.default_rng(seed); ea, eb = np.abs(y-p_a), np.abs(y-p_b); T = len(y); diffs = []
    for _ in range(n):
        idx = np.concatenate([np.arange(s, s+block) % T for s in rng.integers(0, T, int(np.ceil(T/block)))])[:T]
        diffs.append(ea[idx].mean()-eb[idx].mean())
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
