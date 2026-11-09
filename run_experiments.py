"""전체 실험 (재현 → 진단 → 검증구간에서 모델 선정 → 시험구간 1회 평가 → 경보 → 재배치).

평가 원칙
- 시간순 확장 윈도: 매달 그 직전까지의 자료로 다시 학습해 다음 달 하나만 예측(실제 운영과 동일).
- 모델 선정은 검증구간(2021.01~2024.12, 48개월)에서만 하고, 시험구간(2025.01~2026.08, 20개월)은 마지막에 한 번만 본다.
실행: PYTHONPATH=src python3 scripts/run_experiments.py
"""
import json, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from emptybox.data import build_dataset, load_busan_monthly, load_port_annual, FEATURES, TARGET
from emptybox import models as M
from emptybox.evaluate import fixed_split, expanding_window, mae, rmse, r2, bootstrap_diff
from emptybox.alert import thresholds, level
from emptybox.reallocation import summarize

OUT = Path("results"); OUT.mkdir(exist_ok=True)
VAL = (202101, 202412); TEST = (202501, 202608)
R = {"protocol": {"validation": VAL, "test": TEST}}
data = build_dataset(extended=True)
hist = load_busan_monthly(); hist["laden_imb_lag1"] = hist.laden_imb.shift(1)
hist = hist[hist.ym >= 201610].reset_index(drop=True)

# A. 포털 결과 재현 --------------------------------------------------------------------------
rep = {}
for m in [M.Linear(), M.Forest(), M.Baseline("lag1"), M.Baseline("lag12"), M.Baseline("mean")]:
    ym, y, p = fixed_split(m, data); rep[m.name] = {"MAE": round(mae(y, p), 2), "RMSE": round(rmse(y, p), 2), "R2": round(r2(y, p), 3)}
tr, te = data[data.split != "test"], data[data.split == "test"]
lin = M.Linear().fit(tr); rf = M.Forest().fit(tr)
R["A_reproduce"] = {"metrics": rep, "linear_coef": {k: round(v, 4) for k, v in lin.coef.items()},
                    "rf_importance": {k: round(v, 3) for k, v in sorted(rf.importance.items(), key=lambda x: -x[1])}}
assert (rep["선형 예측(8변수, 포털과 동일)"]["MAE"], rep["랜덤포레스트(포털과 동일)"]["MAE"], rep["기준: 전월값"]["MAE"], rep["기준: 전년동월값"]["MAE"]) == (12.05, 12.94, 17.85, 16.95)
print("A 재현 OK")

# B. 진단: 과적합, 완전 공선성 ----------------------------------------------------------------
X = tr[FEATURES].to_numpy(float)
vif = {}
for j, f in enumerate(FEATURES):
    o = np.delete(X, j, 1); A_ = np.c_[np.ones(len(o)), o]; r = X[:, j] - A_ @ np.linalg.lstsq(A_, X[:, j], rcond=None)[0]
    r2j = 1 - r.var() / X[:, j].var(); vif[f] = "inf" if r2j > 0.999999 else round(1 / (1 - r2j), 2)
diag = {}
for m in [M.Linear(), M.Forest()]:
    m.fit(tr); diag[m.name] = {"train_MAE": round(mae(tr[TARGET], m.predict(tr)), 2), "test_MAE": round(mae(te[TARGET], m.predict(te)), 2)}
tss = TimeSeriesSplit(n_splits=4, test_size=12); grid = []
for depth in [2, 3, 4, 6, None]:
    for leaf in [1, 3, 5, 10]:
        errs = [mae(tr.iloc[b][TARGET], M.Forest(depth, leaf).fit(tr.iloc[a]).predict(tr.iloc[b])) for a, b in tss.split(tr)]
        grid.append((float(np.mean(errs)), depth, leaf))
cv_mae, bd, bl = min(grid, key=lambda g: g[0])
best_rf = M.Forest(bd, bl).fit(tr)
diag[best_rf.name] = {"train_MAE": round(mae(tr[TARGET], best_rf.predict(tr)), 2), "test_MAE": round(mae(te[TARGET], best_rf.predict(te)), 2), "cv_MAE": round(cv_mae, 2)}
R["B_diagnosis"] = {"vif": vif, "identity": "empty_net_lag1 = empty_in_lag1 - empty_out_lag1 (완전 공선성)",
                    "max_identity_error": float((tr.empty_net_lag1 - (tr.empty_in_lag1 - tr.empty_out_lag1)).abs().max()),
                    "overfit": diag, "rf_best": {"max_depth": bd, "min_leaf": bl}}
print("B 진단", diag)

# C. 검증구간 모델 선정 (+ D. 시험구간 1회 평가) -------------------------------------------------
cands = [M.Baseline("lag1"), M.Baseline("lag12"), M.Baseline("mean"),
         M.Linear(), M.Linear(M.CLEAN, "선형 예측(겹치는 변수 제거, 6변수)"), M.Linear(M.CORE, "선형 예측(핵심 4변수)"),
         M.Forest(), M.Forest(bd, bl),
         M.HoltWinters(), M.SARIMAX_CSS(1, 0), M.SARIMAX_CSS(1, 1), M.SARIMAX_CSS(1, 1, 1),
         M.SARIMAX_CSS(1, 1, 1, exog="laden_imb_lag1")]
cands += [M.SeasonalRidge(lv, a) for lv in [None, "empty_net_rm6", "empty_net_rm12"] for a in [1, 3, 10, 30]]
res, P = {}, {}
t0 = time.time()
for m in cands:
    h = hist if hasattr(m, "predict_series") else None
    ym, y, p = expanding_window(m, data, VAL[0], history=h)
    sel_v, sel_t = (ym >= VAL[0]) & (ym <= VAL[1]), ym >= TEST[0]
    res[m.name] = {"VAL_MAE": round(mae(y[sel_v], p[sel_v]), 2), "VAL_RMSE": round(rmse(y[sel_v], p[sel_v]), 2),
                   "TEST_MAE": round(mae(y[sel_t], p[sel_t]), 2), "TEST_RMSE": round(rmse(y[sel_t], p[sel_t]), 2)}
    P[m.name] = p
    print(f"  {m.name:48s} {res[m.name]} ({time.time()-t0:.0f}s)")
yy = y
chosen = min(res, key=lambda k: res[k]["VAL_MAE"])
R["C_selection"] = {"candidates": res, "chosen_by_VAL_MAE": chosen}
print("C 선정:", chosen, res[chosen])

# D. 시험구간: 선택 모델 vs 기준, 부트스트랩 구간 --------------------------------------------------
sel_t = ym >= TEST[0]
boot = {}
for other in ["기준: 전월값", "기준: 전년동월값", "기준: 학습기간 평균", "선형 예측(8변수, 포털과 동일)"]:
    lo, hi = bootstrap_diff(yy[sel_t], P[chosen][sel_t], P[other][sel_t])
    boot[other] = {"MAE_diff": round(mae(yy[sel_t], P[chosen][sel_t]) - mae(yy[sel_t], P[other][sel_t]), 2), "CI95": [round(lo, 2), round(hi, 2)]}
allp = np.ones_like(yy, dtype=bool)
boot_all = {}
for other in ["기준: 전월값", "기준: 학습기간 평균", "선형 예측(8변수, 포털과 동일)"]:
    lo, hi = bootstrap_diff(yy, P[chosen], P[other])
    boot_all[other] = {"MAE_diff": round(mae(yy, P[chosen]) - mae(yy, P[other]), 2), "CI95": [round(lo, 2), round(hi, 2)]}
final = [m for m in cands if m.name == chosen][0].fit(data[data.ym < TEST[0]])
final_all = [m for m in cands if m.name == chosen][0].__class__
R["D_test"] = {"bootstrap_test_20m": boot, "bootstrap_val+test_68m": boot_all,
               "final_coef_trained_to_2024": {k: round(v, 3) for k, v in final.coef.items()}}
print("D 시험", boot, boot_all)

# E. 조기경보 백테스트(선택 모델, 2021.01~2026.08) ---------------------------------------------------
rows = []
for i, (m_, a_, p_) in enumerate(zip(ym, yy, P[chosen])):
    th = thresholds(data[data.ym < m_][TARGET])
    rows.append({"ym": int(m_), "actual": a_, "pred": round(float(p_), 1), "warn": round(th["warn"], 1), "danger": round(th["danger"], 1),
                 "pred_level": level(p_, th), "actual_level": level(a_, th)})
bt = pd.DataFrame(rows)
def conf(df, key):
    a = df.actual >= df[key]; p = df.pred >= df[key]
    n = {"적중": int((a & p).sum()), "미탐지": int((a & ~p).sum()), "오경보": int((~a & p).sum()), "정상확인": int((~a & ~p).sum())}
    n["재현율"] = round(n["적중"] / max(1, n["적중"] + n["미탐지"]), 2); n["정밀도"] = round(n["적중"] / max(1, n["적중"] + n["오경보"]), 2); return n
base_rate = {"warn": round(float((bt.actual >= bt.warn).mean()), 2)}
R["E_alert"] = {"model": chosen, "test_warn": conf(bt[bt.ym >= TEST[0]], "warn"), "test_danger": conf(bt[bt.ym >= TEST[0]], "danger"),
                "all_warn": conf(bt, "warn"), "all_danger": conf(bt, "danger"), "base_rate": base_rate,
                "portal_rule_test": {"warn": 102.2, "danger": 113.4}}
bt.to_csv(OUT / "alert_backtest.csv", index=False)
print("E 경보", R["E_alert"])

# F. 재배치 -------------------------------------------------------------------------------
port = load_port_annual(); R["F_reallocation"] = [summarize(port, 2024), summarize(port, 2025)]

pd.DataFrame({"ym": ym, "actual": yy, **P}).to_csv(OUT / "predictions_rolling.csv", index=False)
json.dump(R, open(OUT / "metrics.json", "w"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("saved")
