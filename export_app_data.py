"""대시보드(app/index.html)에 넣을 데이터를 만든다: 최종 모델 계수, 다음 달 예보, 백테스트, 항만 현황."""
import json
import numpy as np, pandas as pd
from emptybox.data import build_dataset, load_busan_monthly, load_port_annual, TARGET
from emptybox import models as M
from emptybox.alert import thresholds, level

R = json.load(open("results/metrics.json"))
chosen = R["C_selection"]["chosen_by_VAL_MAE"]
pred = pd.read_csv("results/predictions_rolling.csv")
data = build_dataset(extended=True)
b = load_busan_monthly()

final = M.SeasonalRidge("empty_net_rm6", 1.0)
assert final.name == chosen, (final.name, chosen)
final.fit(data)                                   # 2018.01~2026.08 전체로 다시 학습 → 2026.09 예보
c = final.coef
last = b.iloc[-1]
rm6 = float(b.empty_net_inflow.tail(6).mean())
nxt_month = 9
x = {"laden_imb_lag1": float(last.laden_imb), "empty_net_rm6": rm6, "month": nxt_month}
season = c.get(f"m{nxt_month}", 0.0)
fc = c["intercept"] + c["laden_imb_lag1"] * x["laden_imb_lag1"] + c["empty_net_rm6"] * rm6 + season
err = (pred.actual - pred[chosen]).to_numpy()
q10, q90 = np.percentile(err, [10, 90])
th = thresholds(data[TARGET])

bt = pd.read_csv("results/alert_backtest.csv")
out = {
  "asof": "2026-08", "next": "2026-09", "model": chosen,
  "coef": {k: round(v, 4) for k, v in c.items()},
  "inputs": {"laden_out": float(last.laden_out), "laden_in": float(last.laden_in), "laden_imb": float(last.laden_imb),
             "empty_net_rm6": round(rm6, 2), "recent6": [int(v) for v in b.empty_net_inflow.tail(6)]},
  "forecast": round(float(fc), 1), "interval80": [round(float(fc + q10), 1), round(float(fc + q90), 1)],
  "threshold": {k: round(v, 1) for k, v in th.items()}, "level": level(fc, th),
  "backtest": [{"ym": int(r.ym), "actual": float(r.actual), "model": round(float(r[chosen]), 1),
                "portal": round(float(r["선형 예측(8변수, 포털과 동일)"]), 1), "naive": float(r["기준: 전월값"]),
                "warn": float(bt.loc[i, "warn"]), "danger": float(bt.loc[i, "danger"])} for i, r in pred.iterrows()],
  "leaderboard": [{"name": k, **v} for k, v in R["C_selection"]["candidates"].items()],
  "alert": R["E_alert"], "boot": R["D_test"],
  "ports": load_port_annual().to_dict(orient="records"),
  "realloc": R["F_reallocation"],
  "history": [{"ym": int(r.ym), "net": float(r.empty_net_inflow), "imb": float(r.laden_imb), "lin": float(r.laden_in), "lout": float(r.laden_out)} for _, r in b[b.ym >= 201701].iterrows()],
}
json.dump(out, open("app/data.json", "w"), ensure_ascii=False)
print(json.dumps({k: out[k] for k in ["model", "coef", "inputs", "forecast", "interval80", "threshold", "level"]}, ensure_ascii=False, indent=1))
