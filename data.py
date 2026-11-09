"""원자료를 읽어 분석용 월별 데이터셋을 만든다.

- 부산항 적·공컨테이너 수입·수출(환적 제외): 부산항만공사 체인포털 화면을 판독해 만든
  data/processed/busan_monthly_bpa.csv (단위 천TEU, 2016.09~2026.08)
- SCFI: 국가물류통합정보센터 주간 지수 → 월평균
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

FEATURES = ["month", "laden_imb_lag1", "laden_imb_lag2", "empty_net_lag1",
            "empty_net_lag12", "empty_in_lag1", "empty_out_lag1", "scfi_lag1"]
TARGET = "empty_net_inflow"


def load_busan_monthly(path: Path | None = None) -> pd.DataFrame:
    d = pd.read_csv(path or DATA / "processed" / "busan_monthly_bpa.csv")
    d["empty_net_inflow"] = d.empty_in - d.empty_out          # 공컨 순유입 = 수입 - 수출
    d["laden_imb"] = d.laden_out - d.laden_in                 # 적컨 불균형 = 수출 - 수입
    return d.sort_values("ym").reset_index(drop=True)


def load_scfi_monthly(folder: Path | None = None) -> pd.Series:
    frames = []
    for f in sorted((folder or DATA / "raw" / "scfi").glob("*.xlsx")):
        d = pd.read_excel(f)
        d.columns = ["kind", "value", "date", "unit"]
        d = d[d.kind == "SCFI"]
        d["value"] = pd.to_numeric(d.value.astype(str).str.replace(",", ""))
        d["date"] = pd.to_datetime(d.date)
        frames.append(d[["date", "value"]])
    s = pd.concat(frames).drop_duplicates("date").sort_values("date")
    m = s.groupby(s.date.dt.year * 100 + s.date.dt.month).value.mean().round(1)
    m.index.name = "ym"
    return m.rename("scfi")


def build_dataset(start: int = 201801, end: int = 202608,
                  test_start: int = 202501, valid_start: int = 202401, extended: bool = False) -> pd.DataFrame:
    """레시피·포털 분석과 같은 104개월 데이터셋. 모든 독립변수는 예측 시점에 알 수 있는 전월 이전 값."""
    b = load_busan_monthly()
    b = b.merge(load_scfi_monthly().reset_index(), on="ym", how="left")
    b["laden_imb_lag1"] = b.laden_imb.shift(1)
    b["laden_imb_lag2"] = b.laden_imb.shift(2)
    b["empty_net_lag1"] = b.empty_net_inflow.shift(1)
    b["empty_net_lag12"] = b.empty_net_inflow.shift(12)
    b["empty_in_lag1"] = b.empty_in.shift(1)
    b["empty_out_lag1"] = b.empty_out.shift(1)
    b["scfi_lag1"] = b.scfi.shift(1)
    # 확장 변수(개선 모델용): 최근 6·12개월 평균 수준
    b["empty_net_rm6"] = b.empty_net_inflow.shift(1).rolling(6).mean()
    b["empty_net_rm12"] = b.empty_net_inflow.shift(1).rolling(12).mean()
    d = b[(b.ym >= start) & (b.ym <= end)].copy()
    d["split"] = "train"
    d.loc[d.ym >= valid_start, "split"] = "valid"
    d.loc[d.ym >= test_start, "split"] = "test"
    cols = ["ym", "year", "month", TARGET] + FEATURES[1:] + ["split"]
    if extended:
        cols += ["empty_net_rm6", "empty_net_rm12"]
    return d[cols].reset_index(drop=True)


def load_port_annual(path: Path | None = None) -> pd.DataFrame:
    return pd.read_csv(path or DATA / "processed" / "port_empty_container_2024_2025.csv")


if __name__ == "__main__":
    ds = build_dataset()
    out = DATA / "processed" / "busan_empty_container_dataset.csv"
    ds.to_csv(out, index=False)
    print(out, ds.shape)
