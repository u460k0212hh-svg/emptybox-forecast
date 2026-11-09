"""항만 간 재배치: 부족형(순유입>0)과 잉여형(순유입<0) 합계로 국내 대체 가능 범위를 계산한다."""
import pandas as pd

def summarize(port: pd.DataFrame, year: int, shares=(0.1, 0.3, 0.5)) -> dict:
    d = port[port.year == year]
    short = d[d.empty_net_inflow > 0].empty_net_inflow.sum()
    surplus = -d[d.empty_net_inflow < 0].empty_net_inflow.sum()
    cap = min(short, surplus)
    return {"year": year, "shortage": round(short, 1), "surplus": round(surplus, 1), "max_transfer": round(cap, 1),
            "scenarios": {f"{int(s*100)}%": round(cap*s, 1) for s in shares}}
