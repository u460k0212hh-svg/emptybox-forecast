"""부산항 공컨테이너 한 달 앞 예보 — Streamlit 앱.

실행: streamlit run streamlit_app.py
데이터: app/data.json (scripts/export_app_data.py가 매달 갱신)
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
D = json.loads((ROOT / "app" / "data.json").read_text(encoding="utf-8"))
C, TH = D["coef"], D["threshold"]

st.set_page_config(page_title="부산항 공컨 예보", page_icon="🚢", layout="wide")


def ym_label(ym: int) -> str:
    return f"{ym // 100}.{ym % 100:02d}"


def predict(laden_imb: float, rm6: float, month: int) -> dict:
    parts = {
        "기본값": C["intercept"],
        f"{month}월 효과": C.get(f"m{month}", 0.0),
        "전월 적컨 불균형": C["laden_imb_lag1"] * laden_imb,
        "최근 6개월 평균": C["empty_net_rm6"] * rm6,
    }
    return {"parts": parts, "value": sum(parts.values())}


def level(v: float) -> str:
    return "위험" if v >= TH["danger"] else "주의" if v >= TH["warn"] else "정상"


ICON = {"정상": "🟢", "주의": "🟠", "위험": "🔴"}
next_month = int(D["next"].split("-")[1])
inp = D["inputs"]
lo_off = D["interval80"][0] - D["forecast"]
hi_off = D["interval80"][1] - D["forecast"]

# ── 헤더 ────────────────────────────────────────────
st.title("부산항 빈 컨테이너, 한 달 먼저")
st.caption(f"기준 자료 {D['asof']} · 모델: {D['model']} · 단위 천TEU (공컨 수입 − 수출, 환적 제외)")

tab_now, tab_sim, tab_bt, tab_lb, tab_port = st.tabs(
    ["다음 달 예보", "시뮬레이터", "백테스트", "모델 비교", "항만 재배치"]
)

# ── 1. 다음 달 예보 ──────────────────────────────────
with tab_now:
    lv = D["level"]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"{D['next']} 순유입 예보", f"{D['forecast']:.1f}")
    c2.metric("경보", f"{ICON[lv]} {lv}")
    c3.metric("80% 예측 구간", f"{D['interval80'][0]:.1f} ~ {D['interval80'][1]:.1f}")
    st.write(f"주의 기준 **{TH['warn']}** · 위험 기준 **{TH['danger']}** (학습 기간 평균 {TH['mean']} + 0.5σ / 1σ)")

    st.subheader("왜 이 값인가")
    p = predict(inp["laden_imb"], inp["empty_net_rm6"], next_month)
    st.dataframe(
        pd.DataFrame({"항목": list(p["parts"]), "기여(천TEU)": [round(v, 1) for v in p["parts"].values()]}),
        hide_index=True, use_container_width=True,
    )
    st.caption(
        f"전월 적컨 수출 {inp['laden_out']:.0f} − 수입 {inp['laden_in']:.0f} = 불균형 {inp['laden_imb']:.0f} · "
        f"최근 6개월 순유입 {inp['recent6']} → 평균 {inp['empty_net_rm6']:.2f}"
    )

# ── 2. 시뮬레이터 ────────────────────────────────────
with tab_sim:
    st.write("전월 적컨 수출입이 달라지면 예보와 경보가 어떻게 바뀌는지 확인합니다.")
    a, b, c = st.columns(3)
    lout = a.slider("전월 적컨 수출", 250, 450, int(inp["laden_out"]))
    lin = b.slider("전월 적컨 수입", 180, 350, int(inp["laden_in"]))
    rm6 = c.slider("최근 6개월 평균 순유입", 50.0, 130.0, float(inp["empty_net_rm6"]), 0.5)
    month = st.select_slider("예보 대상 월", options=list(range(1, 13)), value=next_month)
    s = predict(lout - lin, rm6, month)
    lv = level(s["value"])
    m1, m2, m3 = st.columns(3)
    m1.metric("예보", f"{s['value']:.1f}", f"{round(s['value'] - D['forecast'], 1) + 0.0:+.1f} vs 현재")
    m2.metric("경보", f"{ICON[lv]} {lv}")
    m3.metric("80% 구간(근사)", f"{s['value'] + lo_off:.1f} ~ {s['value'] + hi_off:.1f}")

# ── 3. 백테스트 ──────────────────────────────────────
with tab_bt:
    bt = pd.DataFrame(D["backtest"])
    bt["월"] = bt["ym"].map(ym_label)
    st.write("매달 그 직전까지의 자료로만 다시 학습해 다음 달을 예보한 결과 (2021.01~2026.08, 68개월).")
    st.line_chart(
        bt.set_index("월")[["actual", "model", "portal", "warn"]].rename(
            columns={"actual": "실제", "model": "계절 릿지", "portal": "공모전 모델", "warn": "주의 기준"}
        ),
        height=420,
    )
    aw = D["alert"]["all_warn"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("주의 경보 정밀도", f"{aw['정밀도']:.0%}")
    k2.metric("재현율", f"{aw['재현율']:.0%}")
    k3.metric("적중 / 오경보 / 미탐지", f"{aw['적중']} / {aw['오경보']} / {aw['미탐지']}")
    k4.metric("무작위 기준", f"{D['alert']['base_rate']['warn']:.0%}")

    bt["예보 경보"] = bt["model"].ge(bt["warn"])
    bt["실제 초과"] = bt["actual"].ge(bt["warn"])
    bt["판정"] = [
        "적중" if p and a else "오경보" if p else "미탐지" if a else "정상"
        for p, a in zip(bt["예보 경보"], bt["실제 초과"])
    ]
    with st.expander("월별 표"):
        st.dataframe(
            bt[["월", "actual", "model", "portal", "warn", "판정"]].rename(
                columns={"actual": "실제", "model": "계절 릿지", "portal": "공모전 모델", "warn": "주의 기준"}
            ),
            hide_index=True, use_container_width=True,
        )

# ── 4. 모델 비교 ─────────────────────────────────────
with tab_lb:
    lb = pd.DataFrame(D["leaderboard"]).sort_values("VAL_MAE")
    st.write("검증 구간(2021~2024) MAE로만 모델을 골랐고, 시험 구간(2025.01~2026.08)은 한 번만 평가했습니다.")
    st.bar_chart(lb.set_index("name")["VAL_MAE"], horizontal=True, height=640)
    st.dataframe(
        lb.rename(columns={"name": "모델", "VAL_MAE": "검증 MAE", "TEST_MAE": "시험 MAE",
                           "VAL_RMSE": "검증 RMSE", "TEST_RMSE": "시험 RMSE"}),
        hide_index=True, use_container_width=True,
    )

# ── 5. 항만 재배치 ───────────────────────────────────
with tab_port:
    years = sorted({r["year"] for r in D["ports"]})
    yr = st.radio("연도", years, index=len(years) - 1, horizontal=True)
    pt = pd.DataFrame([r for r in D["ports"] if r["year"] == yr]).sort_values("empty_net_inflow")
    st.write("양수 = 빈 컨테이너가 모자라 해외에서 들여오는 항만(부족), 음수 = 남아서 내보내는 항만(잉여).")
    st.bar_chart(pt.set_index("port")["empty_net_inflow"], horizontal=True, height=420)
    ra = next(r for r in D["realloc"] if r["year"] == yr)
    share = st.slider("남는 물량 중 미리 옮기는 비율(%)", 0, 100, 10, 5)
    st.metric("재배치 가능 물량(천TEU)", f"{ra['max_transfer'] * share / 100:,.1f}",
              help=f"min(부족 {ra['shortage']}, 잉여 {ra['surplus']}) × 비율")

st.divider()
st.caption("자료: BPA 컨테이너 포털(OCR, KOSIS와 37개월 대조 최대 차이 0.5) · KOSIS 항만별 공컨테이너 · SCFI")
