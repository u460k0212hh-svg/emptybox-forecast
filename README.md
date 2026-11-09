# 부산항 공컨테이너 한 달 앞 예보

2026 데이터+AI 혁신 챌린지 출품작(「수출입 불균형 지표를 활용한 공컨테이너 수급 예측 및 항만별 선제적 재배치」)을 다시 검증하고 발전시킨 코드입니다.

- **목표 변수**: 부산항 공컨테이너 순유입 = 공컨 수입 − 공컨 수출 (천TEU, 환적 제외)
- **최종 모델**: 계절 릿지 = 전월 적컨 불균형 + 최근 6개월 평균 순유입 + 월 효과 (α=1)
- **평가**: 매달 다시 학습하는 확장 윈도. 검증 구간 2021.01–2024.12에서만 모델을 고르고, 시험 구간 2025.01–2026.08은 한 번만 평가

| | 검증 MAE | 시험 MAE |
|---|---|---|
| 계절 릿지 (선정) | 12.49 | 10.99 |
| 공모전 모델 (8변수 선형) | 18.30 | 12.13 |
| 학습기간 평균 | 15.00 | 11.97 |
| 전월값 | 20.06 | 17.85 |

## 실행

```bash
pip install -r requirements.txt
PYTHONPATH=src python3 scripts/run_experiments.py   # 25개 후보 실험 → results/
PYTHONPATH=src python3 scripts/export_app_data.py   # 운영용 재학습, 다음 달 예보 → app/data.json
python3 scripts/build_app.py                        # 대시보드 → app/index.html
```

`run_experiments.py`는 처음에 공모전(Brightics) 결과(LR 12.05, RF 12.94, 전월값 17.85, 전년동월값 16.95)를 재현하는지 assert로 확인합니다. 약 1분 걸립니다.

## Streamlit 앱

```bash
streamlit run streamlit_app.py
```

Streamlit Community Cloud 배포: 이 폴더를 GitHub 저장소에 올리고 share.streamlit.io에서 **Create app** → 저장소·브랜치 선택 → Main file path `streamlit_app.py` → Deploy. 앱은 `app/data.json`만 읽으므로, 새 달 자료가 들어오면 `export_app_data.py`를 다시 돌려 data.json을 커밋하면 됩니다.

## 구성

```
data/raw/          BPA 그래프 OCR 결과, SCFI, KOSIS 항만별 공컨 원자료
data/processed/    월별 데이터셋 (2016.09–2026.08)
tools/             BPA 캡처 OCR(ocr_bpa.py, run_ocr.py)과 데이터셋 생성
src/emptybox/      data · models · evaluate · alert · reallocation
scripts/           실험, 앱 데이터 내보내기, 대시보드 빌드
results/           metrics.json, 매달 예보 결과, 경보 백테스트
app/               대시보드(정적 HTML)
```

## 참고

- ARIMA·SARIMAX(조건부 제곱합)와 Holt-Winters는 statsmodels 없이 numpy·scipy로 직접 구현했습니다 (`src/emptybox/models.py`).
- 부산항 월별 자료는 BPA 컨테이너 포털 그래프 캡처를 OCR로 읽은 뒤 손으로 보정했습니다. KOSIS와 겹치는 37개월 × 4계열을 대조한 최대 차이는 0.5천TEU입니다.
- 경보 기준: 학습 기간 평균 + 0.5σ(주의), + 1σ(위험).
