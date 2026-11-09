"""조기경보: 학습기간 평균·표준편차로 정상/주의/위험 기준을 정하고 예측값으로 판정한다."""
from __future__ import annotations
import numpy as np
import pandas as pd

def thresholds(train_y, k_warn=0.5, k_danger=1.0):
    mu, sd = float(np.mean(train_y)), float(np.std(train_y, ddof=1))
    return {"mean": mu, "sd": sd, "warn": mu+k_warn*sd, "danger": mu+k_danger*sd}

def level(v, th):
    return "위험" if v >= th["danger"] else ("주의" if v >= th["warn"] else "정상")

def confusion(y, pred, th, key="warn"):
    a, p = np.asarray(y) >= th[key], np.asarray(pred) >= th[key]
    return {"적중": int((a & p).sum()), "미탐지": int((a & ~p).sum()), "오경보": int((~a & p).sum()), "정상확인": int((~a & ~p).sum())}
