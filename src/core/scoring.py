# src/core/scoring.py
from __future__ import annotations
import polars as pl

def score_and_gate(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    w = cfg["scoring"]["weights"]
    g = cfg["gates"]

    trend = df["ema21_slope_n"].clip(-1,1)
    mom   = df["rsi_n"].clip(-1,1)
    vol   = (1 - (df["atr_p"]/df["atr_p"].quantile(0.9))).clip(-1,1)  # 과열 패널티
    reg   = df["adx_n"].clip(-1,1)

    denom = sum([w["trend"], w["momentum"], w["volatility"], w["regime"]]) or 1.0
    score = (w["trend"]*trend + w["momentum"]*mom + w["volatility"]*vol + w["regime"]*reg)/denom
    score = score.clip(-1,1)

    chop_ok  = (df["adx_n"] >= (g["adx_min"]/50 - 1))
    trend_ok = (df["ema21_slope_n"] >= g["ema_slope_min"])

    return df.with_columns([
        pl.Series("score", score),
        pl.Series("gate_ok", chop_ok & trend_ok),
    ])
