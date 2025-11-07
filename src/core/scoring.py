# src/core/scoring.py (전체 교체)
from __future__ import annotations
import polars as pl

def score_and_gate(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    w = cfg["scoring"]["weights"]
    g = cfg["gates"]
    adx_len = cfg["indicators"]["adx"]
    atr_len = cfg["indicators"]["atr"]

    # --- Sub-scores (±1 스케일 가정: indicators.py에서 보장) ---
    trend = pl.col("ema21_slope_n")
    mom   = pl.col("rsi_n")
    reg   = pl.col("adx_n")

    # q90을 DataFrame에서 스칼라로 추출 → Expr에 주입
    q90_val = float(df.get_column("atr_p").quantile(0.90))
    vol = (1.0 - (pl.col("atr_p") / (q90_val + 1e-12))).clip(-1.0, 1.0)

    denom = (w["trend"] + w["momentum"] + w["volatility"] + w["regime"]) or 1.0
    raw   = (w["trend"]*trend + w["momentum"]*mom + w["volatility"]*vol + w["regime"]*reg) / denom

    # tails 확장: 간단 클립(엔진 호환 안전). 필요시 tanh(raw/0.6)로 교체 가능.
    score = (raw / 0.6).clip(-1.0, 1.0).alias("score")

    # --- Gates (단위 일치) ---
    adx_ok   = pl.lit(True) if not g.get("use_adx_gate", True)   else (pl.col(f"adx{adx_len}") >= g["adx_min"])
    slope_ok = pl.lit(True) if not g.get("use_trend_gate", True) else (pl.col("ema21_slope_pct").abs() >= g["ema_slope_min"])
    range_ok = pl.lit(True) if not g.get("use_range_gate", False) else (
        ((pl.col("high") - pl.col("low")) / (pl.col(f"atr{atr_len}_abs") + 1e-12)) >= g.get("min_range_atr", 0.6)
    )

    gate_ok = (adx_ok & slope_ok & range_ok).alias("gate_ok")
    return df.with_columns([score, gate_ok])
