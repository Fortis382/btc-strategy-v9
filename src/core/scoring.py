# src/core/scoring.py
from __future__ import annotations
import polars as pl

def score_and_gate(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    w = cfg["scoring"]["weights"]
    g = cfg["gates"]

    trend = pl.col("ema21_slope_n").clip(-1, 1)
    mom   = pl.col("rsi_n").clip(-1, 1)
    vol_q90 = pl.col("atr_p").quantile(0.9)
    vol   = (1.0 - (pl.col("atr_p") / (vol_q90 + 1e-12))).clip(-1.0, 1.0)
    reg   = pl.col("adx_n").clip(-1, 1)

    denom = (w["trend"] + w["momentum"] + w["volatility"] + w["regime"]) or 1.0
    score_expr = (w["trend"]*trend + w["momentum"]*mom + w["volatility"]*vol + w["regime"]*reg) / denom
    score_expr = score_expr.clip(-1.0, 1.0)

    use_adx   = bool(g.get("use_adx_gate", True))
    use_trend = bool(g.get("use_trend_gate", True))
    use_range = bool(g.get("use_range_gate", False))

    adx_thr_norm = (g["adx_min"] / 50.0) - 1.0
    trend_thr    = g["ema_slope_min"]

    chop_ok  = pl.when(pl.lit(use_adx)).then(pl.col("adx_n") >= adx_thr_norm).otherwise(pl.lit(True))
    trend_ok = pl.when(pl.lit(use_trend)).then(pl.col("ema21_slope_n") >= trend_thr).otherwise(pl.lit(True))
    range_ok = pl.when(pl.lit(use_range)).then(
        (pl.col("high") - pl.col("low")) >= (float(g.get("min_range_atr", 0.6)) * pl.col("atr14_abs"))
    ).otherwise(pl.lit(True))

    gate_ok_expr = (chop_ok & trend_ok & range_ok).fill_null(False)

    return df.with_columns([
        score_expr.alias("score"),
        gate_ok_expr.alias("gate_ok"),
    ])
