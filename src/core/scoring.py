# src/core/scoring.py
from __future__ import annotations
import polars as pl

_EPS = 1e-12

def _clip01_to_pm1(expr: pl.Expr) -> pl.Expr:
    # map 0..100 -> -1..+1 by x/50 - 1
    return (expr / 50.0) - 1.0

def score_and_gate(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    w = cfg["scoring"]["weights"]
    g = cfg["gates"]

    # --- features (all in -1..+1) ---
    slope_n = pl.col("ema21_slope_n").clip(-1, 1)
    rsi_n   = _clip01_to_pm1(pl.col("rsi14")).clip(-1, 1)  # derive rsi_n from rsi14 if not present
    adx_n   = _clip01_to_pm1(pl.col("adx14")).clip(-1, 1)

    # volatility score: penalize high ATR%
    vol_ref = df["atr_p"].quantile(0.90)
    vol_sc  = (1.0 - (pl.col("atr_p") / (vol_ref + _EPS))).clip(-1.0, 1.0)

    score = (
        w["trend"]      * slope_n +
        w["momentum"]   * rsi_n +
        w["volatility"] * vol_sc +
        w["regime"]     * adx_n
    ) / (w["trend"] + w["momentum"] + w["volatility"] + w["regime"] or 1.0)

    # --- gates ---
    conds = []
    if g.get("use_adx_gate", True):
        conds.append(adx_n >= (_clip01_to_pm1(pl.lit(float(g.get("adx_min", 20))))))
    if g.get("use_trend_gate", True):
        conds.append(pl.col("ema21_slope_n") >= pl.lit(float(g.get("ema_slope_min", 0.06))))
    if g.get("use_range_gate", False):
        conds.append(((pl.col("high") - pl.col("low")) / (pl.col("atr14_abs") + _EPS)) >= pl.lit(float(g.get("min_range_atr", 0.6))))

    gate_ok = pl.all_horizontal(conds) if conds else pl.lit(True)

    return df.with_columns([
        score.clip(-1.0, 1.0).alias("score"),
        gate_ok.alias("gate_ok"),
    ])
