# src/core/scoring.py
from __future__ import annotations
import polars as pl

def score_and_gate(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    w = cfg.get("scoring", {}).get("weights", {})
    g = cfg.get("gates", {})
    ind = cfg.get("indicators", {})

    # ----- 스코어 구성요소 (Expr로 통일) -----
    trend_expr = pl.col("ema21_slope_n").clip(-1, 1)
    mom_expr   = pl.col("rsi_n").clip(-1, 1)

    # vol: 과열 패널티. q90은 스칼라(float)로 미리 뽑아 Expr에 끼운다.
    try:
        q90 = float(df["atr_p"].quantile(0.9)) if "atr_p" in df.columns and df.height else 1.0
    except Exception:
        q90 = 1.0
    if not q90 or q90 != q90:
        q90 = 1.0
    vol_expr = (1 - (pl.col("atr_p") / q90)).clip(-1, 1)

    reg_expr = pl.col("adx_n").clip(-1, 1)

    denom = float(w.get("trend", 1)) + float(w.get("momentum", 1)) + float(w.get("volatility", 1)) + float(w.get("regime", 1))
    if denom <= 0:
        denom = 1.0

    score_expr = (
        float(w.get("trend", 1))      * trend_expr +
        float(w.get("momentum", 1))   * mom_expr   +
        float(w.get("volatility", 1)) * vol_expr   +
        float(w.get("regime", 1))     * reg_expr
    ) / denom
    score_expr = score_expr.clip(-1, 1).alias("score")

    # ----- 게이트 조건 (Expr로 통일) -----
    use_adx   = bool(g.get("use_adx_gate", True))
    use_trend = bool(g.get("use_trend_gate", True))
    use_range = bool(g.get("use_range_gate", True))

    adx_min  = float(g.get("adx_min", 26.0))          # 0..100
    ema_min  = float(g.get("ema_slope_min", 0.05))    # -1..1 정규화 구간
    k_range  = float(g.get("min_range_atr", 0.58))    # (H-L) >= k*ATR
    adx_buf  = float(g.get("adx_strict_buffer", 0.20))

    adx_thr_norm = adx_min / 50.0 - 1.0               # [-1..1]로 정규화된 임계

    atr_col = f"atr{ind.get('atr', 14)}"

    cond_adx_expr   = (pl.col("adx_n") >= adx_thr_norm)                  if use_adx   else pl.lit(True)
    cond_trend_expr = (pl.col("ema21_slope_n") >= ema_min)               if use_trend else pl.lit(True)
    cond_range_expr = ((pl.col("high") - pl.col("low")) >= k_range * pl.col(atr_col)) if use_range else pl.lit(True)

    sum_true_expr = (
        cond_adx_expr.cast(pl.UInt8) +
        cond_trend_expr.cast(pl.UInt8) +
        cond_range_expr.cast(pl.UInt8)
    )

    # ADX가 약하면 3-of-3, 강하면 2-of-3  →  k_needed = 2 + need_strict(0/1)
    need_strict_expr = (pl.col("adx_n") < (adx_thr_norm + adx_buf)).cast(pl.UInt8)
    k_needed_expr    = pl.lit(2) + need_strict_expr

    gate_ok_expr = (sum_true_expr >= k_needed_expr).alias("gate_ok")

    # ----- 컬럼 추가 -----
    return df.with_columns([score_expr, gate_ok_expr])
