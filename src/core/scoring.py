# src/core/scoring.py
from __future__ import annotations
import polars as pl

__all__ = ["score_and_gate"]

def score_and_gate(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """
    - 스코어: trend(ema21_slope_n), momentum(rsi_n), volatility(atr_p의 q90 과열 패널티), regime(adx_n)
      모두 -1..+1 클램프 후 가중합/정규화
    - 게이트: ADX/Trend/Range 2-of-3 (단, ADX 약하면 3-of-3로 격상)
    - 반환: score(float), gate_ok(bool)
    """
    w = cfg.get("scoring", {}).get("weights", {})
    g = cfg.get("gates", {})
    ind = cfg.get("indicators", {})

    # -------- 스코어 구성요소 (모두 Expr) --------
    trend = pl.col("ema21_slope_n").clip(-1, 1)
    mom   = pl.col("rsi_n").clip(-1, 1)

    # q90(과열 기준치) 스칼라 추출 → Expr에 주입
    try:
        q90_val = float(df.select(pl.col("atr_p").quantile(0.90)).item())
        q90_val = q90_val if q90_val not in (None, 0.0) else 1.0
    except Exception:
        q90_val = 1.0
    vol = (1 - (pl.col("atr_p") / q90_val)).clip(-1, 1)

    reg = pl.col("adx_n").clip(-1, 1)

    w_trend = float(w.get("trend",     1.0))
    w_mom   = float(w.get("momentum",  1.0))
    w_vol   = float(w.get("volatility",1.0))
    w_reg   = float(w.get("regime",    1.0))
    denom   = (w_trend + w_mom + w_vol + w_reg) or 1.0

    score_expr = ((w_trend*trend + w_mom*mom + w_vol*vol + w_reg*reg) / denom).clip(-1, 1)

    # -------- 게이트 (2-of-3, ADX 약하면 3-of-3) --------
    use_adx   = bool(g.get("use_adx_gate",   True))
    use_trend = bool(g.get("use_trend_gate", True))
    use_range = bool(g.get("use_range_gate", False))

    # adx_n은 [-1, +1] 정규화 전제. adx_min은 0..100 값 → 변환
    adx_min_raw   = float(g.get("adx_min", 20))
    adx_thr_norm  = adx_min_raw/50.0 - 1.0         # 예: 20 → -0.6
    adx_strict_buf= 0.10                           # 약간 더 약하면 3-of-3로 강화

    ema_min       = float(g.get("ema_slope_min", 0.06))
    atr_n         = int(ind.get("atr", 14))
    atr_col       = f"atr{atr_n}"
    k_range       = float(g.get("min_range_atr", 0.60))

    cond_adx   = (pl.col("adx_n")           >= adx_thr_norm) if use_adx   else pl.lit(True)
    cond_trend = (pl.col("ema21_slope_n")   >= ema_min)       if use_trend else pl.lit(True)
    cond_range = ((pl.col("high") - pl.col("low")) >= k_range*pl.col(atr_col)) if use_range else pl.lit(True)

    sum_true = (cond_adx.cast(pl.UInt8) + cond_trend.cast(pl.UInt8) + cond_range.cast(pl.UInt8))

    need_strict = pl.col("adx_n") < (adx_thr_norm + adx_strict_buf)
    k_needed = pl.when(need_strict).then(pl.lit(3)).otherwise(pl.lit(2))

    gate_ok_expr = (sum_true >= k_needed)

    # -------- 컬럼 추가 --------
    return df.with_columns([
        score_expr.alias("score"),
        gate_ok_expr.alias("gate_ok"),
    ])
