# src/signals/indicators.py
from __future__ import annotations
import polars as pl

_EPS = 1e-12

def _clamp(expr: pl.Expr, lo: float, hi: float) -> pl.Expr:
    # polars 전버전 호환용: Expr.clip 대체
    return pl.min_horizontal(pl.max_horizontal(expr, pl.lit(lo)), pl.lit(hi))

def _clip_lower(expr: pl.Expr, lo: float) -> pl.Expr:
    # polars 전버전 호환용: clip_min 대체
    return pl.max_horizontal(expr, pl.lit(lo))

def add_indicators(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """
    9.4v 계약 준수 컬럼 생성:
      - ema{fast}, ema{slow}
      - rsi{rlen}
      - atr{alen}_abs
      - adx{dlen}
      - ema21_slope_pct, ema21_slope_n, atr_p, rsi_n, adx_n
    주의: scoring/gates가 'ema21_*' 명칭을 기대하므로 fast=21 기준으로 이름 고정.
    """
    ema_fast, ema_slow = cfg["indicators"]["ema"]
    rlen = int(cfg["indicators"]["rsi"])
    alen = int(cfg["indicators"]["atr"])
    dlen = int(cfg["indicators"]["adx"])
    slope_norm = float(cfg["indicators"]["ema_slope_norm"])  # % per bar

    lf = df.lazy()

    # 1) EMA
    lf = lf.with_columns([
        pl.col("close").ewm_mean(span=ema_fast).alias(f"ema{ema_fast}"),
        pl.col("close").ewm_mean(span=ema_slow).alias(f"ema{ema_slow}"),
    ])

    # 2) RSI (구버전 호환: clip_min 대신 max_horizontal)
    diff = pl.col("close").diff()
    gain = pl.when(diff > 0).then(diff).otherwise(0.0)
    loss = pl.when(diff < 0).then(-diff).otherwise(0.0)
    avg_gain = gain.ewm_mean(span=rlen)
    avg_loss = loss.ewm_mean(span=rlen) + _EPS
    rs = avg_gain / avg_loss
    rs_pos = _clip_lower(rs, 0.0)
    rsi = (100.0 - (100.0 / (1.0 + rs_pos))).alias(f"rsi{rlen}")

    # 3) ATR(절대값)
    tr = pl.max_horizontal(
        (pl.col("high") - pl.col("low")),
        (pl.col("high") - pl.col("close").shift(1)).abs(),
        (pl.col("low")  - pl.col("close").shift(1)).abs(),
    )
    atr_abs = tr.ewm_mean(span=alen).alias(f"atr{alen}_abs")

    # 4) ADX
    up   = pl.col("high") - pl.col("high").shift(1)
    down = pl.col("low").shift(1) - pl.col("low")
    plus_dm  = pl.when((up > down) & (up > 0)).then(up).otherwise(0.0)
    minus_dm = pl.when((down > up) & (down > 0)).then(down).otherwise(0.0)
    tr_ewm   = tr.ewm_mean(span=dlen) + _EPS
    pdi = (plus_dm.ewm_mean(span=dlen)  / tr_ewm) * 100.0
    mdi = (minus_dm.ewm_mean(span=dlen) / tr_ewm) * 100.0
    dx  = ((pdi - mdi).abs() / ((pdi + mdi).abs() + _EPS)) * 100.0
    adx = dx.ewm_mean(span=dlen).alias(f"adx{dlen}")

    lf = lf.with_columns([rsi, atr_abs, adx])

    # 5) 파생/정규화 (9.4v 스코어 계약)
    ema_fast_col = pl.col(f"ema{ema_fast}")
    ema_slope_pct = ((ema_fast_col / ema_fast_col.shift(1) - 1.0) * 100.0).fill_null(0.0)

    lf = lf.with_columns([
        ema_slope_pct.alias("ema21_slope_pct"),  # 계약명 고정
        _clamp(pl.col("ema21_slope_pct") / (slope_norm + _EPS), -1.0, 1.0).alias("ema21_slope_n"),
        ((pl.col(f"atr{alen}_abs") / (pl.col("close") + _EPS)) * 100.0).alias("atr_p"),
        _clamp((pl.col(f"rsi{rlen}") - 50.0) / 50.0, -1.0, 1.0).alias("rsi_n"),
        _clamp(pl.col(f"adx{dlen}") / 50.0 - 1.0, -1.0, 1.0).alias("adx_n"),
    ])

    return lf.collect()
