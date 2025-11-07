# src/signals/indicators.py
from __future__ import annotations
import polars as pl

_EPS = 1e-12

def _clamp(expr: pl.Expr, lo: float, hi: float) -> pl.Expr:
    # polars 전버전 호환: clip(lower=,upper=) 대신 수평 max/min 사용
    return pl.min_horizontal(pl.max_horizontal(expr, pl.lit(lo)), pl.lit(hi))

def _clip_lower(expr: pl.Expr, lo: float) -> pl.Expr:
    # polars 전버전 호환: clip_min 대체
    return pl.max_horizontal(expr, pl.lit(lo))

def add_indicators(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """
    9.4v 계약 컬럼 보장:
      ema21_slope_pct, ema21_slope_n, atr_p, rsi_n, adx_n
    + 기본: ema{fast}, ema{slow}, rsi{rlen}, atr{alen}_abs, adx{dlen}
    (config는 ema: [21,55] 가정. fast=21 기준으로 'ema21_*' 네이밍 고정)
    """
    ema_fast, ema_slow = cfg["indicators"]["ema"]
    rlen = int(cfg["indicators"]["rsi"])
    alen = int(cfg["indicators"]["atr"])
    dlen = int(cfg["indicators"]["adx"])
    slope_norm = float(cfg["indicators"]["ema_slope_norm"])

    lf = df.lazy()

    # 1) EMA
    lf = lf.with_columns([
        pl.col("close").ewm_mean(span=ema_fast).alias(f"ema{ema_fast}"),
        pl.col("close").ewm_mean(span=ema_slow).alias(f"ema{ema_slow}"),
    ])

    # 2) RSI (버전 호환: clip_min 대신 max_horizontal)
    diff = pl.col("close").diff()
    gain = pl.when(diff > 0).then(diff).otherwise(0.0)
    loss = pl.when(diff < 0).then(-diff).otherwise(0.0)
    avg_gain = gain.ewm_mean(span=rlen)
    avg_loss = loss.ewm_mean(span=rlen) + _EPS
    rs = avg_gain / avg_loss
    rsi = (100.0 - (100.0 / (1.0 + _clip_lower(rs, 0.0)))).alias(f"rsi{rlen}")

    # 3) ATR(절대)
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

    # 1차 컬럼 주입
    lf = lf.with_columns([rsi, atr_abs, adx])

    # 5) 파생/정규화 — 같은 블록 내 alias 참조 금지! (표현식 재사용)
    ema_fast_col = pl.col(f"ema{ema_fast}")
    ema_slope_pct_expr = ((ema_fast_col / ema_fast_col.shift(1) - 1.0) * 100.0).fill_null(0.0)

    lf = lf.with_columns([
        # 먼저 pct를 물리 컬럼으로 만든 다음…
        ema_slope_pct_expr.alias("ema21_slope_pct"),
    ])

    # …그 다음 블록에서 해당 컬럼을 참조 (버전 안전)
    lf = lf.with_columns([
        _clamp(ema_slope_pct_expr / (slope_norm + _EPS), -1.0, 1.0).alias("ema21_slope_n"),
        ((pl.col(f"atr{alen}_abs") / (pl.col("close") + _EPS)) * 100.0).alias("atr_p"),
        _clamp((pl.col(f"rsi{rlen}") - 50.0) / 50.0, -1.0, 1.0).alias("rsi_n"),
        _clamp(pl.col(f"adx{dlen}") / 50.0 - 1.0, -1.0, 1.0).alias("adx_n"),
    ])

    return lf.collect()
