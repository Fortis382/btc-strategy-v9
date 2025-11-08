# src/signals/indicators.py
from __future__ import annotations
import polars as pl

_EPS = 1e-12

def _clamp(expr: pl.Expr, lo: float, hi: float) -> pl.Expr:
    return pl.min_horizontal(pl.max_horizontal(expr, pl.lit(lo)), pl.lit(hi))

def _clip_lower(expr: pl.Expr, lo: float) -> pl.Expr:
    return pl.max_horizontal(expr, pl.lit(lo))

def add_indicators(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """
    v9.4 완전 호환 지표 계산
    - 5-factor: trend/momentum/volatility/participation/location
    - 기본: ema{fast}, ema{slow}, rsi{rlen}, atr{alen}_abs, adx{dlen}
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

    # 2) RSI
    diff = pl.col("close").diff()
    gain = pl.when(diff > 0).then(diff).otherwise(0.0)
    loss = pl.when(diff < 0).then(-diff).otherwise(0.0)
    avg_gain = gain.ewm_mean(span=rlen)
    avg_loss = loss.ewm_mean(span=rlen) + _EPS
    rs = avg_gain / avg_loss
    rsi = (100.0 - (100.0 / (1.0 + _clip_lower(rs, 0.0)))).alias(f"rsi{rlen}")

    # 3) ATR (절대)
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

    # 5) EMA slope (pct)
    ema_fast_col = pl.col(f"ema{ema_fast}")
    ema_slope_pct_expr = ((ema_fast_col / ema_fast_col.shift(1) - 1.0) * 100.0).fill_null(0.0)
    lf = lf.with_columns([ema_slope_pct_expr.alias("ema21_slope_pct")])

    # 6) Participation (거래량 Z-score) ✅ 신규
    vol_ma = pl.col("volume").rolling_mean(window_size=50)
    vol_std = pl.col("volume").rolling_std(window_size=50) + _EPS
    participation_raw = ((pl.col("volume") - vol_ma) / vol_std).alias("participation_raw")
    lf = lf.with_columns([participation_raw])

    # 7) Location (Close 위치 in 20봉 range) ✅ 신규
    high_20 = pl.col("high").rolling_max(window_size=20)
    low_20 = pl.col("low").rolling_min(window_size=20)
    location_raw = (
        (pl.col("close") - low_20) / (high_20 - low_20 + _EPS)
    ).alias("location_raw")
    lf = lf.with_columns([location_raw])

    # 8) 정규화 (모든 factor를 [-1, 1])
    lf = lf.with_columns([
        # ✅ 수정: tanh로 soft clipping (v9.4 문서 기준)
        (ema_slope_pct_expr / (slope_norm + _EPS)).tanh().alias("ema21_slope_n"),
        
        ((pl.col(f"atr{alen}_abs") / (pl.col("close") + _EPS)) * 100.0).alias("atr_p"),
        
        _clamp((pl.col(f"rsi{rlen}") - 50.0) / 50.0, -1.0, 1.0).alias("rsi_n"),
        
        # ✅ 수정: ADX 정규화 (0~100 → -1~1, 중심 25)
        _clamp((pl.col(f"adx{dlen}") - 25.0) / 25.0, -1.0, 1.0).alias("adx_n"),
        
        # ✅ 신규: participation 정규화
        _clamp(pl.col("participation_raw"), -1.0, 1.0).alias("participation_n"),
        
        # ✅ 신규: location 정규화 (0~1 → -1~1)
        _clamp(pl.col("location_raw") * 2.0 - 1.0, -1.0, 1.0).alias("location_n"),
    ])

    return lf.collect()