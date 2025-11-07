# src/signals/indicators.py
from __future__ import annotations
import polars as pl

_EPS = 1e-12

def ema(expr: pl.Expr, span: int) -> pl.Expr:
    alpha = 2.0 / (span + 1.0)
    return expr.ewm_mean(alpha=alpha)

def rsi(close: pl.Expr, length: int) -> pl.Expr:
    d = close.diff()
    gain = pl.when(d > 0).then(d).otherwise(0.0).ewm_mean(alpha=1.0/length)
    loss = pl.when(d < 0).then(-d).otherwise(0.0).ewm_mean(alpha=1.0/length)
    rs = gain / (loss + _EPS)
    return 100.0 - (100.0 / (1.0 + pl.max_horizontal(rs, pl.lit(0.0))))

def atr_abs(high: pl.Expr, low: pl.Expr, close: pl.Expr, length: int) -> pl.Expr:
    tr = pl.max_horizontal(
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    )
    return tr.ewm_mean(alpha=1.0/length)

def adx(high: pl.Expr, low: pl.Expr, close: pl.Expr, length: int) -> pl.Expr:
    up  = high - high.shift(1)
    dn  = low.shift(1) - low
    plus_dm  = pl.when((up > dn) & (up > 0)).then(up).otherwise(0.0)
    minus_dm = pl.when((dn > up) & (dn > 0)).then(dn).otherwise(0.0)

    tr = pl.max_horizontal(
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    )
    alpha = 1.0/length
    atr = tr.ewm_mean(alpha=alpha) + _EPS
    pdi = (plus_dm .ewm_mean(alpha=alpha) * 100.0) / atr
    mdi = (minus_dm.ewm_mean(alpha=alpha) * 100.0) / atr
    dx  = ((pdi - mdi).abs() / (pdi + mdi + _EPS)) * 100.0
    return dx.ewm_mean(alpha=alpha)

def add_indicators(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    e1, e2     = cfg["indicators"]["ema"]
    rlen       = cfg["indicators"]["rsi"]
    alen       = cfg["indicators"]["atr"]
    dxlen      = cfg["indicators"]["adx"]
    slope_norm = float(cfg["indicators"]["ema_slope_norm"] or 0.05)

    lf = df.lazy().with_columns([
        ema(pl.col("close"), e1).alias("ema21"),
        ema(pl.col("close"), e2).alias("ema55"),
        rsi(pl.col("close"), rlen).alias("rsi14"),
        adx(pl.col("high"),  pl.col("low"),  pl.col("close"), dxlen).alias("adx14"),
    ])

    # ---- ATR을 먼저 Expr로 잡은 뒤 두 이름으로 export(호환) ----
    atr_expr = atr_abs(pl.col("high"), pl.col("low"), pl.col("close"), alen)

    # ---- 파생/정규화: 식을 변수로 먼저 만든 뒤 재사용 ----
    slope_expr = ((pl.col("ema21") / pl.col("ema21").shift(1)) - 1.0) * 100.0
    bias_expr  = (pl.col("close") / pl.col("ema21") - 1.0)
    rsi_n_expr = ((pl.col("rsi14") - 50.0) / 50.0).clip(-1.0, 1.0)
    atr_p_expr = (atr_expr / (pl.col("close") + _EPS))
    adx_n_expr = ((pl.col("adx14") / 50.0) - 1.0).clip(-1.0, 1.0)

    lf = lf.with_columns([
        atr_expr.alias("atr14_abs"),
        atr_expr.alias("atr14"),  # ← 백워드 호환용 별칭
        slope_expr.alias("ema21_slope_pct"),
        (slope_expr / slope_norm).clip(-1.0, 1.0).alias("ema21_slope_n"),
        bias_expr.alias("ema21_bias"),
        rsi_n_expr.alias("rsi_n"),
        atr_p_expr.alias("atr_p"),
        adx_n_expr.alias("adx_n"),
    ])

    return lf.collect()
