# src/signals/indicators.py
from __future__ import annotations
import polars as pl

_EPS = 1e-12

def ema(col: pl.Expr, length: int) -> pl.Expr:
    alpha = 2.0 / (length + 1.0)
    return col.ewm_mean(alpha=alpha, adjust=True)

def rsi(close: pl.Expr, length: int) -> pl.Expr:
    d = close.diff()
    gain = pl.when(d > 0).then(d).otherwise(0.0)
    loss = pl.when(d < 0).then(-d).otherwise(0.0)

    avg_gain = gain.ewm_mean(alpha=1.0/length, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1.0/length, adjust=False)

    # Expr에는 clip_min이 없음 → max_horizontal로 0 이하 방지
    rs = pl.max_horizontal([avg_gain / (avg_loss + _EPS), pl.lit(0.0)])
    return 100.0 - (100.0 / (1.0 + rs))

def atr(h: pl.Expr, l: pl.Expr, c: pl.Expr, length: int) -> pl.Expr:
    tr1 = (h - l)
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pl.max_horizontal([tr1, tr2, tr3])
    return tr.ewm_mean(alpha=1.0/length, adjust=False)

def adx(h: pl.Expr, l: pl.Expr, c: pl.Expr, length: int) -> pl.Expr:
    up = h - h.shift(1)
    dn = l.shift(1) - l
    plus_dm  = pl.when((up > dn) & (up > 0)).then(up).otherwise(0.0)
    minus_dm = pl.when((dn > up) & (dn > 0)).then(dn).otherwise(0.0)

    tr1 = (h - l)
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pl.max_horizontal([tr1, tr2, tr3])
    atr_sm = tr.ewm_mean(alpha=1.0/length, adjust=False)

    pdi = 100.0 * (plus_dm.ewm_mean(alpha=1.0/length, adjust=False)  / (atr_sm + _EPS))
    mdi = 100.0 * (minus_dm.ewm_mean(alpha=1.0/length, adjust=False) / (atr_sm + _EPS))
    dx  = 100.0 * ((pdi - mdi).abs() / ((pdi + mdi).abs() + _EPS))
    return dx.ewm_mean(alpha=1.0/length, adjust=False)

def add_indicators(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    e1, e2   = cfg["indicators"]["ema"]
    rlen     = cfg["indicators"]["rsi"]
    alen     = cfg["indicators"]["atr"]
    dlen     = cfg["indicators"]["adx"]
    slope_n_pct = float(cfg["indicators"].get("ema_slope_norm", 0.05))  # “봉당 %”

    lazy = df.lazy().with_columns([
        ema(pl.col("close"), e1).alias(f"ema{e1}"),
        ema(pl.col("close"), e2).alias(f"ema{e2}"),
        rsi(pl.col("close"), rlen).alias(f"rsi{rlen}"),
        atr(pl.col("high"), pl.col("low"), pl.col("close"), alen).alias(f"atr{alen}_abs"),
        adx(pl.col("high"), pl.col("low"), pl.col("close"), dlen).alias(f"adx{dlen}"),
    ]).with_columns([
        # ATR %
        (pl.col(f"atr{alen}_abs") / pl.col("close")).alias("atr_p"),

        # EMA21 slope: (% per bar, 기반은 EMA 자체의 직전 대비)
        ((pl.col(f"ema{e1}") - pl.col(f"ema{e1}").shift(1)) / (pl.col(f"ema{e1}").shift(1) + _EPS) * 100.0)
            .alias("ema21_slope_pct"),

        # 정규화(설정값: “봉당 %”):  (slope% / slope_n_pct) → [-1, +1]
        ((pl.col("ema21_slope_pct") / slope_n_pct).clip(-1.0, 1.0)).alias("ema21_slope_n"),

        # RSI 정규화: [-1,+1]
        ((pl.col(f"rsi{rlen}") - 50.0) / 50.0).clip(-1.0, 1.0).alias("rsi_n"),

        # ADX 정규화: 기준 20을 0으로 센터링 (20→0, 35→+0.5, 50→+1)
        (((pl.col(f"adx{dlen}") - 20.0) / 30.0).clip(-1.0, 1.0)).alias("adx_n"),
    ])
    return lazy.collect()
