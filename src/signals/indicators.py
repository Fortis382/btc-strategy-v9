# src/signals/indicators.py
from __future__ import annotations
import polars as pl

def ema(col: pl.Expr, n: int) -> pl.Expr:
    # Polars ewm_mean: alpha / adjust / min_periods / ignore_nulls 만 지원 (bias 없음)
    alpha = 2 / (n + 1)
    return col.ewm_mean(alpha=alpha, adjust=False)

def rsi(close: pl.Expr, n: int) -> pl.Expr:
    diff = close.diff()
    gain = pl.when(diff > 0).then(diff).otherwise(0.0)
    loss = pl.when(diff < 0).then(-diff).otherwise(0.0)
    avg_gain = gain.ewm_mean(alpha=1/n, adjust=False)
    avg_loss = loss.ewm_mean(alpha=1/n, adjust=False)
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))

def true_range(h: pl.Expr, l: pl.Expr, c: pl.Expr) -> pl.Expr:
    prev_c = c.shift(1)
    return pl.max_horizontal([(h - l), (h - prev_c).abs(), (l - prev_c).abs()])

def atr(h: pl.Expr, l: pl.Expr, c: pl.Expr, n: int) -> pl.Expr:
    tr = true_range(h, l, c)
    return tr.ewm_mean(alpha=1/n, adjust=False)

def adx(h: pl.Expr, l: pl.Expr, c: pl.Expr, n: int) -> pl.Expr:
    up = h - h.shift(1)
    dn = l.shift(1) - l
    plus_dm  = pl.when((up > dn) & (up > 0)).then(up).otherwise(0.0)
    minus_dm = pl.when((dn > up) & (dn > 0)).then(dn).otherwise(0.0)
    tr = true_range(h, l, c)
    atr_ = tr.ewm_mean(alpha=1/n, adjust=False)
    plus_di  = 100 * (plus_dm.ewm_mean(alpha=1/n, adjust=False) / (atr_ + 1e-12))
    minus_di = 100 * (minus_dm.ewm_mean(alpha=1/n, adjust=False) / (atr_ + 1e-12))
    dx = 100 * ((plus_di - minus_di).abs() / ((plus_di + minus_di) + 1e-12))
    return dx.ewm_mean(alpha=1/n, adjust=False)

def add_indicators(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    e1, e2 = cfg["indicators"]["ema"]
    rsi_n  = cfg["indicators"]["rsi"]
    atr_n  = cfg["indicators"]["atr"]
    adx_n  = cfg["indicators"]["adx"]
    slope_norm = cfg["indicators"]["ema_slope_norm"]

    out = (
        df.lazy()
          .with_columns([
              ema(pl.col("close"), e1).alias(f"ema{e1}"),
              ema(pl.col("close"), e2).alias(f"ema{e2}"),
              rsi(pl.col("close"), rsi_n).alias(f"rsi{rsi_n}"),
              atr(pl.col("high"), pl.col("low"), pl.col("close"), atr_n).alias(f"atr{atr_n}"),
              adx(pl.col("high"), pl.col("low"), pl.col("close"), adx_n).alias(f"adx{adx_n}")
          ])
          .with_columns([
              (pl.col(f"ema{e1}").diff() / pl.col(f"ema{e1}").shift(1)).alias("ema21_slope_raw"),
              ((pl.col("close") - pl.col(f"ema{e1}")) / pl.col(f"ema{e1}")).alias("ema21_bias")
          ])
          .with_columns([
              (pl.col("ema21_slope_raw") / slope_norm).clip(-3, 3).alias("ema21_slope_n"),
              ((pl.col(f"rsi{rsi_n}") - 50) / 50).clip(-1, 1).alias("rsi_n"),
              (pl.col(f"atr{atr_n}") / pl.col("close")).alias("atr_p"),
              (pl.col(f"adx{adx_n}") / 50 - 1).clip(-1, 1).alias("adx_n")
          ])
          .collect()
    )
    return out
