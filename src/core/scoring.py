# src/core/scoring.py
from __future__ import annotations
import polars as pl

def _col(df: pl.DataFrame, name: str, default: float = 0.0) -> pl.Series:
    """df[name]이 없으면 길이 맞는 기본 시리즈 리턴."""
    if name in df.columns:
        return df[name]
    n = len(df)
    return pl.Series(name, [default] * n)

def score_and_gate(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    w = cfg.get("scoring", {}).get("weights", {})
    g = cfg.get("gates", {})

    # ---- 필요한 컬럼 확보(없으면 안전 기본값) ----
    ema_slope = _col(df, "ema21_slope_n")
    rsi_n     = _col(df, "rsi_n")
    adx_n     = _col(df, "adx_n")
    atr_p     = _col(df, "atr_p")
    high      = _col(df, "high")
    low       = _col(df, "low")
    atr_col   = f"atr{cfg['indicators']['atr']}"
    atr       = _col(df, atr_col)

    # ---- 점수(score) 계산: 기존 로직을 그대로 보존 ----
    trend = ema_slope.clip(-1, 1)
    mom   = rsi_n.clip(-1, 1)

    # 과열 페널티: 1 - (ATR% / q90), 0으로 나누기/비정상 q90 보호
    try:
        q90 = float(atr_p.quantile(0.9)) if len(atr_p) else 1.0
    except Exception:
        q90 = 1.0
    if not q90 or q90 != q90:  # 0 또는 NaN 방지
        q90 = 1.0
    vol   = (1 - (atr_p / q90)).clip(-1, 1)

    reg   = adx_n.clip(-1, 1)

    denom = (
        float(w.get("trend", 1))
        + float(w.get("momentum", 1))
        + float(w.get("volatility", 1))
        + float(w.get("regime", 1))
    )
    if denom <= 0:
        denom = 1.0

    score = (
        float(w.get("trend", 1))      * trend
        + float(w.get("momentum", 1)) * mom
        + float(w.get("volatility", 1))* vol
        + float(w.get("regime", 1))   * reg
    ) / denom
    score = score.clip(-1, 1)

    # ---- 게이트: 옵션별 on/off 가능 ----
    use_adx   = bool(g.get("use_adx_gate", True))
    use_trend = bool(g.get("use_trend_gate", True))
    use_range = bool(g.get("use_range_gate", False))  # 새로 추가: 범위 게이트

    # 임계값
    adx_min = float(g.get("adx_min", 20.0))       # ADX(0..100) 스케일
    adx_thr_norm = adx_min / 50.0 - 1.0           # [-1, +1] 정규화된 adx_n과 비교
    ema_min = float(g.get("ema_slope_min", 0.0))  # [-1, +1]에서의 최소 기울기
    k_range = float(g.get("min_range_atr", 0.0))  # (high-low) >= k_range * ATR

    # 각 게이트(없으면 True로 대체)
    chop_ok  = (adx_n >= adx_thr_norm) if use_adx else pl.Series([True] * len(df))
    trend_ok = (ema_slope >= ema_min)  if use_trend else pl.Series([True] * len(df))
    range_ok = ((high - low) >= k_range * atr) if use_range else pl.Series([True] * len(df))

    gate_ok = (chop_ok & trend_ok & range_ok).fill_null(False)

    return df.with_columns([
        pl.Series("score", score),
        pl.Series("gate_ok", gate_ok),
    ])
