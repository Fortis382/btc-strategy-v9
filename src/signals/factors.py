# src/signals/factors.py
"""
5-Factor 계산 (v9.4 Section 4.2)
indicators.py의 고수준 래퍼
"""
from __future__ import annotations
import polars as pl
from typing import Dict, Any

def compute_factors_polars(
    df: pl.DataFrame,
    cfg: Dict[str, Any]
) -> pl.DataFrame:
    """
    5-Factor 계산 + 정규화
    
    Factors:
        1. Trend: EMA slope (tanh 정규화)
        2. Momentum: RSI (중심 50, [-1, 1])
        3. Volatility: ATR (낮을수록 좋음, q90 기준)
        4. Participation: Volume Z-score
        5. Location: Close position in 20-bar range
    
    Args:
        df: OHLCV + 지표 DataFrame (indicators.py 출력)
        cfg: 설정
    
    Returns:
        df with normalized factors
    
    NOTE: 실제 계산은 indicators.py에서 이미 수행됨.
          이 함수는 호환성을 위한 래퍼.
    """
    from .indicators import add_indicators
    
    # indicators.py가 이미 정규화까지 수행
    df = add_indicators(df, cfg)
    
    # 검증
    required = ["ema21_slope_n", "rsi_n", "adx_n", "participation_n", "location_n"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing normalized factors: {missing}")
    
    return df


def normalize_factors(df: pl.DataFrame) -> pl.DataFrame:
    """
    Factor 정규화 (이미 indicators.py에서 수행됨)
    
    호환성을 위한 no-op 함수
    """
    return df


def score_weighted_sum(
    df: pl.DataFrame,
    weights: Dict[str, float]
) -> pl.Series:
    """
    가중합 스코어 계산
    
    Args:
        df: normalized factors
        weights: {"trend": 0.25, "momentum": 0.25, ...}
    
    Returns:
        score: [-1, 1] Series
    
    Formula:
        score = (w_tr * trend + w_mo * momentum + w_vo * volatility +
                 w_pa * participation + w_lo * location) / sum(weights)
    """
    _EPS = 1e-12
    warmup = 300
    w_tr = float(weights.get("trend", 0.25))
    w_mo = float(weights.get("momentum", 0.25))
    w_vo = float(weights.get("volatility", 0.15))
    w_pa = float(weights.get("participation", 0.20))
    w_lo = float(weights.get("location", 0.15))
    w_sum = (w_tr + w_mo + w_vo + w_pa + w_lo) or 1.0
    
    # Volatility 스코어 (낮을수록 좋음)
    df_valid = df.slice(warmup, df.height - warmup) if df.height > warmup else df
    vol_ref = float(df["atr_p"].quantile(0.90)) if df_valid.height else 1.0
    vol_sc = (1.0 - (pl.col("atr_p") / (vol_ref + _EPS))).clip(-1.0, 1.0)
    
    score = (
        w_tr * pl.col("ema21_slope_n") +
        w_mo * pl.col("rsi_n") +
        w_vo * vol_sc +
        w_pa * pl.col("participation_n") +
        w_lo * pl.col("location_n")
    ) / w_sum
    
    return score.clip(-1.0, 1.0)