# src/core/gate.py
"""
게이트 시스템 (v9.4 Section 4.3)
"""
from __future__ import annotations
import polars as pl
from typing import Dict, Any
from .constants import GateType

def calculate_gate(
    df: pl.DataFrame,
    cfg: Dict[str, Any]
) -> pl.Series:
    """
    게이트 통과 여부 계산
    
    Args:
        df: OHLCV + 지표 DataFrame
        cfg: 설정 (gates 섹션)
    
    Returns:
        gate_ok: bool Series
    
    예시:
        df = df.with_columns(calculate_gate(df, cfg).alias("gate_ok"))
    """
    g = cfg.get("gates", {})
    conds = []
    
    # ADX Gate
    if g.get("use_adx_gate", True):
        adx_min = float(g.get("adx_min", 25.0))
        adx_thr = (adx_min - 25.0) / 25.0
        conds.append(pl.col("adx_n") >= pl.lit(adx_thr))
    
    # Trend Gate (동적)
    if g.get("use_trend_gate", True):
        slope_min = float(g.get("ema_slope_min", 0.06))
        slope_chop = float(g.get("ema_slope_min_chop", 0.10))
        adx_n = pl.col("adx_n")
        slope_thr = pl.when(adx_n < 0.0).then(pl.lit(slope_chop)).otherwise(pl.lit(slope_min))
        conds.append(pl.col("ema21_slope_n") >= slope_thr)
    
    # Range Gate
    if g.get("use_range_gate", False):
        min_range = float(g.get("min_range_atr", 0.60))
        atr_len = int(cfg["indicators"].get("atr", 14))
        atr_col = pl.col(f"atr{atr_len}_abs")
        range_ok = (pl.col("high") - pl.col("low")) / (atr_col + 1e-12) >= pl.lit(min_range)
        conds.append(range_ok)
    
    # EMA Bias
    if g.get("use_ema_bias_gate", False):
        ema_fast = int(cfg["indicators"]["ema"][0])
        ema_bias_norm = float(cfg["indicators"].get("ema_bias_norm", 0.010))
        bias = (pl.col("close") - pl.col(f"ema{ema_fast}")) / (pl.col(f"ema{ema_fast}") + 1e-12)
        conds.append(bias >= pl.lit(ema_bias_norm))
    
    # Dev Guard
    if g.get("use_dev_guard", False):
        ema_fast = int(cfg["indicators"]["ema"][0])
        atr_len = int(cfg["indicators"].get("atr", 14))
        max_dev = float(g.get("max_dev_atr", 0.80))
        dev = (pl.col("close") - pl.col(f"ema{ema_fast}")).abs() / (pl.col(f"atr{atr_len}_abs") + 1e-12)
        conds.append(dev <= pl.lit(max_dev))
    
    if not conds:
        return pl.lit(True)
    
    return pl.all_horizontal(conds)


def check_hard_gates(
    df: pl.DataFrame,
    cfg: Dict[str, Any]
) -> pl.Series:
    """
    금지 조건 체크 (v9.4 Section 4.3.1)
    
    Returns:
        forbidden: bool Series (True = 거래 금지)
    
    예시:
        - 뉴스 이벤트 (FOMC, CPI)
        - 과열 (ATR > 3%)
        - 세션 제한 (아시아 시간대 제외)
    """
    forbidden = pl.lit(False)
    
    # 과열 체크
    atr_len = int(cfg["indicators"].get("atr", 14))
    if cfg.get("gates", {}).get("use_vol_filter", False):
        max_atr = float(cfg["gates"].get("max_atr_p", 2.5))
        forbidden = forbidden | (pl.col("atr_p") > max_atr)
    
    # 세션 제한
    if cfg.get("gates", {}).get("use_session_gate", False):
        allowed = cfg["gates"].get("allow_sessions", ["asia", "eu", "us"])
        # TODO: 세션 판별 로직 (시간대 기반)
    
    return forbidden


def check_soft_gates(
    df: pl.DataFrame,
    cfg: Dict[str, Any]
) -> Dict[str, float]:
    """
    소프트 게이트 → 리스크 조정
    
    Returns:
        {"risk_multiplier": float, "confidence": float}
    
    예시:
        ADX < 20 → risk_multiplier=0.5
        ADX > 30 → risk_multiplier=1.5
    """
    adx = float(df["adx_n"][-1])
    
    if adx < -0.2:  # ADX < 20
        return {"risk_multiplier": 0.5, "confidence": 0.6}
    elif adx > 0.2:  # ADX > 30
        return {"risk_multiplier": 1.5, "confidence": 0.9}
    else:
        return {"risk_multiplier": 1.0, "confidence": 0.75}