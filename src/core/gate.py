# src/core/gate.py
"""
게이트 시스템 (v9.4 Section 4.3)

수정 내역:
- Expression → Series 반환 (evaluate)
- 미사용 변수 제거
- 위험한 인덱싱 수정
"""
from __future__ import annotations
import polars as pl
from typing import Dict, Any


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
        gate_ok: bool Series (evaluate된 결과)
    
    수정사항:
        - pl.all_horizontal() → df.select()로 evaluate
        - 빈 조건 처리: pl.Series 직접 생성
    
    예시:
        gate_series = calculate_gate(df, cfg)
        df = df.with_columns(gate_series.alias("gate_ok"))
    """
    g = cfg.get("gates", {})
    conds = []
    
    # 1. ADX Gate
    if g.get("use_adx_gate", True):
        adx_min = float(g.get("adx_min", 25.0))
        adx_thr = (adx_min - 25.0) / 25.0
        conds.append(pl.col("adx_n") >= pl.lit(adx_thr))
    
    # 2. Trend Gate (동적 threshold)
    if g.get("use_trend_gate", True):
        slope_min = float(g.get("ema_slope_min", 0.06))
        slope_chop = float(g.get("ema_slope_min_chop", 0.10))
        
        # ADX < 0 (횡보) → 높은 threshold
        slope_thr = (
            pl.when(pl.col("adx_n") < 0.0)
            .then(pl.lit(slope_chop))
            .otherwise(pl.lit(slope_min))
        )
        conds.append(pl.col("ema21_slope_n") >= slope_thr)
    
    # 3. Range Gate
    if g.get("use_range_gate", False):
        min_range = float(g.get("min_range_atr", 0.60))
        atr_len = int(cfg["indicators"].get("atr", 14))
        
        # 직접 계산 (중간 변수 제거)
        range_ok = (
            (pl.col("high") - pl.col("low")) 
            / (pl.col(f"atr{atr_len}_abs") + 1e-12) 
            >= pl.lit(min_range)
        )
        conds.append(range_ok)
    
    # 4. EMA Bias Gate
    if g.get("use_ema_bias_gate", False):
        ema_fast = int(cfg["indicators"]["ema"][0])
        ema_bias_norm = float(cfg["indicators"].get("ema_bias_norm", 0.010))
        
        bias = (
            (pl.col("close") - pl.col(f"ema{ema_fast}")) 
            / (pl.col(f"ema{ema_fast}") + 1e-12)
        )
        conds.append(bias >= pl.lit(ema_bias_norm))
    
    # 5. Dev Guard (과도한 이탈 방지)
    if g.get("use_dev_guard", False):
        ema_fast = int(cfg["indicators"]["ema"][0])
        atr_len = int(cfg["indicators"].get("atr", 14))
        max_dev = float(g.get("max_dev_atr", 0.80))
        
        dev = (
            (pl.col("close") - pl.col(f"ema{ema_fast}")).abs() 
            / (pl.col(f"atr{atr_len}_abs") + 1e-12)
        )
        conds.append(dev <= pl.lit(max_dev))
    
    # Expression → Series 변환
    if not conds:
        # 조건 없으면 모두 True
        return pl.Series("gate_ok", [True] * len(df), dtype=pl.Boolean)
    
    # all_horizontal() Expression을 evaluate
    gate_expr = pl.all_horizontal(*conds)
    gate_series = df.select(gate_expr.alias("gate_ok")).get_column("gate_ok")
    
    return gate_series


def check_hard_gates(
    df: pl.DataFrame,
    cfg: Dict[str, Any]
) -> pl.Series:
    """
    금지 조건 체크 (v9.4 Section 4.3.1)
    
    Returns:
        forbidden: bool Series (True = 거래 금지)
    
    예시:
        - 과열 (ATR > 3%)
        - 뉴스 이벤트 (FOMC, CPI)
        - 세션 제한 (아시아 시간대 제외)
    
    수정사항:
        - Expression → Series로 evaluate
    """
    atr_len = int(cfg["indicators"].get("atr", 14))
    g = cfg.get("gates", {})
    
    # Base: 모두 False (금지 없음)
    forbidden_expr = pl.lit(False)
    
    # 1. 변동성 과열 체크
    if g.get("use_vol_filter", False):
        max_atr = float(g.get("max_atr_p", 2.5))
        forbidden_expr = forbidden_expr | (pl.col("atr_p") > max_atr)
    
    # 2. 세션 제한 (TODO: 추후 구현)
    if g.get("use_session_gate", False):
        # allowed = g.get("allow_sessions", ["asia", "eu", "us"])
        # 시간대 기반 필터링
        pass
    
    # Expression → Series
    forbidden_series = df.select(forbidden_expr.alias("forbidden")).get_column("forbidden")
    
    return forbidden_series


def check_soft_gates(
    df: pl.DataFrame,
    cfg: Dict[str, Any],
    bar_idx: int = -1
) -> Dict[str, float]:
    """
    소프트 게이트 → 리스크 조정
    
    Args:
        df: DataFrame
        cfg: 설정
        bar_idx: 평가할 행 인덱스 (기본값: -1 = 마지막)
    
    Returns:
        {"risk_multiplier": float, "confidence": float}
    
    예시:
        ADX < 20 (횡보) → risk_multiplier=0.5
        ADX > 30 (트렌드) → risk_multiplier=1.5
    
    수정사항:
        - df["adx_n"][-1] → .row(bar_idx) 또는 .item()
    """
    # 마지막 행의 adx_n 값 추출
    if bar_idx == -1:
        adx = df.select(pl.col("adx_n").last()).item()
    else:
        adx = df.select(pl.col("adx_n").gather([bar_idx])).item()
    
    # ADX 정규화 값 기반 조정
    if adx < -0.2:  # ADX < 20 (횡보)
        return {
            "risk_multiplier": 0.5, 
            "confidence": 0.6
        }
    elif adx > 0.2:  # ADX > 30 (강한 트렌드)
        return {
            "risk_multiplier": 1.5, 
            "confidence": 0.9
        }
    else:  # 20 <= ADX <= 30 (중립)
        return {
            "risk_multiplier": 1.0, 
            "confidence": 0.75
        }