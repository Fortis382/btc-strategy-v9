# src/portfolio/liquidation.py (신규 생성)

import polars as pl
from typing import Dict, Any

def staggered_exit(
    position_size: float,
    current_pnl_pct: float,
    levels: list = [0.03, 0.04, 0.05]
) -> float:
    """
    점진적 청산 (v9.4 Section 11.3)
    
    Args:
        position_size: 현재 포지션 크기
        current_pnl_pct: 현재 PnL (%)
        levels: 청산 레벨 (기본 3%, 4%, 5%)
    
    Returns:
        청산할 비율 (0~1)
    """
    if current_pnl_pct < levels[0]:
        return 0.0
    elif current_pnl_pct < levels[1]:
        return 0.33  # 1/3 청산
    elif current_pnl_pct < levels[2]:
        return 0.67  # 2/3 청산
    else:
        return 1.0   # 전체 청산


def apply_mdd_breaker(
    equity_curve: pl.Series,
    mdd_threshold: float = 0.05
) -> pl.Series:
    """
    MDD Breaker (v9.4 Section 11.4)
    
    Args:
        equity_curve: 누적 R 수익 곡선
        mdd_threshold: MDD 임계값 (기본 5%)
    
    Returns:
        청산 시그널 (bool Series)
    """
    peak = equity_curve.cum_max()
    dd = (peak - equity_curve) / (peak + 1e-12)
    
    return dd >= mdd_threshold