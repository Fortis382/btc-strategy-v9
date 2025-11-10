# src/portfolio/position_sizing.py
"""
포지션 사이징 (v9.4 Section 10.3)
"""
from __future__ import annotations
from typing import Dict, Any

def calculate_position_size(
    price: float,
    atr: float,
    risk_pct: float,
    account_balance: float,
    leverage: int = 1
) -> float:
    """
    포지션 크기 계산 (ATR 기반)
    
    Args:
        price: 현재 가격
        atr: ATR 값
        risk_pct: 리스크 비율 (0.02 = 2%)
        account_balance: 계좌 잔고
        leverage: 레버리지
    
    Returns:
        포지션 크기 (BTC 수량)
    
    Formula:
        risk_dollar = account_balance × risk_pct
        position_size = risk_dollar / atr
    
    예시:
        balance=10,000, risk_pct=0.02, atr=500
        → risk_dollar=200
        → position_size=200/500=0.4 BTC
    """
    risk_dollar = account_balance * risk_pct
    position_size = risk_dollar / atr
    
    # 레버리지 적용
    position_size *= leverage
    
    return position_size


def kelly_fraction(
    win_rate: float,
    profit_factor: float,
    conservative: float = 0.5
) -> float:
    """
    Kelly Criterion
    
    Args:
        win_rate: 승률 (0.55 = 55%)
        profit_factor: 수익 배수 (1.5)
        conservative: Kelly의 비율 (0.5 = Half Kelly)
    
    Returns:
        포지션 비율
    
    Formula:
        f = (bp - q) / b
        b = profit_factor
        p = win_rate
        q = 1 - p
    """
    b = profit_factor
    p = win_rate
    q = 1 - p
    
    f = ((b * p) - q) / b
    f = max(0, f)  # 음수 방지
    
    return f * conservative


def atr_based_sizing(
    price: float,
    atr: float,
    atr_multiplier: float = 1.0
) -> float:
    """
    ATR 기반 사이징 (변동성 조정)
    
    Args:
        price: 현재 가격
        atr: ATR 값
        atr_multiplier: ATR 배수 (기본 1.0)
    
    Returns:
        포지션 크기 (상대값)
    
    로직:
        ATR이 클수록 작은 포지션
        ATR이 작을수록 큰 포지션
    """
    volatility = atr / price
    base_size = 1.0 / (volatility * atr_multiplier)
    
    return base_size