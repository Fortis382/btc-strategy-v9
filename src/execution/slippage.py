# src/execution/slippage.py
"""
슬리피지 추정 (v9.4 실행 레이어)
"""
from __future__ import annotations
from typing import Dict, Any

def estimate_slippage(
    symbol: str,
    quantity: float,
    side: str,
    orderbook: Dict[str, Any]
) -> float:
    """
    슬리피지 추정 (호가창 기반)
    
    Args:
        symbol: "BTCUSDT"
        quantity: 주문 수량
        side: "BUY" | "SELL"
        orderbook: {"bids": [[price, qty], ...], "asks": [...]}
    
    Returns:
        슬리피지 (bps, basis points)
    
    예시:
        orderbook = {
            "bids": [[65000, 0.5], [64999, 1.0], ...],
            "asks": [[65001, 0.3], [65002, 0.8], ...],
        }
        slippage = estimate_slippage("BTCUSDT", 0.1, "BUY", orderbook)
        # → 2.5 bps
    """
    if side == "BUY":
        levels = orderbook.get("asks", [])
    else:
        levels = orderbook.get("bids", [])
    
    if not levels:
        return 0.0
    
    # 평균 체결가 계산
    remaining = quantity
    total_cost = 0.0
    
    for price, available in levels:
        fill_qty = min(remaining, available)
        total_cost += price * fill_qty
        remaining -= fill_qty
        
        if remaining <= 0:
            break
    
    if remaining > 0:
        # 호가창 부족 → 높은 슬리피지 가정
        return 50.0  # 50 bps
    
    avg_price = total_cost / quantity
    best_price = float(levels[0][0])
    
    # 슬리피지 (bps)
    slippage_bps = abs((avg_price - best_price) / best_price) * 10000
    
    return slippage_bps


def adaptive_limit_price(
    market_price: float,
    side: str,
    slippage_bps: float = 5.0
) -> float:
    """
    적응형 지정가 (슬리피지 보정)
    
    Args:
        market_price: 시장가
        side: "BUY" | "SELL"
        slippage_bps: 허용 슬리피지 (bps)
    
    Returns:
        지정가
    
    예시:
        market=65000, side=BUY, slippage=5bps
        → limit=65000 × 1.0005 = 65032.5
    """
    slippage_ratio = slippage_bps / 10000
    
    if side == "BUY":
        return market_price * (1 + slippage_ratio)
    else:
        return market_price * (1 - slippage_ratio)