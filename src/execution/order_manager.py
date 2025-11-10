# src/execution/order_manager.py
"""
주문 관리 (v9.4 Section 7.4)
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass
class Order:
    """단일 주문"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: float  # LIMIT/STOP_LIMIT 전용
    status: OrderStatus = OrderStatus.PENDING
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

class OrderManager:
    """
    주문 관리자
    
    기능:
        - 주문 생성/취소/수정
        - 상태 추적
        - 레이턴시 모니터링
    """
    
    def __init__(self, client=None):
        self.client = client  # BinanceClient 등
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0
    
    def place_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None
    ) -> Order:
        """
        주문 생성
        
        Args:
            symbol: "BTCUSDT"
            side: BUY | SELL
            order_type: MARKET | LIMIT | STOP | STOP_LIMIT
            quantity: 수량
            price: 지정가 (LIMIT/STOP_LIMIT)
            stop_price: 스톱가 (STOP/STOP_LIMIT)
        
        Returns:
            Order 객체
        
        예시:
            # 시장가 매수
            order = om.place_order("BTCUSDT", OrderSide.BUY, OrderType.MARKET, 0.1)
            
            # 지정가 매도
            order = om.place_order("BTCUSDT", OrderSide.SELL, OrderType.LIMIT, 0.1, price=65000)
        """
        self.order_counter += 1
        order_id = f"ORDER_{self.order_counter:06d}"
        
        order = Order(
            order_id=order_id,
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price or 0.0
        )
        
        # 실제 거래소 주문 (client가 있으면)
        if self.client:
            try:
                response = self.client.place_order(
                    symbol=symbol,
                    side=side.value,
                    order_type=order_type.value,
                    quantity=quantity,
                    price=price,
                    stop_price=stop_price
                )
                order.status = OrderStatus.FILLED if response.get("status") == "FILLED" else OrderStatus.PENDING
            except Exception as e:
                order.status = OrderStatus.REJECTED
                print(f"[ORDER ERROR] {e}")
        else:
            # 백테스트 모드: 즉시 체결
            order.status = OrderStatus.FILLED
            order.filled_qty = quantity
            order.avg_fill_price = price or 0.0
        
        self.orders[order_id] = order
        return order
    
    def cancel_order(self, order_id: str) -> bool:
        """
        주문 취소
        
        Returns:
            성공 여부
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            return False
        
        # 실제 거래소 취소
        if self.client:
            try:
                self.client.cancel_order(order.symbol, order_id)
                order.status = OrderStatus.CANCELLED
                return True
            except Exception as e:
                print(f"[CANCEL ERROR] {e}")
                return False
        else:
            # 백테스트 모드
            order.status = OrderStatus.CANCELLED
            return True
    
    def modify_order(
        self,
        order_id: str,
        new_price: Optional[float] = None,
        new_quantity: Optional[float] = None
    ) -> bool:
        """
        주문 수정 (취소 후 재생성)
        
        Returns:
            성공 여부
        """
        if order_id not in self.orders:
            return False
        
        old_order = self.orders[order_id]
        
        # 취소
        if not self.cancel_order(order_id):
            return False
        
        # 재생성
        new_order = self.place_order(
            symbol=old_order.symbol,
            side=old_order.side,
            order_type=old_order.order_type,
            quantity=new_quantity or old_order.quantity,
            price=new_price or old_order.price
        )
        
        return new_order.status != OrderStatus.REJECTED
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """주문 조회"""
        return self.orders.get(order_id)
    
    def get_active_orders(self) -> List[Order]:
        """활성 주문 조회"""
        return [
            o for o in self.orders.values()
            if o.status in [OrderStatus.PENDING, OrderStatus.PARTIAL]
        ]