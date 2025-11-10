# src/core/contracts.py
"""
계약 시스템 (v9.4 Section 6.3)
"""
from __future__ import annotations
from typing import Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from .constants import ContractStatus

@dataclass
class Contract:
    """
    단일 거래 계약
    
    불변 속성 (진입 시 고정):
        - entry_price, entry_ts
        - tp_levels, sl_level
        - max_hold_bars
    
    가변 속성:
        - status
        - exit_price, exit_ts
        - actual_rr
    """
    # 불변
    entry_price: float
    entry_ts: datetime
    tp_levels: List[float]
    sl_level: float
    max_hold_bars: int
    atr: float
    
    # 가변
    status: ContractStatus = ContractStatus.ACTIVE
    exit_price: float = 0.0
    exit_ts: datetime = None
    actual_rr: float = 0.0
    bars_held: int = 0
    
    def __post_init__(self):
        if self.exit_ts is None:
            self.exit_ts = self.entry_ts
    
    def check_exit(
        self,
        current_bar: Dict[str, Any],
        bar_index: int
    ) -> bool:
        """
        청산 조건 체크
        
        Args:
            current_bar: {"high": ..., "low": ..., "close": ...}
            bar_index: 현재 봉 인덱스
        
        Returns:
            청산 여부
        """
        if self.status != ContractStatus.ACTIVE:
            return False
        
        self.bars_held += 1
        
        # 손절
        if current_bar["low"] <= self.sl_level:
            self.status = ContractStatus.SL_HIT
            self.exit_price = self.sl_level
            self.actual_rr = -(self.entry_price - self.sl_level) / self.atr
            return True
        
        # 익절
        for i, tp in enumerate(self.tp_levels, start=1):
            if current_bar["high"] >= tp:
                self.status = ContractStatus.TP_HIT
                self.exit_price = tp
                self.actual_rr = (tp - self.entry_price) / self.atr
                return True
        
        # 타임아웃
        if self.bars_held >= self.max_hold_bars:
            self.status = ContractStatus.TIMEOUT
            self.exit_price = current_bar["close"]
            self.actual_rr = (current_bar["close"] - self.entry_price) / self.atr
            return True
        
        return False


class ContractManager:
    """계약 관리자"""
    
    def __init__(self):
        self.contracts: List[Contract] = []
    
    def create(
        self,
        entry_price: float,
        entry_ts: datetime,
        atr: float,
        cfg: Dict[str, Any]
    ) -> Contract:
        """
        새 계약 생성
        
        Args:
            entry_price: 진입가
            entry_ts: 진입 시간
            atr: ATR 값
            cfg: 설정 (risk 섹션)
        
        Returns:
            Contract 객체
        """
        risk = cfg["risk"]
        tp_R = [float(x) for x in risk.get("atr_tp", [1.2, 1.5, 2.0])]
        sl_R = float(risk.get("atr_sl", 1.0))
        max_hold = int(risk.get("max_hold_min", 120) // 15)
        
        tp_levels = [entry_price + r * atr for r in tp_R]
        sl_level = entry_price - sl_R * atr
        
        contract = Contract(
            entry_price=entry_price,
            entry_ts=entry_ts,
            tp_levels=tp_levels,
            sl_level=sl_level,
            max_hold_bars=max_hold,
            atr=atr
        )
        
        self.contracts.append(contract)
        return contract
    
    def get_active(self) -> List[Contract]:
        """활성 계약 조회"""
        return [c for c in self.contracts if c.status == ContractStatus.ACTIVE]