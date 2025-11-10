# src/core/state_machine.py
"""
상태 머신 (v9.4 Section 5)
"""
from __future__ import annotations
from typing import Dict, Any, Optional
from .constants import State

class StateManager:
    """
    상태 전환 관리
    
    우선순위 (Section 5.2):
        1. EMERGENCY (MDD Breaker)
        2. PAUSE (일일 손실 초과)
        3. DRAIN (포지션 청산)
        4. ACTIVE (거래 가능)
        5. STANDBY (대기)
    """
    
    def __init__(self, initial_state: State = State.STANDBY):
        self.state = initial_state
        self.history = [(initial_state, "init")]
    
    def transition(
        self,
        new_state: State,
        reason: str,
        force: bool = False
    ) -> bool:
        """
        상태 전환
        
        Args:
            new_state: 목표 상태
            reason: 전환 이유
            force: 우선순위 무시
        
        Returns:
            성공 여부
        """
        if not force and not self._is_valid_transition(new_state):
            return False
        
        old = self.state
        self.state = new_state
        self.history.append((new_state, reason))
        
        print(f"[STATE] {old.value} → {new_state.value} ({reason})")
        return True
    
    def _is_valid_transition(self, target: State) -> bool:
        """유효한 전환인지 검증"""
        current = self.state
        
        # EMERGENCY는 언제나 진입 가능
        if target == State.EMERGENCY:
            return True
        
        # EMERGENCY에서는 OFF로만 가능
        if current == State.EMERGENCY:
            return target == State.OFF
        
        # 일반 전환
        valid_map = {
            State.OFF: [State.STANDBY],
            State.STANDBY: [State.ACTIVE, State.OFF],
            State.ACTIVE: [State.PAUSE, State.DRAIN, State.STANDBY, State.OFF],
            State.PAUSE: [State.ACTIVE, State.DRAIN, State.OFF],
            State.DRAIN: [State.STANDBY, State.OFF],
        }
        
        return target in valid_map.get(current, [])
    
    def check_auto_transition(
        self,
        metrics: Dict[str, float],
        cfg: Dict[str, Any]
    ) -> Optional[State]:
        """
        자동 전환 체크 (v9.4 Section 5.3)
        
        Args:
            metrics: {"daily_loss": 0.03, "mdd": 0.05, ...}
            cfg: 설정
        
        Returns:
            전환할 상태 (None이면 유지)
        """
        # 1) MDD Breaker
        if cfg["risk"].get("enable_mdd_breaker", False):
            mdd_threshold = float(cfg["risk"].get("mdd_breaker", 0.05))
            if metrics.get("mdd", 0) >= mdd_threshold:
                return State.EMERGENCY
        
        # 2) 일일 손실
        max_daily = float(cfg["risk"].get("max_daily_loss", 0.05))
        if metrics.get("daily_loss", 0) >= max_daily:
            return State.PAUSE
        
        # 3) 복구 조건
        if self.state == State.PAUSE:
            if metrics.get("daily_loss", 0) < max_daily * 0.5:
                return State.ACTIVE
        
        return None