# src/portfolio/risk_budget.py
"""
리스크 예산 관리 (v9.4 Section 10.2)
"""
from __future__ import annotations
from typing import Dict, List, Optional

class RiskBudget:
    """
    리스크 예산 할당
    
    원칙 (Section 10.2):
        - 총 리스크: 2% (단일 거래)
        - 일일 한도: 5%
        - 동시 포지션: 최대 3개
    
    할당 방식:
        - Equal: 각 포지션에 균등 할당
        - Confidence: 신뢰도 기반 할당
        - Kelly: Kelly Criterion
    """
    
    def __init__(
        self,
        max_risk_pct: float = 0.02,
        max_daily_loss: float = 0.05,
        max_positions: int = 3
    ):
        self.max_risk_pct = max_risk_pct
        self.max_daily_loss = max_daily_loss
        self.max_positions = max_positions
        
        self.daily_used = 0.0
        self.position_risks: Dict[str, float] = {}
    
    def allocate_risk(
        self,
        method: str = "equal",
        confidences: Optional[List[float]] = None
    ) -> List[float]:
        """
        리스크 할당
        
        Args:
            method: "equal" | "confidence" | "kelly"
            confidences: 각 arm의 신뢰도 [0, 1]
        
        Returns:
            리스크 비율 리스트 (합 = max_risk_pct)
        
        예시:
            method="equal", max_positions=3
            → [0.0067, 0.0067, 0.0067] (각 0.67%)
        """
        if method == "equal":
            n = self.max_positions
            return [self.max_risk_pct / n] * n
        
        elif method == "confidence":
            if not confidences:
                return self.allocate_risk("equal")
            
            total_conf = sum(confidences)
            return [
                (c / total_conf) * self.max_risk_pct
                for c in confidences
            ]
        
        elif method == "kelly":
            # Kelly Criterion: f = (bp - q) / b
            # 여기서는 단순화 (승률 55%, 1:1 배당)
            kelly_frac = 0.1  # 10% (Kelly의 1/2)
            return [kelly_frac * self.max_risk_pct] * self.max_positions
        
        return self.allocate_risk("equal")
    
    def check_budget(
        self,
        position_id: str,
        risk_pct: float
    ) -> bool:
        """
        예산 체크
        
        Returns:
            사용 가능 여부
        """
        # 1) 일일 한도
        if self.daily_used + risk_pct > self.max_daily_loss:
            return False
        
        # 2) 동시 포지션 수
        if len(self.position_risks) >= self.max_positions:
            return False
        
        return True
    
    def use(self, position_id: str, risk_pct: float) -> None:
        """리스크 사용"""
        self.position_risks[position_id] = risk_pct
        self.daily_used += risk_pct
    
    def release(self, position_id: str) -> None:
        """리스크 반환 (포지션 청산 시)"""
        if position_id in self.position_risks:
            del self.position_risks[position_id]
    
    def reset_daily(self) -> None:
        """일일 리셋"""
        self.daily_used = 0.0