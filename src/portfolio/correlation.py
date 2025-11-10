# src/portfolio/correlation.py
"""
상관계수 관리 (v9.4 Section 10.4)
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple

class CosineCorrelation:
    """
    Cosine 상관계수 (v9.4 Section 10.4)
    
    장점:
        - 방향성만 측정 (크기 무관)
        - [-1, 1] 범위
        - 빠른 계산
    
    용도:
        - 동시 포지션 간 상관 체크
        - 높은 상관(>0.7) → 포지션 거부
    """
    
    def __init__(self, decay: float = 0.95, window: int = 20):
        self.decay = decay
        self.window = window
        self.history: List[np.ndarray] = []
    
    def compute_correlation(
        self,
        returns_a: np.ndarray,
        returns_b: np.ndarray
    ) -> float:
        """
        Cosine 상관계수 계산
        
        Args:
            returns_a: 수익률 배열 A
            returns_b: 수익률 배열 B
        
        Returns:
            상관계수 [-1, 1]
        
        Formula:
            cos(θ) = (A·B) / (||A|| × ||B||)
        """
        if len(returns_a) != len(returns_b):
            raise ValueError("Length mismatch")
        
        dot = np.dot(returns_a, returns_b)
        norm_a = np.linalg.norm(returns_a)
        norm_b = np.linalg.norm(returns_b)
        
        if norm_a < 1e-12 or norm_b < 1e-12:
            return 0.0
        
        return dot / (norm_a * norm_b)
    
    def correlation_decay(
        self,
        initial_corr: float,
        n_periods: int
    ) -> float:
        """
        상관계수 감쇠
        
        Args:
            initial_corr: 초기 상관계수
            n_periods: 경과 기간
        
        Returns:
            감쇠된 상관계수
        
        Formula:
            corr_t = corr_0 × decay^t
        """
        return initial_corr * (self.decay ** n_periods)
    
    def check_correlation(
        self,
        new_returns: np.ndarray,
        threshold: float = 0.7
    ) -> Tuple[bool, float]:
        """
        기존 포지션들과 상관 체크
        
        Args:
            new_returns: 새 포지션의 수익률
            threshold: 상관 임계값
        
        Returns:
            (허용 여부, 최대 상관계수)
        """
        if not self.history:
            return True, 0.0
        
        max_corr = 0.0
        for existing in self.history:
            corr = abs(self.compute_correlation(new_returns, existing))
            max_corr = max(max_corr, corr)
        
        return max_corr < threshold, max_corr
    
    def add_position(self, returns: np.ndarray) -> None:
        """포지션 추가 (상관 히스토리)"""
        self.history.append(returns[-self.window:])
    
    def remove_position(self, idx: int) -> None:
        """포지션 제거"""
        if 0 <= idx < len(self.history):
            del self.history[idx]