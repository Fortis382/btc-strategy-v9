# src/signals/meta_learning.py
"""
메타학습 Prior (v9.4 Section 7.2)
"""
from __future__ import annotations
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
from enum import Enum

class Regime(Enum):
    """시장 regime"""
    BULL = "BULL"
    BEAR = "BEAR"
    SIDEWAYS = "SIDEWAYS"

class BayesianPrior:
    """
    Bayesian Prior 관리 (Beta 분포)
    
    용도:
        - Safe Thompson Sampling 초기값
        - Cold-start 문제 해결
        - Regime별 prior 유지
    
    예시:
        prior = BayesianPrior()
        alpha, beta = prior.get(Regime.BULL, arm_idx=0)
        # Thompson Sampling에서 사용
    """
    
    def __init__(self, default_alpha: float = 1.0, default_beta: float = 1.0):
        self.default_alpha = default_alpha
        self.default_beta = default_beta
        
        # Regime별 prior: {regime: {arm_idx: (alpha, beta)}}
        self.priors: Dict[Regime, Dict[int, Tuple[float, float]]] = {
            Regime.BULL: {},
            Regime.BEAR: {},
            Regime.SIDEWAYS: {},
        }
    
    def get(
        self,
        regime: Regime,
        arm_idx: int
    ) -> Tuple[float, float]:
        """
        Prior 조회
        
        Returns:
            (alpha, beta)
        """
        if arm_idx not in self.priors[regime]:
            return (self.default_alpha, self.default_beta)
        
        return self.priors[regime][arm_idx]
    
    def update(
        self,
        regime: Regime,
        arm_idx: int,
        alpha: float,
        beta: float
    ) -> None:
        """Prior 업데이트 (in-place)"""
        self.priors[regime][arm_idx] = (alpha, beta)
    
    def save(self, path: Path) -> None:
        """디스크 저장"""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.priors, f)
    
    def load(self, path: Path) -> None:
        """디스크 로드"""
        if not path.exists():
            return
        
        with open(path, "rb") as f:
            self.priors = pickle.load(f)


def detect_regime(
    df,
    method: str = "adx_slope"
) -> Regime:
    """
    Regime 감지
    
    Args:
        df: OHLCV + 지표
        method: "adx_slope" (기본) | "ema_cross"
    
    Returns:
        Regime
    
    로직 (adx_slope):
        - ADX > 25 and EMA slope > 0 → BULL
        - ADX > 25 and EMA slope < 0 → BEAR
        - ADX < 25 → SIDEWAYS
    """
    if method == "adx_slope":
        adx = float(df["adx_n"][-1])
        slope = float(df["ema21_slope_n"][-1])
        
        if adx > 0.0:  # ADX > 25 (정규화 기준)
            return Regime.BULL if slope > 0 else Regime.BEAR
        else:
            return Regime.SIDEWAYS
    
    elif method == "ema_cross":
        ema_fast = float(df["ema21"][-1])
        ema_slow = float(df["ema55"][-1]) if "ema55" in df.columns else ema_fast
        
        return Regime.BULL if ema_fast > ema_slow else Regime.BEAR
    
    return Regime.SIDEWAYS


def update_prior_from_trades(
    prior: BayesianPrior,
    regime: Regime,
    trades: list,
    n_trades_min: int = 20
) -> None:
    """
    거래 결과로 prior 업데이트
    
    Args:
        prior: BayesianPrior 객체
        regime: 현재 regime
        trades: [(arm_idx, reward), ...]
        n_trades_min: 최소 거래 수 (20 미만이면 업데이트 안 함)
    
    로직:
        성공: alpha += 1
        실패: beta += 1
    """
    if len(trades) < n_trades_min:
        return
    
    # Arm별 집계
    arm_stats = {}
    for arm_idx, reward in trades:
        if arm_idx not in arm_stats:
            arm_stats[arm_idx] = {"alpha": 0, "beta": 0}
        
        if reward > 0:
            arm_stats[arm_idx]["alpha"] += 1
        else:
            arm_stats[arm_idx]["beta"] += 1
    
    # Prior 업데이트
    for arm_idx, stats in arm_stats.items():
        alpha, beta = prior.get(regime, arm_idx)
        prior.update(
            regime,
            arm_idx,
            alpha + stats["alpha"],
            beta + stats["beta"]
        )