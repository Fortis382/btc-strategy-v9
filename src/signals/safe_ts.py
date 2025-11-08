# src/signals/safe_ts.py (신규 생성)

import numpy as np
from numba import jit

@jit(nopython=True, cache=True)
def beta_sample(alpha: float, beta: float) -> float:
    """Beta 분포 샘플링"""
    return np.random.beta(alpha, beta)

@jit(nopython=True, cache=True)
def safe_ts_select(
    alphas: np.ndarray,
    betas: np.ndarray,
    c_min: float = 0.6,
    q_min: float = 1.0
) -> int:
    """
    Safe Thompson Sampling 선택
    
    Args:
        alphas: 각 arm의 α (성공 횟수 + 1)
        betas: 각 arm의 β (실패 횟수 + 1)
        c_min: 최소 confidence
        q_min: 최소 quality
    
    Returns:
        선택된 arm index
    """
    n_arms = len(alphas)
    samples = np.empty(n_arms, dtype=np.float64)
    
    for i in range(n_arms):
        # Thompson Sampling
        samples[i] = beta_sample(alphas[i], betas[i])
    
    return int(np.argmax(samples))

def safe_ts_update(
    arm_idx: int,
    reward: float,
    alphas: np.ndarray,
    betas: np.ndarray
) -> None:
    """Safe-TS 업데이트 (in-place)"""
    if reward > 0:
        alphas[arm_idx] += 1
    else:
        betas[arm_idx] += 1