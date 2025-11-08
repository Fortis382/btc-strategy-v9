# src/signals/ewq_numba.py (신규 파일)

from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def ewq_batch_numba(
    q_init: float,
    scores: np.ndarray,
    theta: float,
    alpha: float,
    daily_cap: float,
    tf_per_day: int
) -> np.ndarray:
    """
    Numba JIT EWQ 배치 업데이트
    
    Args:
        q_init: 초기 임계값 (70.0)
        scores: 스코어 배열
        theta: 분위 타겟 (0.7)
        alpha: 학습률 (0.05)
        daily_cap: 일일 최대 변동 (0.03 = 3%)
        tf_per_day: 하루 봉 수 (96 for 15m)
    
    Returns:
        phi 배열 (scores와 같은 길이)
    """
    n = len(scores)
    phis = np.empty(n, dtype=np.float64)
    phis[0] = q_init
    
    max_delta_per_tf = daily_cap / tf_per_day
    
    for i in range(1, n):
        # 기본 업데이트
        step = 1.0 if scores[i] > phis[i-1] else -1.0
        delta = alpha * step * abs(scores[i] - phis[i-1])
        
        # 일일 cap 적용
        delta_capped = max(-max_delta_per_tf, min(delta, max_delta_per_tf))
        phis[i] = phis[i-1] + delta_capped
    
    return phis

# ===== 캐시 warming (첫 실행시만 0.5초 소요) =====
if __name__ == "__main__":
    dummy = np.array([70.0] * 100)
    ewq_batch_numba(70.0, dummy, 0.7, 0.05, 0.03, 96)
    print("✓ Numba cache warmed")