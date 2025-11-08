# src/signals/ewq_numba.py
"""
EWQ (Exponentially Weighted Quantile) with Numba JIT

참고: v9.4 Section 16.7.4
"""
from __future__ import annotations
from numba import jit
import numpy as np

@jit(nopython=True, cache=True)
def ewq_update_numba(
    q_prev: float,
    x: float,
    theta: float,
    alpha: float
) -> float:
    """
    단일 EWQ 업데이트 (Numba JIT)
    
    Args:
        q_prev: 이전 임계값
        x: 현재 스코어
        theta: 목표 분위 (0.7 = 상위 30%)
        alpha: 학습률 (0.05)
    
    Returns:
        q_new: 새 임계값
    
    수식:
        step = +1 if x > q_prev else -1
        delta = α × step × |x - q_prev|
        q_new = q_prev + delta
    
    예시:
        q_prev=70, x=75 → step=+1, delta=+0.25, q_new=70.25
        q_prev=70, x=65 → step=-1, delta=-0.25, q_new=69.75
    """
    step = 1.0 if x > q_prev else -1.0
    delta = alpha * step * abs(x - q_prev)
    return q_prev + delta


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
    배치 EWQ 업데이트 + 일일 변동 제한
    
    Args:
        q_init: 초기 임계값 (70.0)
        scores: 스코어 배열 (shape: [n])
        theta: 목표 분위 (0.7)
        alpha: 학습률 (0.05)
        daily_cap: 일일 최대 변동 (0.03 = 3%)
        tf_per_day: 하루 봉 수 (96 for 15m)
    
    Returns:
        phis: 임계값 배열 (shape: [n])
    
    일일 변동 제한:
        - 15분봉 96개 = 1일
        - daily_cap=0.03이면 하루 최대 3% 변동
        - 봉당 최대 delta = 0.03 / 96 = 0.03125%
    """
    n = len(scores)
    phis = np.empty(n, dtype=np.float64)
    phis[0] = q_init
    
    max_delta_per_tf = daily_cap / tf_per_day
    
    for i in range(1, n):
        # 기본 업데이트
        q_new = ewq_update_numba(phis[i-1], scores[i], theta, alpha)
        
        # 일일 cap 적용
        delta = q_new - phis[i-1]
        delta_capped = max(-max_delta_per_tf, min(delta, max_delta_per_tf))
        phis[i] = phis[i-1] + delta_capped
    
    return phis


# ===== 벤치마크 함수 =====
def benchmark_ewq(n: int = 10000) -> None:
    """
    EWQ 성능 측정
    
    Args:
        n: 스코어 개수 (기본 1만 = 3년 15분봉 약 10만)
    """
    import time
    
    print(f"[EWQ Benchmark] n={n:,} scores")
    
    # 더미 데이터
    np.random.seed(42)
    scores = np.random.uniform(60, 80, n).astype(np.float64)
    
    # 1) Python loop (baseline)
    print("\n1) Python loop:")
    t0 = time.perf_counter()
    q = 70.0
    for s in scores:
        step = 1.0 if s > q else -1.0
        delta = 0.05 * step * abs(s - q)
        q = q + delta
    t1 = time.perf_counter()
    print(f"   Time: {(t1-t0):.3f}s")
    print(f"   Result: q={q:.4f}")
    
    # 2) Numba batch (첫 실행 = 컴파일 포함)
    print("\n2) Numba batch (첫 실행):")
    t2 = time.perf_counter()
    phis = ewq_batch_numba(70.0, scores, 0.7, 0.05, 0.03, 96)
    t3 = time.perf_counter()
    print(f"   Time: {(t3-t2):.3f}s (컴파일 포함)")
    print(f"   Result: phi[-1]={phis[-1]:.4f}")
    
    # 3) Numba batch (2차 실행 = 캐시 히트)
    print("\n3) Numba batch (캐시 히트):")
    t4 = time.perf_counter()
    phis = ewq_batch_numba(70.0, scores, 0.7, 0.05, 0.03, 96)
    t5 = time.perf_counter()
    print(f"   Time: {(t5-t4):.3f}s")
    print(f"   Result: phi[-1]={phis[-1]:.4f}")
    
    # 속도 향상
    speedup = (t1 - t0) / (t5 - t4)
    print(f"\n✓ Speedup: {speedup:.1f}x")
    print(f"✓ Python: {(t1-t0):.3f}s → Numba: {(t5-t4):.3f}s")


# ===== 캐시 warming 함수 =====
def warm_cache() -> None:
    """
    첫 실행시 Numba 컴파일 캐시 생성
    
    Production 배포 시 이 함수를 먼저 호출하여
    실제 거래 시작 전에 컴파일 오버헤드 제거.
    """
    dummy = np.array([70.0] * 100, dtype=np.float64)
    _ = ewq_batch_numba(70.0, dummy, 0.7, 0.05, 0.03, 96)
    print("✓ Numba EWQ cache warmed")


if __name__ == "__main__":
    # 벤치마크 실행
    benchmark_ewq(n=10000)
    
    # 캐시 warming
    print("\n" + "="*50)
    warm_cache()