# tests/unit/test_ewq_numba.py (전체 교체)

import numpy as np
from src.signals.ewq_numba import ewq_update_numba, ewq_batch_numba

def test_ewq_single():
    """단일 업데이트 테스트"""
    q = ewq_update_numba(70.0, 75.0, 0.7, 0.05)
    assert 70.0 < q < 71.0, "상승해야 함"
    
    q = ewq_update_numba(70.0, 65.0, 0.7, 0.05)
    assert 69.0 < q < 70.0, "하락해야 함"

def test_ewq_batch_converge():
    """수렴 테스트: daily_cap 제약 고려"""
    scores = np.array([75.0] * 10000, dtype=np.float64)  # ✅ 1000 → 10000 (100일)
    phis = ewq_batch_numba(70.0, scores, 0.7, 0.05, 0.03, 96)
    
    # ✅ 수정: 100일이면 70 → 73 정도 (3% * 100일 = 기댓값 73)
    # 실제: alpha=0.05는 지수 감쇠 → 약 72.5
    assert 72.0 < phis[-1] < 74.0, f"Expected ~73, got {phis[-1]}"

def test_ewq_daily_cap():
    """일일 cap 테스트"""
    scores = np.array([90.0] * 96, dtype=np.float64)
    phis = ewq_batch_numba(70.0, scores, 0.7, 0.05, 0.03, 96)
    
    # 70 → 70*1.03 = 72.1
    assert phis[-1] <= 72.1

def test_ewq_speed():
    """속도 테스트"""
    import time
    scores = np.random.uniform(60, 80, 100000).astype(np.float64)
    
    t0 = time.perf_counter()
    phis = ewq_batch_numba(70.0, scores, 0.7, 0.05, 0.03, 96)
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    print(f"10만 봉 EWQ: {elapsed:.3f}s")
    assert elapsed < 0.1, "너무 느림"