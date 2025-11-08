# tests/unit/test_ewq_numba.py (신규 생성)

import numpy as np
from src.signals.ewq_numba import ewq_update_numba, ewq_batch_numba

def test_ewq_single():
    """단일 업데이트 테스트"""
    q = ewq_update_numba(70.0, 75.0, 0.7, 0.05)
    assert 70.0 < q < 71.0, "상승해야 함"
    
    q = ewq_update_numba(70.0, 65.0, 0.7, 0.05)
    assert 69.0 < q < 70.0, "하락해야 함"

def test_ewq_batch_converge():
    """수렴 테스트: 동일 스코어 반복 → phi가 그 값에 수렴"""
    scores = np.array([75.0] * 1000, dtype=np.float64)
    phis = ewq_batch_numba(70.0, scores, 0.7, 0.05, 0.03, 96)
    
    # 마지막 phi는 75에 근접해야 함
    assert abs(phis[-1] - 75.0) < 1.0

def test_ewq_daily_cap():
    """일일 cap 테스트: 급등해도 3% 제한"""
    scores = np.array([90.0] * 96, dtype=np.float64)  # 1일치
    phis = ewq_batch_numba(70.0, scores, 0.7, 0.05, 0.03, 96)
    
    # 70 → 70*1.03 = 72.1 이하여야 함
    assert phis[-1] <= 72.1

def test_ewq_speed():
    """속도 테스트: 10만 봉 < 0.1초"""
    import time
    scores = np.random.uniform(60, 80, 100000).astype(np.float64)
    
    t0 = time.perf_counter()
    phis = ewq_batch_numba(70.0, scores, 0.7, 0.05, 0.03, 96)
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    print(f"10만 봉 EWQ: {elapsed:.3f}s")
    assert elapsed < 0.1, "너무 느림"