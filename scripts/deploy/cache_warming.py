# scripts/deploy/cache_warming.py (신규 생성)

"""
Numba JIT 캐시 Warming

Production 배포 전 실행하여 컴파일 오버헤드 제거
"""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

def warm_all():
    """모든 Numba 함수 캐시 warming"""
    print("=== Numba Cache Warming ===")
    
    # 1) EWQ
    from src.signals.ewq_numba import warm_cache as warm_ewq
    warm_ewq()
    
    # 2) 향후 추가 Numba 함수 (Safe-TS 등)
    # from src.signals.safe_ts_numba import warm_cache as warm_ts
    # warm_ts()
    
    print("\n✓ All Numba caches warmed")

if __name__ == "__main__":
    warm_all()