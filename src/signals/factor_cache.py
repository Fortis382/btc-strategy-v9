# src/signals/factor_cache.py
"""
Factor 계산 캐시 (v9.4 Section 16.7.5)
속도: 디스크 캐시 히트시 0.1초 (계산 5초 → 50배)
"""
from __future__ import annotations
import polars as pl
from pathlib import Path
import hashlib
import json
from typing import Dict, Any, Optional

class FactorCache:
    """
    Polars DataFrame 캐시 (메모리맵 방식)
    
    캐시 키: MD5(파라미터 dict)
    저장 형식: Parquet (zstd level 3, 빠른 압축)
    만료: 7일 (자동 삭제)
    
    예시:
        cache = FactorCache()
        params = {"ema_fast": 20, "ema_slow": 50}
        
        df = cache.get("data/partitioned", params)
        if df is None:
            df = compute_factors_polars(data, cfg)  # 5초
            cache.set("data/partitioned", params, df)
        # 이후 호출: 0.1초
    """
    
    def __init__(self, cache_dir: str = "cache/factors"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _hash_params(self, params: Dict[str, Any]) -> str:
        """파라미터 기반 캐시 키 생성"""
        s = json.dumps(params, sort_keys=True)
        return hashlib.md5(s.encode()).hexdigest()[:12]
    
    def get(
        self,
        data_path: str,
        params: Dict[str, Any]
    ) -> Optional[pl.DataFrame]:
        """
        캐시 로드
        
        Args:
            data_path: 데이터 경로 (캐시 키에 포함 안 됨)
            params: 파라미터 dict
        
        Returns:
            DataFrame or None (miss)
        """
        key = self._hash_params(params)
        cache_path = self.cache_dir / f"{key}.parquet"
        
        if cache_path.exists():
            # 만료 체크 (7일)
            import time
            age_days = (time.time() - cache_path.stat().st_mtime) / 86400
            if age_days > 7:
                cache_path.unlink()
                return None
            
            return pl.read_parquet(cache_path)
        
        return None
    
    def set(
        self,
        data_path: str,
        params: Dict[str, Any],
        df: pl.DataFrame
    ) -> None:
        """
        캐시 저장
        
        Args:
            data_path: 데이터 경로
            params: 파라미터 dict
            df: Factor 계산 결과
        """
        key = self._hash_params(params)
        cache_path = self.cache_dir / f"{key}.parquet"
        
        df.write_parquet(
            cache_path,
            compression="zstd",
            compression_level=3  # 빠른 압축 (level 9 대비 2배 빠름)
        )
    
    def clear(self) -> int:
        """모든 캐시 삭제"""
        count = 0
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
            count += 1
        return count
    
    def clear_expired(self, max_age_days: int = 7) -> int:
        """만료된 캐시 삭제"""
        import time
        count = 0
        now = time.time()
        
        for f in self.cache_dir.glob("*.parquet"):
            age = (now - f.stat().st_mtime) / 86400
            if age > max_age_days:
                f.unlink()
                count += 1
        
        return count