# src/execution/latency.py
"""
레이턴시 모니터링 (v9.4 Section 12 SLO)
"""
from __future__ import annotations
import time
from typing import List, Dict
from collections import deque

class LatencyMonitor:
    """
    레이턴시 측정 및 SLO 검증
    
    SLO 목표 (v9.4 Section 12):
        - P95 < 120ms
        - P99 < 250ms
        - 15분봉 처리 < 1초
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies: deque = deque(maxlen=window_size)
    
    def measure_latency(self, func, *args, **kwargs):
        """
        함수 실행 시간 측정
        
        Args:
            func: 측정할 함수
            *args, **kwargs: 함수 인자
        
        Returns:
            (결과, 레이턴시_ms)
        
        예시:
            result, latency = monitor.measure_latency(place_order, "BTCUSDT", "BUY", 0.1)
            print(f"Order placed in {latency:.2f}ms")
        """
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        t1 = time.perf_counter()
        
        latency_ms = (t1 - t0) * 1000
        self.latencies.append(latency_ms)
        
        return result, latency_ms
    
    def get_percentiles(self) -> Dict[str, float]:
        """
        백분위수 계산
        
        Returns:
            {"p50": ..., "p95": ..., "p99": ...}
        """
        if not self.latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        
        import numpy as np
        arr = np.array(self.latencies)
        
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "mean": float(arr.mean()),
            "max": float(arr.max()),
        }
    
    def check_slo(self) -> Dict[str, bool]:
        """
        SLO 검증
        
        Returns:
            {"p95_ok": True, "p99_ok": False, ...}
        """
        perc = self.get_percentiles()
        
        return {
            "p95_ok": perc["p95"] < 120.0,
            "p99_ok": perc["p99"] < 250.0,
            "all_ok": perc["p95"] < 120.0 and perc["p99"] < 250.0,
        }
    
    def report(self) -> str:
        """레이턴시 리포트 생성"""
        perc = self.get_percentiles()
        slo = self.check_slo()
        
        report = [
            "=== Latency Report ===",
            f"Samples: {len(self.latencies)}",
            f"Mean: {perc['mean']:.2f}ms",
            f"P50: {perc['p50']:.2f}ms",
            f"P95: {perc['p95']:.2f}ms {'✓' if slo['p95_ok'] else '✗ (>120ms)'}",
            f"P99: {perc['p99']:.2f}ms {'✓' if slo['p99_ok'] else '✗ (>250ms)'}",
            f"Max: {perc['max']:.2f}ms",
        ]
        
        return "\n".join(report)