# src/validation/chaos.py
"""
카오스 엔지니어링 (v9.4 Section 8.2)
"""
from __future__ import annotations
import polars as pl
import numpy as np
from typing import Dict, Any, Callable
from enum import Enum

class ChaosScenario(Enum):
    """카오스 시나리오"""
    LATENCY_SPIKE = "latency_spike"           # 레이턴시 급증
    DATA_LOSS = "data_loss"                   # 데이터 손실
    FLASH_CRASH = "flash_crash"               # 급락
    ORDERBOOK_IMBALANCE = "orderbook_imbalance"  # 호가창 불균형
    API_FAILURE = "api_failure"               # API 실패

def inject_latency(
    df: pl.DataFrame,
    latency_ms: float = 500.0,
    failure_rate: float = 0.1
) -> pl.DataFrame:
    """
    레이턴시 주입
    
    Args:
        df: 데이터
        latency_ms: 레이턴시 (밀리초)
        failure_rate: 실패율 (0.1 = 10%)
    
    Returns:
        수정된 df (타임스탬프 지연)
    
    시뮬레이션:
        - 10% 확률로 500ms 지연
        - 거래 기회 손실 가능성
    """
    n = len(df)
    delayed_indices = np.random.choice(
        n,
        size=int(n * failure_rate),
        replace=False
    )
    
    # 지연된 행 마킹 (실제로는 거래 금지 플래그)
    mask = np.zeros(n, dtype=bool)
    mask[delayed_indices] = True
    
    df = df.with_columns(
        pl.Series("latency_delayed", mask)
    )
    
    return df


def inject_flash_crash(
    df: pl.DataFrame,
    crash_idx: int,
    crash_pct: float = 0.10,
    recovery_bars: int = 4
) -> pl.DataFrame:
    """
    플래시 크래시 주입
    
    Args:
        df: OHLCV 데이터
        crash_idx: 크래시 발생 인덱스
        crash_pct: 급락 비율 (0.10 = 10%)
        recovery_bars: 회복 봉 수
    
    Returns:
        수정된 df
    
    시뮬레이션:
        - crash_idx에서 가격 10% 급락
        - 이후 4봉에 걸쳐 회복
        - 손절 대량 발생 가능성
    """
    if crash_idx >= len(df):
        return df
    
    df_list = df.to_dicts()
    
    # 크래시
    original_price = df_list[crash_idx]["close"]
    crash_price = original_price * (1 - crash_pct)
    
    df_list[crash_idx]["low"] = crash_price
    df_list[crash_idx]["close"] = crash_price
    
    # 회복
    for i in range(1, recovery_bars + 1):
        idx = crash_idx + i
        if idx >= len(df_list):
            break
        
        recovery_ratio = i / recovery_bars
        recovered_price = crash_price + (original_price - crash_price) * recovery_ratio
        df_list[idx]["close"] = recovered_price
    
    return pl.DataFrame(df_list)


def inject_data_loss(
    df: pl.DataFrame,
    loss_rate: float = 0.05
) -> pl.DataFrame:
    """
    데이터 손실 주입
    
    Args:
        df: 데이터
        loss_rate: 손실 비율 (0.05 = 5%)
    
    Returns:
        수정된 df (일부 행 제거)
    
    시뮬레이션:
        - 5% 확률로 봉 데이터 누락
        - 지표 계산 오류 가능성
    """
    n = len(df)
    keep_indices = np.random.choice(
        n,
        size=int(n * (1 - loss_rate)),
        replace=False
    )
    keep_indices = np.sort(keep_indices)
    
    return df[keep_indices.tolist()]


def chaos_scenario(
    df: pl.DataFrame,
    backtest_fn: Callable,
    cfg: Dict[str, Any],
    scenario: ChaosScenario
) -> Dict[str, Any]:
    """
    카오스 시나리오 실행
    
    Args:
        df: 원본 데이터
        backtest_fn: 백테스트 함수
        cfg: 설정
        scenario: 시나리오 종류
    
    Returns:
        {"scenario": ..., "original_wr": 0.65, "chaos_wr": 0.58, "degradation": -10.7%}
    
    예시:
        result = chaos_scenario(
            df,
            simple_backtest,
            cfg,
            ChaosScenario.FLASH_CRASH
        )
        print(f"Performance degradation: {result['degradation']:.1%}")
    """
    # 원본 백테스트
    _, original_result = backtest_fn(df, cfg)
    original_wr = original_result["winrate"]
    
    # 카오스 주입
    if scenario == ChaosScenario.LATENCY_SPIKE:
        df_chaos = inject_latency(df, latency_ms=500, failure_rate=0.1)
        # 레이턴시 지연된 행 제외
        df_chaos = df_chaos.filter(~pl.col("latency_delayed"))
    
    elif scenario == ChaosScenario.DATA_LOSS:
        df_chaos = inject_data_loss(df, loss_rate=0.05)
    
    elif scenario == ChaosScenario.FLASH_CRASH:
        crash_idx = len(df) // 2
        df_chaos = inject_flash_crash(df, crash_idx, crash_pct=0.10, recovery_bars=4)
    
    else:
        df_chaos = df
    
    # 카오스 백테스트
    _, chaos_result = backtest_fn(df_chaos, cfg)
    chaos_wr = chaos_result["winrate"]
    
    degradation = (chaos_wr - original_wr) / (original_wr + 1e-12)
    
    return {
        "scenario": scenario.value,
        "original_wr": round(float(original_wr), 4),
        "chaos_wr": round(float(chaos_wr), 4),
        "degradation": round(float(degradation), 4),
        "robust": abs(degradation) < 0.15,  # 15% 이내 허용
    }