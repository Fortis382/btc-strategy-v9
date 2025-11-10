# src/core/constants.py
"""
전역 상수 정의 (v9.4 Section 4.1)
"""
from __future__ import annotations
from enum import Enum

# 타임프레임
TIMEFRAMES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}

# 리스크 제한
RISK_LIMITS = {
    "max_risk_per_trade": 0.02,      # 2%
    "max_daily_loss": 0.05,          # 5%
    "max_position_size": 0.1,        # 10%
    "max_leverage": 7,
    "max_concurrent_positions": 3,
}

# SLO 목표
SLO_TARGETS = {
    "latency_p95_ms": 120,
    "latency_p99_ms": 250,
    "processing_time_15m_s": 1.0,
    "decision_log_delay_s": 5.0,
}

# 상태 (Section 5.1)
class State(Enum):
    OFF = "OFF"              # 시스템 정지
    STANDBY = "STANDBY"      # 대기 (게이트 통과 대기)
    ACTIVE = "ACTIVE"        # 활성 (거래 가능)
    PAUSE = "PAUSE"          # 일시정지 (리스크 초과)
    DRAIN = "DRAIN"          # 점진 청산
    EMERGENCY = "EMERGENCY"  # 긴급 중단

# 게이트 타입
class GateType(Enum):
    HARD = "HARD"  # 금지 조건 (뉴스, 과열)
    SOFT = "SOFT"  # 리스크 조정 (ADX, Slope)
    RANGE = "RANGE"  # 레인지 필터

# 계약 상태
class ContractStatus(Enum):
    ACTIVE = "ACTIVE"
    TP_HIT = "TP_HIT"
    SL_HIT = "SL_HIT"
    TIMEOUT = "TIMEOUT"
    FORCE_CLOSED = "FORCE_CLOSED"

# EWQ 기본값
EWQ_DEFAULTS = {
    "initial_threshold": 70.0,
    "theta": 0.7,
    "alpha": 0.05,
    "daily_cap": 0.03,
    "tf_per_day": 96,
}

# 5-Factor 기본 가중치
FACTOR_WEIGHTS = {
    "trend": 0.25,
    "momentum": 0.25,
    "volatility": 0.15,
    "participation": 0.20,
    "location": 0.15,
}