# src/validation/metrics.py
"""
평가 지표 (v9.4 Section 12 KPI)
"""
from __future__ import annotations
import numpy as np
import polars as pl
from typing import Dict, Any, Optional

# src/validation/metrics.py (라인 11 교체)
def calculate_sharpe(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252  # ← 일별 기준으로 변경
) -> float:
    """
    Sharpe Ratio (Daily basis)
    
    NOTE: 15분봉 거래를 일별로 aggregation 후 계산
          backtest_polars.py에서 사전 처리 필요
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - (risk_free_rate / periods_per_year)
    mean_r = np.mean(excess_returns)
    std_r = np.std(excess_returns, ddof=1)
    
    if std_r < 1e-12:
        return 0.0
    
    return float(mean_r / std_r * np.sqrt(periods_per_year))
def calculate_mdd(equity_curve: np.ndarray) -> float:
    """
    Maximum Drawdown (MDD)
    
    Args:
        equity_curve: 누적 수익 곡선
    
    Returns:
        MDD (비율, 0~1)
    
    Formula:
        DD_t = (Peak_t - Equity_t) / Peak_t
        MDD = max(DD_t)
    
    예시:
        equity = [100, 110, 105, 95, 120]
        mdd = calculate_mdd(equity)
        # → 0.136 (13.6% 최대 낙폭)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / (peak + 1e-12)
    
    return float(np.max(drawdown))

def calculate_profit_factor(
    returns: np.ndarray
) -> float:
    """
    Profit Factor
    
    Args:
        returns: 수익률 배열
    
    Returns:
        Profit Factor
    
    Formula:
        PF = sum(wins) / abs(sum(losses))
    
    예시:
        returns = [1.2, -0.9, 1.5, -0.8, 2.0]
        pf = calculate_profit_factor(returns)
        # → 2.76 (좋음, 1.5 이상)
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    gross_profit = np.sum(wins) if len(wins) > 0 else 0.0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 0.0
    
    if gross_loss < 1e-12:
        return float('inf') if gross_profit > 0 else 0.0
    
    return float(gross_profit / gross_loss)


def calculate_calmar(
    annual_return: float,
    mdd: float
) -> float:
    """
    Calmar Ratio
    
    Args:
        annual_return: 연간 수익률 (0.35 = 35%)
        mdd: Maximum Drawdown (0.12 = 12%)
    
    Returns:
        Calmar Ratio
    
    Formula:
        Calmar = Annual Return / MDD
    
    예시:
        calmar = calculate_calmar(0.35, 0.12)
        # → 2.92 (좋음, 3.0 이상 우수)
    """
    if mdd < 1e-12:
        return float('inf') if annual_return > 0 else 0.0
    
    return float(annual_return / mdd)


def calculate_win_rate(returns: np.ndarray) -> float:
    """승률"""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns > 0))


def calculate_expectancy(returns: np.ndarray) -> float:
    """기댓값 (R 단위)"""
    if len(returns) == 0:
        return 0.0
    return float(np.mean(returns))


def calculate_kelly_fraction(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Kelly Criterion
    
    Args:
        win_rate: 승률 (0.55)
        avg_win: 평균 승리 (1.5R)
        avg_loss: 평균 손실 (1.0R, 양수)
    
    Returns:
        Kelly 비율
    
    Formula:
        f = (p × b - q) / b
        p = win_rate
        q = 1 - p
        b = avg_win / avg_loss
    """
    if avg_loss < 1e-12:
        return 0.0
    
    p = win_rate
    q = 1 - p
    b = avg_win / avg_loss
    
    f = (p * b - q) / b
    
    return max(0.0, float(f))


def compute_all_metrics(
    trades_df: pl.DataFrame,
    initial_balance: float = 10000.0
) -> Dict[str, Any]:
    """
    모든 지표 계산
    
    Args:
        trades_df: 거래 결과 (columns: rr, bars, reason)
        initial_balance: 초기 잔고
    
    Returns:
        지표 dict
    
    예시:
        metrics = compute_all_metrics(trades_df)
        print(f"Sharpe: {metrics['sharpe']:.2f}")
        print(f"MDD: {metrics['mdd']:.1%}")
    """
    if trades_df.height == 0:
        return {
            "n_trades": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "calmar": 0.0,
            "mdd": 0.0,
            "expectancy": 0.0,
            "kelly": 0.0,
        }
    
    returns = trades_df["rr"].to_numpy()
    
    # 누적 수익 곡선
    equity = initial_balance * (1 + np.cumsum(returns) * 0.01)  # R → %
    
    # 연간 수익률 (15분봉 기준)
    total_return = (equity[-1] - initial_balance) / initial_balance
    n_periods = len(returns)
    periods_per_year = 35040  # 365 * 24 * 4
    annual_return = total_return * (periods_per_year / n_periods) if n_periods > 0 else 0.0
    
    # 개별 지표
    win_rate = calculate_win_rate(returns)
    pf = calculate_profit_factor(returns)
    sharpe = calculate_sharpe(returns)
    mdd = calculate_mdd(equity)
    calmar = calculate_calmar(annual_return, mdd)
    expectancy = calculate_expectancy(returns)
    
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 1.0
    kelly = calculate_kelly_fraction(win_rate, avg_win, avg_loss)
    
    return {
        "n_trades": len(returns),
        "win_rate": round(float(win_rate), 4),
        "profit_factor": round(float(pf), 3),
        "sharpe": round(float(sharpe), 2),
        "calmar": round(float(calmar), 2),
        "mdd": round(float(mdd), 4),
        "expectancy": round(float(expectancy), 4),
        "kelly": round(float(kelly), 4),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
    }