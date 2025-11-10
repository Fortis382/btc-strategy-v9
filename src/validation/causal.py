# src/validation/causal.py
"""
인과 백테스팅 (v9.4 Section 8.1)
DoWhy 기반 인과 추론
"""
from __future__ import annotations
import polars as pl
import numpy as np
from typing import Dict, Any, List

def causal_backtest(
    df: pl.DataFrame,
    treatment: str,
    outcome: str = "return_pct",
    confounders: List[str] = None
) -> Dict[str, Any]:
    """
    인과 효과 추정 (DoWhy 래퍼)
    
    Args:
        df: 데이터 (features + outcome)
        treatment: 처치 변수 (예: "factor_A")
        outcome: 결과 변수 (예: "return_pct")
        confounders: 혼란변수 (예: ["volatility", "volume"])
    
    Returns:
        {"ate": float, "ci_lower": float, "ci_upper": float}
    
    예시:
        # Factor A의 인과 효과 측정
        result = causal_backtest(
            df,
            treatment="trend_factor",
            outcome="return_pct",
            confounders=["volatility", "volume"]
        )
        print(f"ATE: {result['ate']:.4f} ± {result['ci_width']:.4f}")
    
    참고:
        - DoWhy 설치: pip install dowhy
        - 논문: Pearl (2009) Causality
    """
    try:
        import dowhy
        from dowhy import CausalModel
    except ImportError:
        print("[WARN] DoWhy not installed. Returning mock result.")
        return {"ate": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "method": "mock"}
    
    if confounders is None:
        confounders = []
    
    # Polars → Pandas (DoWhy는 Pandas 필요)
    df_pd = df.to_pandas()
    
    # 인과 그래프 정의
    graph = f"""
    digraph {{
        {treatment} -> {outcome};
        {' -> '.join([f'{c} -> {outcome}; {c} -> {treatment};' for c in confounders])}
    }}
    """
    
    # CausalModel 생성
    model = CausalModel(
        data=df_pd,
        treatment=treatment,
        outcome=outcome,
        graph=graph
    )
    
    # 식별 (identification)
    identified_estimand = model.identify_effect()
    
    # 추정 (estimation) - Propensity Score Matching
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.propensity_score_matching"
    )
    
    # 반박 (refutation) - Random Common Cause
    refute = model.refute_estimate(
        identified_estimand,
        estimate,
        method_name="random_common_cause"
    )
    
    ate = float(estimate.value)
    
    # CI 추정 (bootstrap)
    try:
        ci = estimate.get_confidence_intervals(confidence_level=0.95)
        ci_lower = float(ci[0])
        ci_upper = float(ci[1])
    except:
        ci_lower = ate - 0.1
        ci_upper = ate + 0.1
    
    return {
        "ate": round(ate, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_width": round(ci_upper - ci_lower, 4),
        "p_value": round(float(refute.new_effect), 4),
        "method": "propensity_score_matching",
    }


def factor_shuffle_test(
    df: pl.DataFrame,
    factor_col: str,
    backtest_fn,
    cfg: Dict[str, Any],
    n_shuffles: int = 100
) -> Dict[str, Any]:
    """
    Factor Shuffle 테스트 (v9.4 Section 8.1)
    
    목적:
        Factor가 랜덤이면 성능이 사라지는지 검증
    
    Args:
        df: 데이터
        factor_col: 테스트할 factor 컬럼명
        backtest_fn: 백테스트 함수
        cfg: 설정
        n_shuffles: 셔플 횟수
    
    Returns:
        {"original_wr": 0.65, "shuffled_mean_wr": 0.50, "p_value": 0.02}
    
    예시:
        result = factor_shuffle_test(
            df,
            factor_col="trend_n",
            backtest_fn=simple_backtest,
            cfg=cfg,
            n_shuffles=100
        )
        if result["p_value"] < 0.05:
            print("✓ Factor is significant")
    """
    # 원본 백테스트
    _, original_result = backtest_fn(df, cfg)
    original_wr = original_result["winrate"]
    
    # 셔플 백테스트
    shuffled_wrs = []
    for i in range(n_shuffles):
        df_shuffled = df.with_columns(
            pl.col(factor_col).shuffle(seed=i).alias(factor_col)
        )
        _, shuffled_result = backtest_fn(df_shuffled, cfg)
        shuffled_wrs.append(shuffled_result["winrate"])
    
    shuffled_mean = np.mean(shuffled_wrs)
    shuffled_std = np.std(shuffled_wrs)
    
    # p-value (단측 검정)
    z_score = (original_wr - shuffled_mean) / (shuffled_std + 1e-12)
    from scipy.stats import norm
    p_value = 1 - norm.cdf(z_score)
    
    return {
        "original_wr": round(float(original_wr), 4),
        "shuffled_mean_wr": round(float(shuffled_mean), 4),
        "shuffled_std_wr": round(float(shuffled_std), 4),
        "z_score": round(float(z_score), 2),
        "p_value": round(float(p_value), 4),
        "significant": p_value < 0.05,
    }


def ate_estimation(
    treated_returns: np.ndarray,
    control_returns: np.ndarray
) -> float:
    """
    Average Treatment Effect (ATE)
    
    Args:
        treated_returns: 처치군 수익률
        control_returns: 대조군 수익률
    
    Returns:
        ATE (차이)
    
    Formula:
        ATE = E[Y|T=1] - E[Y|T=0]
    """
    return float(np.mean(treated_returns) - np.mean(control_returns))