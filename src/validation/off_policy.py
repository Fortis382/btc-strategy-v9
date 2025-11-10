# src/validation/off_policy.py
"""
Off-Policy Evaluation (v9.4 Section 8.3)
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple, Dict, Any

def doubly_robust_estimator(
    rewards: np.ndarray,
    propensities: np.ndarray,
    predicted_rewards: np.ndarray,
    actions: np.ndarray,
    policy_probs: np.ndarray
) -> float:
    """
    Doubly Robust (DR) 추정
    
    Args:
        rewards: 실제 보상 (shape: [n])
        propensities: 행동 확률 (로깅 정책, shape: [n])
        predicted_rewards: 예측 보상 (모델, shape: [n])
        actions: 실제 행동 (shape: [n])
        policy_probs: 평가할 정책의 행동 확률 (shape: [n])
    
    Returns:
        DR 추정값
    
    Formula:
        DR = E[(π(a|x)/μ(a|x)) × (r - Q(x,a)) + Q(x,a)]
        π: 평가할 정책
        μ: 로깅 정책
        Q: 예측 모델
    
    예시:
        # Shadow trading 로그로 Live 정책 평가
        dr = doubly_robust_estimator(
            rewards=shadow_rewards,
            propensities=shadow_probs,
            predicted_rewards=model_preds,
            actions=shadow_actions,
            policy_probs=live_probs
        )
    """
    n = len(rewards)
    
    # Importance weights
    weights = policy_probs / (propensities + 1e-12)
    weights = np.clip(weights, 0, 10)  # Cap at 10
    
    # DR formula
    dr_values = weights * (rewards - predicted_rewards) + predicted_rewards
    
    return float(np.mean(dr_values))


def importance_sampling(
    rewards: np.ndarray,
    propensities: np.ndarray,
    policy_probs: np.ndarray
) -> float:
    """
    Importance Sampling (IS)
    
    Args:
        rewards: 실제 보상
        propensities: 로깅 정책 확률
        policy_probs: 평가할 정책 확률
    
    Returns:
        IS 추정값
    
    Formula:
        IS = E[(π(a|x)/μ(a|x)) × r]
    """
    weights = policy_probs / (propensities + 1e-12)
    weights = np.clip(weights, 0, 10)
    
    return float(np.mean(weights * rewards))


def wis_estimator(
    rewards: np.ndarray,
    propensities: np.ndarray,
    policy_probs: np.ndarray
) -> float:
    """
    Weighted Importance Sampling (WIS)
    
    Args:
        rewards: 실제 보상
        propensities: 로깅 정책 확률
        policy_probs: 평가할 정책 확률
    
    Returns:
        WIS 추정값
    
    Formula:
        WIS = Σ(w_i × r_i) / Σ(w_i)
        w_i = π(a|x)/μ(a|x)
    
    장점:
        - IS보다 분산 낮음
        - 편향 약간 있지만 안정적
    """
    weights = policy_probs / (propensities + 1e-12)
    weights = np.clip(weights, 0, 10)
    
    numerator = np.sum(weights * rewards)
    denominator = np.sum(weights)
    
    if denominator < 1e-12:
        return 0.0
    
    return float(numerator / denominator)


def off_policy_evaluate(
    shadow_log: List[Dict[str, Any]],
    live_policy_fn: callable,
    reward_model: callable = None
) -> Dict[str, float]:
    """
    Off-Policy 평가 (모든 추정량 실행)
    
    Args:
        shadow_log: Shadow trading 로그
            [{"state": ..., "action": 0, "reward": 1.2, "prob": 0.3}, ...]
        live_policy_fn: 평가할 정책 함수
            fn(state) -> (action, prob)
        reward_model: 보상 예측 모델 (DR용, optional)
            fn(state, action) -> predicted_reward
    
    Returns:
        {"is": ..., "wis": ..., "dr": ...}
    
    예시:
        result = off_policy_evaluate(
            shadow_log=shadow_trades,
            live_policy_fn=lambda s: safe_ts_select(s),
            reward_model=lambda s, a: predict_reward(s, a)
        )
        print(f"Expected return (DR): {result['dr']:.2f}R")
    """
    states = [log["state"] for log in shadow_log]
    actions = np.array([log["action"] for log in shadow_log])
    rewards = np.array([log["reward"] for log in shadow_log])
    propensities = np.array([log["prob"] for log in shadow_log])
    
    # Live policy 확률
    policy_results = [live_policy_fn(s) for s in states]
    policy_probs = np.array([p for _, p in policy_results])
    
    # IS, WIS
    is_est = importance_sampling(rewards, propensities, policy_probs)
    wis_est = wis_estimator(rewards, propensities, policy_probs)
    
    # DR (모델 있으면)
    dr_est = None
    if reward_model:
        predicted_rewards = np.array([
            reward_model(s, a) for s, a in zip(states, actions)
        ])
        dr_est = doubly_robust_estimator(
            rewards, propensities, predicted_rewards, actions, policy_probs
        )
    
    return {
        "is": round(float(is_est), 4),
        "wis": round(float(wis_est), 4),
        "dr": round(float(dr_est), 4) if dr_est is not None else None,
        "n_samples": len(shadow_log),
    }