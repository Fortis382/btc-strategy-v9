# src/validation/purged_cv.py

"""
CPCV (Combinatorial Purged Cross-Validation)

참고: v9.4 Section 16.8, López de Prado (2018)
"""
from __future__ import annotations
import numpy as np
from typing import List, Tuple
from itertools import combinations

def purged_kfold_cv(
    data: np.ndarray,
    n_splits: int = 5,
    embargo_pct: float = 0.01
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    기본 Purged K-Fold CV
    
    Args:
        data: 데이터 배열 (shape: [n, ...])
        n_splits: Fold 수 (기본 5)
        embargo_pct: Embargo 비율 (기본 1%)
    
    Returns:
        [(train_idx, test_idx), ...] n_splits개
    """
    n = len(data)
    test_size = n // n_splits
    embargo_size = int(n * embargo_pct)
    
    folds = []
    for i in range(n_splits):
        test_start = i * test_size
        test_end = test_start + test_size
        test_idx = np.arange(test_start, test_end)
        
        train_idx = np.concatenate([
            np.arange(0, max(0, test_start - embargo_size)),
            np.arange(min(n, test_end + embargo_size), n)
        ])
        
        folds.append((train_idx, test_idx))
    
    return folds


def combinatorial_purged_cv(
    data: np.ndarray,
    n_splits: int = 5,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    CPCV (Combinatorial Purged Cross-Validation)
    
    Args:
        data: 데이터 배열
        n_splits: 전체 분할 수
        n_test_groups: 각 fold의 test 그룹 수
        embargo_pct: Embargo 비율
    
    Returns:
        [(train_idx, test_idx), ...] C(n_splits, n_test_groups)개
    
    예시:
        n_splits=5, n_test_groups=2 → C(5,2)=10 folds
    """
    n = len(data)
    test_size = n // n_splits
    embargo_size = int(n * embargo_pct)
    
    all_groups = list(range(n_splits))
    test_combos = list(combinations(all_groups, n_test_groups))
    
    folds = []
    for combo in test_combos:
        test_idx = np.array([], dtype=int)
        for g in combo:
            start = g * test_size
            end = start + test_size
            test_idx = np.concatenate([test_idx, np.arange(start, end)])
        
        train_mask = np.ones(n, dtype=bool)
        for idx in test_idx:
            start_purge = max(0, idx - embargo_size)
            end_purge = min(n, idx + embargo_size + 1)
            train_mask[start_purge:end_purge] = False
        
        train_idx = np.where(train_mask)[0]
        folds.append((train_idx, test_idx))
    
    return folds


def validate_strategy_cpcv(
    df,
    cfg: dict,
    backtest_fn,
    n_splits: int = 5,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01
) -> dict:
    """
    CPCV로 전략 검증
    
    Returns:
        {
            'train_metrics': {...},
            'test_metrics': {...},
            'gap': float,
            'overfitting_risk': float,
        }
    """
    import numpy as np
    
    data = df.to_numpy()
    folds = combinatorial_purged_cv(data, n_splits, n_test_groups, embargo_pct)
    
    train_metrics = []
    test_metrics = []
    
    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"[CPCV] Fold {i+1}/{len(folds)}")
        
        df_train = df[train_idx.tolist()]
        df_test = df[test_idx.tolist()]
        
        _, result_train = backtest_fn(df_train, cfg)
        _, result_test = backtest_fn(df_test, cfg)
        
        train_metrics.append(result_train)
        test_metrics.append(result_test)
    
    train_wr = np.mean([m['winrate'] for m in train_metrics])
    test_wr = np.mean([m['winrate'] for m in test_metrics])
    
    train_pf = np.mean([m['pf'] for m in train_metrics if m['pf'] != float('inf')])
    test_pf = np.mean([m['pf'] for m in test_metrics if m['pf'] != float('inf')])
    
    gap_wr = (test_wr - train_wr) / (train_wr + 1e-12)
    overfitting_risk = max(0, -gap_wr) * 100
    
    return {
        'train_metrics': {'wr': round(train_wr, 4), 'pf': round(train_pf, 3)},
        'test_metrics': {'wr': round(test_wr, 4), 'pf': round(test_pf, 3)},
        'gap_wr': round(gap_wr, 4),
        'overfitting_risk': round(overfitting_risk, 1),
        'n_folds': len(folds),
    }