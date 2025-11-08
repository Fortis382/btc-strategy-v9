# src/validation/purged_cv.py (신규 파일)

import numpy as np
from itertools import combinations

def combinatorial_purged_cv(
    data: np.ndarray,
    n_splits: int = 5,
    n_test_groups: int = 2,
    embargo_pct: float = 0.01
):
    """
    CPCV (v9.4 Section 16.8)
    
    Returns:
        [(train_idx, test_idx), ...] C(5,2)=10 folds
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
        
        # Purge train (embargo)
        train_mask = np.ones(n, dtype=bool)
        for idx in test_idx:
            start_purge = max(0, idx - embargo_size)
            end_purge = min(n, idx + embargo_size + 1)
            train_mask[start_purge:end_purge] = False
        
        train_idx = np.where(train_mask)[0]
        folds.append((train_idx, test_idx))
    
    return folds