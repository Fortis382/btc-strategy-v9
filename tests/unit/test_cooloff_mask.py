# tests/unit/test_cooloff_mask.py (신규 생성)

import polars as pl
from scripts.backtest.run_backtest import cooloff_mask

def test_cooloff_basic():
    """기본 동작: 신호 후 2봉 차단"""
    mask = pl.Series([False, True, True, False, True, False, False, False])
    result = cooloff_mask(mask, bars=2)
    expected = [False, True, False, False, True, False, False, False]
    
    assert result.to_list() == expected, f"Expected {expected}, got {result.to_list()}"

def test_cooloff_no_bars():
    """bars=0이면 원본 유지"""
    mask = pl.Series([True, True, False, True])
    result = cooloff_mask(mask, bars=0)
    
    assert result.to_list() == mask.to_list()

def test_cooloff_overlap():
    """겹치는 신호: 첫 번째만 유지"""
    mask = pl.Series([True, True, True, False])
    result = cooloff_mask(mask, bars=1)
    expected = [True, False, True, False]  # 0번 → 1번 차단 → 2번 허용
    
    assert result.to_list() == expected

def test_cooloff_all_false():
    """신호 없으면 전부 False"""
    mask = pl.Series([False] * 10)
    result = cooloff_mask(mask, bars=2)
    
    assert result.to_list() == [False] * 10