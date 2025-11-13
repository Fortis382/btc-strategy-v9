# scripts/backtest/cpcv_validate.py
"""CPCV-lite with indicators pre-calculation"""
from __future__ import annotations
from pathlib import Path
import argparse
import json
import sys
import polars as pl
from typing import List, Dict, Any
import yaml

sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.backtest.backtest_polars import BacktestEngine, _read_parquet_any
from scripts.backtest.run_backtest import load_ohlcv
from src.signals.indicators import add_indicators


def _time_splits(df: pl.DataFrame, k: int, embargo: int):
    """Time-series k-fold with embargo"""
    n = df.height
    fold_size = n // k
    
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < k - 1 else n
        yield slice(test_start, test_end)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--embargo-bars", type=int, default=96)
    ap.add_argument("--out", type=Path, default=Path("logs/cpcv"))
    args = ap.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Load config
    cfg = yaml.safe_load(args.config.read_text("utf-8"))
    
    # Load data (use run_backtest's loader)
    print("[CPCV] Loading data...")
    df = load_ohlcv(cfg)
    
    # Add indicators once (without _idx - engine will add it)
    print("[CPCV] Computing indicators (once)...")
    df = df.sort("ts")  # ← _idx 제거 (BacktestEngine에서 생성)
    df = add_indicators(df, cfg)
    
    # Create engine
    engine = BacktestEngine(cfg)
    
    folds: List[Dict[str, Any]] = []
    
    print(f"\n[CPCV] Running {args.folds} folds...")
    for i, test_slice in enumerate(_time_splits(df, args.folds, args.embargo_bars), 1):
        test_df = df.slice(test_slice.start, test_slice.stop - test_slice.start)
        
        # Run with skip_indicators=True
        res = engine.run(test_df, skip_indicators=True)
        m = res["metrics"]
        m["fold"] = i
        folds.append(m)
        
        print(f"  [FOLD {i}] n={m['total_trades']} WR={m['winrate']:.2%} "
              f"PF={m['profit_factor']:.2f} Sharpe={m['sharpe']:.2f}")
    
    # Aggregate (weighted by bars)
    weights = []
    for test_slice in _time_splits(df, args.folds, args.embargo_bars):
        weights.append(test_slice.stop - test_slice.start)
    W = sum(weights)
    
    agg = {
        "folds": len(folds),
        "wr_wavg": sum(f["winrate"] * w for f, w in zip(folds, weights)) / W,
        "pf_avg": sum(f["profit_factor"] for f in folds) / len(folds),
        "expR_avg": sum(f["expectancy_R"] for f in folds) / len(folds),
        "sharpe_avg": sum(f["sharpe"] for f in folds) / len(folds),
    }
    
    # Save
    (args.out / "cpcv.json").write_text(
        json.dumps({"folds": folds, "agg": agg}, ensure_ascii=False, indent=2),
        "utf-8"
    )
    
    print(f"\n[CPCV SUMMARY]")
    print(f"  Folds: {agg['folds']}")
    print(f"  WR (weighted): {agg['wr_wavg']:.2%}")
    print(f"  PF (avg): {agg['pf_avg']:.2f}")
    print(f"  ExpR (avg): {agg['expR_avg']:.3f}R")
    print(f"  Sharpe (avg): {agg['sharpe_avg']:.2f}")


if __name__ == "__main__":
    main()