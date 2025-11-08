# scripts/validation/run_cpcv.py

"""
CPCV 기반 전략 검증

사용법:
    python scripts/validation/run_cpcv.py --config config/settings_v9.yaml
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import yaml

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
sys.path.insert(0, str(_ROOT))

from src.signals.indicators import add_indicators
from src.core.scoring import score_and_gate
from src.validation.purged_cv import combinatorial_purged_cv
import polars as pl

def load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_ohlcv(cfg: Dict[str, Any]) -> pl.DataFrame:
    from src.core.loader_polars import load_ohlcv as _load
    return _load(
        _ROOT,
        cfg["data"]["path_primary"],
        cfg["data"]["path_fallback"],
        cfg["data"].get("start"),
        cfg["data"].get("end")
    )

def quantile(series: pl.Series, pct: float) -> float:
    return float(series.quantile(pct))

def compute_thresholds(scores: pl.Series, cfg: Dict[str, Any]):
    dbg = cfg.get("debug", {})
    auto = dbg.get("auto_thresholds", {"cand_pct": 0.75, "enter_pct": 0.85})
    cand_pct = float(auto.get("cand_pct", 0.75))
    enter_pct = float(auto.get("enter_pct", 0.85))
    thr_c = quantile(scores, cand_pct)
    thr_e = quantile(scores, enter_pct)
    return thr_c, thr_e

def cooloff_mask(mask: pl.Series, bars: int) -> pl.Series:
    if bars <= 0:
        return mask
    arr = mask.to_list()
    n = len(arr)
    block = 0
    for i in range(n):
        if block > 0:
            arr[i] = False
            block -= 1
        if arr[i]:
            block = bars
    return pl.Series(arr, dtype=pl.Boolean)

def simple_backtest(df: pl.DataFrame, cfg: Dict[str, Any]):
    risk = cfg["risk"]
    atr_len = int(cfg["indicators"]["atr"])
    atr_col_abs = f"atr{atr_len}_abs"
    close = df["close"]
    atr = df[atr_col_abs]

    tp_R = [float(x) for x in risk.get("atr_tp", [1.2, 1.3, 1.5])]
    sl_R = float(risk.get("atr_sl", 1.0))
    max_hold_bars = int(risk.get("max_hold_min", 720) // 15)

    rows = []
    i = 0
    n = len(df)

    while i < n:
        if bool(df["enter_mask"][i]):
            entry = float(close[i])
            atr_i = float(atr[i])
            tp_levels = [entry + k * atr_i for k in tp_R]
            sl_level = entry - sl_R * atr_i

            j = i + 1
            rr = 0.0
            reason = "timeout"
            
            while j < n and (j - i) < max_hold_bars:
                hi = float(df["high"][j])
                lo = float(df["low"][j])

                if lo <= sl_level:
                    rr = -sl_R
                    reason = "sl"
                    break

                hit_idx = None
                for k, tp in enumerate(tp_levels, start=1):
                    if hi >= tp:
                        hit_idx = k
                        break
                if hit_idx is not None:
                    rr = float(tp_R[hit_idx - 1])
                    reason = f"tp{hit_idx}"
                    break

                j += 1

            last_idx = (j - 1) if j > i else i
            rows.append((
                df["ts"][i],
                entry,
                df["ts"][last_idx],
                float(close[last_idx]),
                reason,
                int(j - i),
                float(rr),
            ))
            i = j
        else:
            i += 1

    schema = {
        "entry_ts": pl.Datetime,
        "entry": pl.Float64,
        "exit_ts": pl.Datetime,
        "exit": pl.Float64,
        "reason": pl.Utf8,
        "bars": pl.Int32,
        "rr": pl.Float64,
    }
    if rows:
        trades = pl.DataFrame(rows, schema=schema, orient="row")
    else:
        trades = pl.DataFrame(schema=schema)

    if trades.height == 0:
        return trades, {"winrate": 0.0, "pf": 0.0, "expR": 0.0, "mdd_R": 0.0}

    wins = trades.filter(pl.col("rr") > 0)
    losses = trades.filter(pl.col("rr") < 0)
    gross_win = float(wins["rr"].sum()) if wins.height else 0.0
    gross_loss_abs = -float(losses["rr"].sum()) if losses.height else 0.0

    pf = (gross_win / gross_loss_abs) if gross_loss_abs > 1e-12 else float("inf")
    winrate = float((trades["rr"] > 0).mean())
    expR = float(trades["rr"].mean())

    return trades, {
        "winrate": round(winrate, 4),
        "pf": round(pf, 3) if pf != float("inf") else float("inf"),
        "expR": round(expR, 4),
        "mdd_R": 0.0,
    }

def prepare_df_for_backtest(df: pl.DataFrame, cfg: Dict[str, Any]) -> pl.DataFrame:
    df = add_indicators(df, cfg)
    df = score_and_gate(df, cfg)
    
    dbg = cfg.get("debug", {})
    use_gate = not bool(dbg.get("no_gate", False))
    
    if use_gate:
        score_pool = df.filter(pl.col("gate_ok"))["score"]
    else:
        score_pool = df["score"]
    
    thr_c, thr_e = compute_thresholds(score_pool, cfg)
    
    g = cfg["gates"]
    base_gate = df["gate_ok"] if use_gate else pl.Series([True] * len(df), dtype=pl.Boolean)
    
    cand_mask = (df["score"] >= thr_c)
    enter_mask = (df["score"] >= thr_e)
    
    if int(g.get("cooloff_bars", 0)) > 0:
        cand_mask = cooloff_mask(cand_mask, int(g["cooloff_bars"]))
    
    cand_mask = cand_mask & base_gate
    enter_mask = enter_mask & base_gate
    
    df = df.with_columns([
        pl.Series("cand_mask", cand_mask),
        pl.Series("enter_mask", enter_mask),
    ])
    
    return df

def backtest_fn(df: pl.DataFrame, cfg: Dict[str, Any]):
    df = prepare_df_for_backtest(df, cfg)
    return simple_backtest(df, cfg)

def run_cpcv(cfg_path: Path) -> None:
    cfg = load_cfg(cfg_path)
    
    print("[CPCV] Loading data...")
    df = load_ohlcv(cfg)
    print(f"[CPCV] Loaded {len(df):,} rows")
    
    cpcv_cfg = cfg.get("validation", {})
    n_splits = int(cpcv_cfg.get("cpcv_splits", 5))
    n_test_groups = int(cpcv_cfg.get("cpcv_test_groups", 2))
    embargo_pct = float(cpcv_cfg.get("embargo_pct", 0.01))
    
    print(f"\n[CPCV] Starting validation ({n_splits} splits, {n_test_groups} test groups)...")
    
    data = df.to_numpy()
    folds = combinatorial_purged_cv(data, n_splits, n_test_groups, embargo_pct)
    
    train_metrics = []
    test_metrics = []
    
    for i, (train_idx, test_idx) in enumerate(folds):
        print(f"[CPCV] Fold {i+1}/{len(folds)} (train={len(train_idx):,}, test={len(test_idx):,})")
        
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
    
    result = {
        'train_metrics': {'wr': round(train_wr, 4), 'pf': round(train_pf, 3)},
        'test_metrics': {'wr': round(test_wr, 4), 'pf': round(test_pf, 3)},
        'gap_wr': round(gap_wr, 4),
        'overfitting_risk': round(overfitting_risk, 1),
        'n_folds': len(folds),
    }
    
    print("\n" + "="*60)
    print("[CPCV RESULTS]")
    print("="*60)
    print(f"Folds: {result['n_folds']}")
    print(f"\nTrain: WR {result['train_metrics']['wr']:.2%}, PF {result['train_metrics']['pf']:.2f}")
    print(f"Test:  WR {result['test_metrics']['wr']:.2%}, PF {result['test_metrics']['pf']:.2f}")
    print(f"\nGap: {result['gap_wr']:.1%}")
    print(f"Overfitting Risk: {result['overfitting_risk']:.1f}%")
    
    if result['overfitting_risk'] < 15:
        print("\n✅ 검증 통과")
    elif result['overfitting_risk'] < 30:
        print("\n⚠️ 주의")
    else:
        print("\n❌ 실패")
    
    out_path = _ROOT / "logs" / "cpcv_result.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n[SAVE] {out_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    args = p.parse_args()
    run_cpcv(Path(args.config))