# scripts/backtest/grid_search.py
"""v9.4 Grid Search - safe dotted-key expansion with run() integration"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import itertools as it
import argparse
import time
import json
import yaml
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.backtest.run_backtest import run, load_cfg


def _load_params(p: Path) -> Dict[str, List[Any]]:
    """YAML load + force list"""
    g = yaml.safe_load(p.read_text("utf-8"))
    if not isinstance(g, dict):
        raise TypeError("params must be dict")
    return {k: (v if isinstance(v, list) else [v]) for k, v in g.items()}


def _deep_update(dst: Dict, src: Dict) -> Dict:
    """Recursive deep merge (in-place)"""
    for k, v in (src or {}).items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _expand_dotted_safe(flat: Dict[str, Any]) -> Dict[str, Any]:
    """
    dotted-key to nested dict with conflict detection
    
    Examples:
        {"a.b": 1, "a.c": 2} → {"a": {"b": 1, "c": 2}}
        {"a.b": 1, "a": 2} → ValueError (conflict)
    """
    nested: Dict[str, Any] = {}
    
    for key, val in flat.items():
        if "." not in key:
            if key in nested and isinstance(nested[key], dict):
                raise ValueError(
                    f"Key conflict: '{key}' tries to overwrite nested dict from other dotted keys"
                )
            nested[key] = val
            continue
        
        parts = key.split(".")
        curr = nested
        
        for i, part in enumerate(parts[:-1]):
            if part not in curr:
                curr[part] = {}
            elif not isinstance(curr[part], dict):
                raise ValueError(
                    f"Key conflict: '{key}' tries to create nested dict at '{'.'.join(parts[:i+1])}', "
                    f"but a non-dict value already exists there"
                )
            curr = curr[part]
        
        final_key = parts[-1]
        if final_key in curr and isinstance(curr[final_key], dict):
            raise ValueError(
                f"Key conflict: '{key}' tries to set value at '{final_key}', "
                f"but a dict already exists there from other dotted keys"
            )
        
        curr[final_key] = val
    
    return nested


def _product(grid: Dict[str, List[Any]]):
    """Cartesian product"""
    keys = list(grid.keys())
    for vals in it.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--params", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("logs/grid"))
    ap.add_argument("--save-every", type=int, default=10)
    args = ap.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Base config load
    base_cfg = load_cfg(args.config)
    
    grid = _load_params(args.params)
    total = 1
    for v in grid.values():
        total *= len(v)
    print(f"[GRID] Total: {total}")
    
    rows = []
    t0 = time.time()
    
    for i, flat_ov in enumerate(_product(grid), 1):
        # dotted-key expansion
        try:
            nested_ov = _expand_dotted_safe(flat_ov)
        except ValueError as e:
            print(f"[{i}/{total}] SKIP (key conflict): {flat_ov}")
            rows.append({**flat_ov, "error": f"key_conflict: {e}"})
            continue
        
        print(f"[{i}/{total}] {flat_ov}")
        
        try:
            # Deep merge cfg
            import copy
            cfg_merged = copy.deepcopy(base_cfg)
            _deep_update(cfg_merged, nested_ov)
            
            # Create temp YAML
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
                yaml.dump(cfg_merged, f)
                temp_path = Path(f.name)
            
            try:
                # Run backtest
                result = run(temp_path, quiet=True, use_polars_engine=True)
                
                # Extract metrics (키 이름 주의: "pf" not "profit_factor")
                row = {
                    **flat_ov,
                    "winrate": result["winrate"],
                    "pf": result["pf"],  # ← 키 수정
                    "expR": result["expR"],
                    "mdd_R": result["mdd_R"],
                    "sharpe": result["sharpe"],
                    "trades": result["counts"]["trades"],
                    "elapsed_sec": result["elapsed_sec"],
                    "error": None
                }
                rows.append(row)
                
                print(f"  PF={row['pf']:.2f} WR={row['winrate']:.2%} Trades={row['trades']}")
                
            finally:
                # Cleanup temp file
                temp_path.unlink(missing_ok=True)
                
        except Exception as e:
            rows.append({**flat_ov, "error": str(e)})
            print(f"  ERROR: {e}")
        
        if i % args.save_every == 0:
            (args.out / "progress.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), "utf-8"
            )
    
    (args.out / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), "utf-8"
    )
    
    print(f"\n[GRID] Done in {time.time()-t0:.1f}s")
    
    # Best output (키 이름 수정)
    valid = [r for r in rows if r.get("error") is None]
    if valid:
        best = max(valid, key=lambda r: r.get("pf", 0))  # ← profit_factor → pf
        print(f"\n[BEST] PF={best['pf']:.2f}")
        for k, v in best.items():
            if k not in ["error", "elapsed_sec"]:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()