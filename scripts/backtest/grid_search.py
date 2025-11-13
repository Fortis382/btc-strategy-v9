# scripts/backtest/grid_search.py
"""v9.4 Grid Search — safe dotted-key expansion"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List
import itertools as it, argparse, time, json, yaml

from run_backtest import run, load_cfg


def _load_params(p: Path) -> Dict[str, List[Any]]:
    """YAML 로드 + 리스트 강제"""
    g = yaml.safe_load(p.read_text("utf-8"))
    if not isinstance(g, dict):
        raise TypeError("params must be dict")
    return {k: (v if isinstance(v, list) else [v]) for k, v in g.items()}


def _expand_dotted_safe(flat: Dict[str, Any]) -> Dict[str, Any]:
    """dotted-key를 중첩 dict로 확장 (충돌 검증)
    
    Examples:
        {"a.b": 1, "a.c": 2} → {"a": {"b": 1, "c": 2}}
        {"a.b": 1, "a": 2} → ValueError (충돌)
    """
    nested: Dict[str, Any] = {}
    
    for key, val in flat.items():
        if "." not in key:
            # 평범한 키
            if key in nested and isinstance(nested[key], dict):
                raise ValueError(
                    f"Key conflict: '{key}' tries to overwrite nested dict from other dotted keys"
                )
            nested[key] = val
            continue
        
        # dotted key 처리
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
    
    grid = _load_params(args.params)
    total = 1
    for v in grid.values():
        total *= len(v)
    print(f"[GRID] Total: {total}")
    
    rows = []
    t0 = time.time()
    
    for i, flat_ov in enumerate(_product(grid), 1):
        # dotted-key 확장 (충돌 검증)
        try:
            nested_ov = _expand_dotted_safe(flat_ov)
        except ValueError as e:
            print(f"[{i}/{total}] SKIP (key conflict): {flat_ov}")
            rows.append({**flat_ov, "error": f"key_conflict: {e}"})
            continue
        
        print(f"[{i}/{total}] {flat_ov}")
        
        try:
            m = run(args.config, nested_ov)
            rows.append({**flat_ov, **m, "error": None})
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
    
    # Best 출력
    valid = [r for r in rows if r.get("error") is None]
    if valid:
        best = max(valid, key=lambda r: r.get("profit_factor", 0))
        print(f"\n[BEST] PF={best['profit_factor']:.2f}")
        for k, v in best.items():
            if k not in ["error", "elapsed_sec"]:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()