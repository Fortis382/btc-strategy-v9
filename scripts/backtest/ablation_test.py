# scripts/backtest/ablation_test.py
"""v9.4 Ablation - seed fixed + weight=0 with run() integration"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import argparse
import json
import yaml
import sys
import tempfile
import copy

sys.path.insert(0, str(Path(__file__).parents[2]))
from scripts.backtest.run_backtest import run, load_cfg


def _deep_update(dst: Dict, src: Dict) -> Dict:
    """Recursive deep merge"""
    for k, v in (src or {}).items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst


def _run_with_override(base_cfg: Dict, override: Dict, quiet: bool = True) -> Dict:
    """Helper: merge cfg + run backtest"""
    cfg_merged = copy.deepcopy(base_cfg)
    _deep_update(cfg_merged, override)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        yaml.dump(cfg_merged, f)
        temp_path = Path(f.name)
    
    try:
        result = run(temp_path, quiet=quiet, use_polars_engine=True)
        return result
    finally:
        temp_path.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--factors", nargs="+", required=True,
                    help="Factor names: trend, momentum, volatility, participation, location")
    ap.add_argument("--out", type=Path, default=Path("logs/ablation"))
    ap.add_argument("--seed", type=int, default=42, help="Fixed seed (reproducibility)")
    args = ap.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Base config
    base_cfg = load_cfg(args.config)
    
    # Baseline (seed fixed)
    print("[ABLATION] Baseline...")
    base_result = _run_with_override(base_cfg, {"seed": args.seed})
    
    base_pf = base_result["pf"]
    base_wr = base_result["winrate"]
    
    rows = [{
        "case": "BASELINE",
        "pf": base_pf,
        "winrate": base_wr,
        "expR": base_result["expR"],
        "trades": base_result["counts"]["trades"],
    }]
    
    print(f"  PF={base_pf:.2f}, WR={base_wr:.2%}, Trades={base_result['counts']['trades']}")
    
    # Remove each factor
    for f in args.factors:
        print(f"\n[ABLATION] Remove: {f}")
        
        override = {
            "seed": args.seed,
            "weights": {f: 0.0}
        }
        
        try:
            result = _run_with_override(base_cfg, override)
            
            pf = result["pf"]
            wr = result["winrate"]
            
            row = {
                "case": f"NO_{f.upper()}",
                "pf": pf,
                "winrate": wr,
                "expR": result["expR"],
                "trades": result["counts"]["trades"],
                "pf_delta": pf - base_pf,
                "wr_delta": wr - base_wr,
                "pf_change_pct": ((pf - base_pf) / base_pf * 100) if base_pf > 0 else 0,
                "error": None
            }
            
            print(f"  PF={pf:.2f} (Δ{row['pf_delta']:+.2f}), WR={wr:.2%}, Trades={row['trades']}")
            
        except Exception as e:
            row = {
                "case": f"NO_{f.upper()}",
                "error": str(e)
            }
            print(f"  ERROR: {e}")
        
        rows.append(row)
    
    # Save
    out_path = args.out / "ablation.json"
    out_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    
    # Factor importance ranking
    valid = [r for r in rows if r.get("error") is None and r["case"] != "BASELINE"]
    if valid:
        ranked = sorted(valid, key=lambda r: r["pf_delta"])
        print("\n[FACTOR IMPORTANCE] (removal impact, worst to best)")
        for r in ranked:
            print(f"  {r['case']}: PF Δ={r['pf_delta']:+.2f} ({r['pf_change_pct']:+.1f}%)")


if __name__ == "__main__":
    main()