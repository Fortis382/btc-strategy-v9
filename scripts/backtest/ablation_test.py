# scripts/backtest/ablation_test.py
"""v9.4 Ablation (seed 고정 + weight=0)"""
from __future__ import annotations
from pathlib import Path
from typing import Dict
import argparse, json

from run_backtest import backtest_once


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--factors", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("logs/ablation"))
    ap.add_argument("--seed", type=int, default=42, help="고정 seed (재현성)")
    args = ap.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    # Baseline (seed 고정)
    print("[ABLATION] Baseline...")
    base = backtest_once(args.config, {"seed": args.seed})
    rows = [{"case": "BASELINE", **base}]
    base_pf, base_wr = base["profit_factor"], base["winrate"]
    print(f"  PF={base_pf:.2f}, WR={base_wr:.2%}")
    
    # 각 factor 제거
    for f in args.factors:
        print(f"\n[ABLATION] Remove: {f}")
        ov = {"seed": args.seed, "weights": {f: 0.0}}
        
        try:
            m = backtest_once(args.config, ov)
            pf, wr = m["profit_factor"], m["winrate"]
            
            row = {
                "case": f"NO_{f.upper()}",
                **m,
                "pf_delta": pf - base_pf,
                "wr_delta": wr - base_wr,
                "pf_change_pct": ((pf - base_pf) / base_pf * 100) if base_pf > 0 else 0,
            }
            print(f"  PF={pf:.2f} (Δ{row['pf_delta']:+.2f}), WR={wr:.2%}")
        except Exception as e:
            row = {"case": f"NO_{f.upper()}", "error": str(e)}
        
        rows.append(row)
    
    out = args.out / "ablation.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), "utf-8")
    
    # 중요도 순위
    valid = [r for r in rows if r.get("error") is None and r["case"] != "BASELINE"]
    if valid:
        ranked = sorted(valid, key=lambda r: r["pf_delta"])
        print("\n[FACTOR IMPORTANCE] (제거시 성능 하락 큰 순)")
        for r in ranked:
            print(f"  {r['case']}: PF Δ={r['pf_delta']:+.2f} ({r['pf_change_pct']:+.1f}%)")


if __name__ == "__main__":
    main()