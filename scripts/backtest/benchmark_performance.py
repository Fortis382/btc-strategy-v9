# scripts/backtest/benchmark_performance.py
"""v9.4 Benchmark — tracemalloc + RSS delta"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import argparse, time, json, platform, tracemalloc

try:
    import psutil
except ImportError:
    psutil = None

from run_backtest import backtest_once


def _sysinfo() -> Dict[str, Any]:
    """시스템 정보"""
    info = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown"
    }
    if psutil:
        info["cpu_count"] = psutil.cpu_count(logical=True)
        info["memory_gb"] = round(psutil.virtual_memory().total / 1e9, 2)
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("logs/bench"))
    ap.add_argument("--target", type=float, default=150.0)
    args = ap.parse_args()
    
    args.out.mkdir(parents=True, exist_ok=True)
    
    print("[BENCH] System:", _sysinfo())
    print(f"[BENCH] Repeat={args.repeat}, Target={args.target}s\n")
    
    runs = []
    for i in range(1, args.repeat + 1):
        # RSS 전후 측정 (참고용)
        if psutil:
            proc = psutil.Process()
            rss_before = proc.memory_info().rss
        
        # tracemalloc (정확)
        tracemalloc.start()
        t0 = time.time()
        
        m = backtest_once(args.config)
        
        elapsed = time.time() - t0
        curr_py, peak_py = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        rec = {
            "run": i,
            "elapsed_sec": round(elapsed, 3),
            "total_trades": m["total_trades"],
            "py_peak_mb": round(peak_py / 1e6, 2),
            "py_current_mb": round(curr_py / 1e6, 2)
        }
        
        if psutil:
            rss_after = proc.memory_info().rss
            rss_delta = (rss_after - rss_before) / 1e6
            rec["rss_delta_mb"] = round(rss_delta, 2)
        
        runs.append(rec)
        print(f"  [{i}] {elapsed:.2f}s, trades={m['total_trades']}, "
              f"py_peak={rec['py_peak_mb']}MB" +
              (f", rss_delta={rec.get('rss_delta_mb')}MB" if psutil else ""))
    
    # 통계
    el = [r["elapsed_sec"] for r in runs]
    avg_el = sum(el) / len(el)
    
    result = {
        "system": _sysinfo(),
        "runs": runs,
        "stats": {
            "avg_elapsed_sec": round(avg_el, 3),
            "min_elapsed_sec": round(min(el), 3),
            "max_elapsed_sec": round(max(el), 3),
            "target_sec": args.target,
            "target_achieved": avg_el <= args.target
        }
    }
    
    if psutil:
        py_peaks = [r["py_peak_mb"] for r in runs]
        result["stats"]["avg_py_peak_mb"] = round(sum(py_peaks) / len(py_peaks), 2)
    
    (args.out / "bench.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), "utf-8"
    )
    
    print("\n" + "="*60)
    print("[BENCHMARK]")
    print(f"  Time: avg={avg_el:.2f}s  min={min(el):.2f}s  max={max(el):.2f}s")
    if psutil:
        print(f"  Python Peak: avg={result['stats']['avg_py_peak_mb']}MB")
    print(f"  Target: {args.target}s")
    print(f"  Status: {'✅ PASS' if result['stats']['target_achieved'] else '❌ FAIL'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()