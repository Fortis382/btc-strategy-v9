# scripts/backtest/run_backtest.py
"""
v9.4 Backtest Entry Point - Final Corrected Version
- 버그 1~6 전부 수정
- Polars 엔진 / loop fallback 명확히 분리
- enter_mask 생성/사용 일관성 확보
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import polars as pl
import yaml

# 프로젝트 루트
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 기존 import
from src.signals.indicators import add_indicators
from src.core.scoring import score_and_gate

# 신규 import (Polars 엔진)
try:
    from scripts.backtest.backtest_polars import BacktestEngine, _read_parquet_any
    POLARS_ENGINE_AVAILABLE = True
except ImportError:
    POLARS_ENGINE_AVAILABLE = False
    print("[WARN] backtest_polars not available, falling back to simple loop")


# ========== Utils (unchanged) ==========
def load_cfg(path: Path) -> Dict[str, Any]:
    """YAML config load"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    for key in ["data", "gates", "risk", "output"]:
        if key not in cfg:
            raise KeyError(f"Missing config key: {key}")
    
    return cfg


def load_ohlcv(cfg: Dict[str, Any]) -> pl.DataFrame:
    """OHLCV data load"""
    try:
        from src.core.loader_polars import load_ohlcv as _load
        return _load(
            _ROOT,
            cfg["data"]["path_primary"],
            cfg["data"]["path_fallback"],
            cfg["data"].get("start"),
            cfg["data"].get("end")
        )
    except (ImportError, KeyError):
        data_path = Path(cfg["data"].get("path_primary", "data/processed/BTCUSDT_15m_cleaned.parquet"))
        if not data_path.is_absolute():
            data_path = _ROOT / data_path
        return pl.read_parquet(data_path)


def quantile(series: pl.Series, pct: float) -> float:
    """Polars quantile with empty series fallback"""
    if series.len() == 0:
        return 0.0
    result = series.quantile(pct)
    return float(result) if result is not None else 0.0


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """JSON save"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(df: pl.DataFrame, path: Path) -> None:
    """CSV save"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(str(path))


# ========== Cooloff (fixed) ==========
def cooloff_mask(mask: pl.Series, bars: int) -> pl.Series:
    """
    Cooloff mask (signal blocking for N bars after trigger)
    
    Example:
        input:  [F, T, T, F, T, F]  (bars=2)
        output: [F, T, F, F, T, F]
    """
    if bars <= 0:
        return mask
    
    arr = mask.to_list()
    n = len(arr)
    result = [False] * n
    block = 0
    
    for i in range(n):
        if block > 0:
            result[i] = False
            block -= 1
        elif arr[i]:
            result[i] = True
            block = bars
        else:
            result[i] = False
    
    return pl.Series(result, dtype=pl.Boolean)


# ========== Threshold Computation ==========
def compute_thresholds_ewq(
    df: pl.DataFrame,
    cfg: Dict[str, Any],
    use_gate: bool = True
) -> Tuple[float, float, Dict[str, Any]]:
    """EWQ-based dynamic threshold"""
    from src.signals.ewq_numba import ewq_batch_numba
    import numpy as np
    
    ewq_cfg = cfg.get("ewq", {})
    q_init = float(ewq_cfg.get("initial_threshold", 70.0))
    theta = float(ewq_cfg.get("theta", 0.7))
    alpha = float(ewq_cfg.get("alpha", 0.05))
    daily_cap = float(ewq_cfg.get("daily_cap", 0.03))
    tf_per_day = int(ewq_cfg.get("tf_per_day", 96))
    
    if use_gate and "gate_ok" in df.columns:
        scores_series = df.filter(pl.col("gate_ok"))["score"]
        pool_tag = "gate_ok"
    else:
        scores_series = df["score"]
        pool_tag = "all"
    
    scores_norm = scores_series.to_numpy() * 100.0
    phis = ewq_batch_numba(q_init, scores_norm, theta, alpha, daily_cap, tf_per_day)
    
    thr_c = float(phis[-1] / 100.0)
    bias = float(cfg.get("scoring", {}).get("bias_trend_enter", 0.04))
    thr_e = thr_c + bias
    
    return thr_c, thr_e, {
        "thr_mode": "ewq",
        "q_init": q_init,
        "theta": theta,
        "alpha": alpha,
        "phi_last": phis[-1],
        "thr_c": thr_c,
        "thr_e": thr_e,
        "pool": pool_tag,
        "pool_size": len(scores_norm),
    }


def compute_thresholds_quantile(
    scores: pl.Series,
    cfg: Dict[str, Any]
) -> Tuple[float, float, Dict[str, Any]]:
    """Quantile-based threshold"""
    dbg = cfg.get("debug", {})
    auto = dbg.get("auto_thresholds", {"cand_pct": 0.75, "enter_pct": 0.85})
    cand_pct = float(auto.get("cand_pct", 0.75))
    enter_pct = float(auto.get("enter_pct", 0.85))
    
    thr_c = quantile(scores, cand_pct)
    thr_e = quantile(scores, enter_pct)
    
    if dbg.get("force_cand_from_score", False):
        thr_c = quantile(scores, float(dbg.get("force_pct", cand_pct)))
    
    return thr_c, thr_e, {
        "thr_mode": "auto_pct",
        "cand_pct": cand_pct,
        "enter_pct": enter_pct,
        "thr_c": thr_c,
        "thr_e": thr_e,
    }


# ========== Backtest (Polars Engine) ==========
def backtest_with_engine(
    df: pl.DataFrame,
    cfg: Dict[str, Any],
    skip_indicators: bool = False
) -> Tuple[pl.DataFrame, Dict[str, float]]:
    """
    backtest_polars engine (v9.4)
    
    주의: 이 함수는 df에 이미 score가 있다고 가정하지 않음
          BacktestEngine 내부에서 score 재계산함
    """
    if not POLARS_ENGINE_AVAILABLE:
        raise ImportError("backtest_polars not available")
    
    # cfg 변환 (버그 1 수정: indicators 키 안전 접근)
    atr_len = int(cfg.get("indicators", {}).get("atr", 14))
    
    cfg_v9 = {
        "warmup_bars": cfg.get("warmup_bars", 300),
        "max_hold_bars": int(cfg["risk"].get("max_hold_min", 720) // 15),
        "side": cfg.get("side", "both"),
        
        "gate": {
            "mode": cfg.get("gate_mode", "fixed"),
            "fixed_threshold": cfg.get("fixed_threshold", 0.02),
            "phi": cfg.get("phi", 0.75),
            "ewm_alpha": cfg.get("ewm_alpha", 0.05),
            "min_sigma": 1e-6,
        },
        
        "risk": {
            "max_risk_per_trade": cfg["risk"].get("atr_sl", 1.0) * 0.01,
        },
        
        "cols": {
            "ts": "ts",
            "close": "close",
            "atr": f"atr{atr_len}_abs",  # ← 버그 1 수정
        },
        
        "weights": cfg.get("weights", {}),
        
        "cache": {
            "enabled": cfg.get("cache_enabled", False),
            "dir": cfg.get("cache_dir", "cache"),
        },
        
        # 버그 3 대응: indicators 정보 전달
        "indicators": cfg.get("indicators", {}),
    }
    
    engine = BacktestEngine(cfg_v9)
    result = engine.run(df, skip_indicators=skip_indicators)
    
    trades_v9 = result["trades"]
    metrics_v9 = result["metrics"]
    
    # trades DataFrame 변환
    if len(trades_v9) > 0:
        trades_df = pl.DataFrame(trades_v9)
        trades_df = trades_df.rename({"R": "rr"})
        trades_df = trades_df.with_columns([
            pl.lit("vectorized").alias("reason"),
            pl.lit(cfg_v9["max_hold_bars"]).alias("bars"),
            pl.col("ts").alias("entry_ts"),
            (pl.col("ts") + pl.duration(minutes=cfg_v9["max_hold_bars"] * 15)).alias("exit_ts"),
        ])
    else:
        trades_df = pl.DataFrame(schema={
            "entry_ts": pl.Datetime,
            "exit_ts": pl.Datetime,
            "rr": pl.Float64,
            "reason": pl.Utf8,
            "bars": pl.Int32,
        })
    
    # metrics 변환 (버그 4 대응: 키 일관성)
    metrics = {
        "winrate": metrics_v9["winrate"],
        "pf": metrics_v9["profit_factor"],  # v9 → 기존 키 변환
        "expR": metrics_v9["expectancy_R"],
        "mdd_R": metrics_v9["max_dd_R"],
        "avg_hold_bars": metrics_v9["avg_hold_bars"],
        "sharpe": metrics_v9.get("sharpe", 0.0),
        "cand_ratio": metrics_v9.get("cand_ratio", 0.0),
        "avg_gap_bars": metrics_v9.get("avg_gap_bars", 0.0),
    }
    
    return trades_df, metrics


# ========== Backtest (Loop Fallback) ==========
def simple_backtest_loop(
    df: pl.DataFrame,
    cfg: Dict[str, Any]
) -> Tuple[pl.DataFrame, Dict[str, float]]:
    """
    Simple loop backtest (fallback)
    
    전제: df에 enter_mask 칼럼 존재해야 함
    """
    # 버그 2 대응: enter_mask 존재 확인
    if "enter_mask" not in df.columns:
        raise ValueError("simple_backtest_loop requires 'enter_mask' column in df")
    
    risk = cfg["risk"]
    atr_len = int(cfg.get("indicators", {}).get("atr", 14))
    atr_col = f"atr{atr_len}_abs"
    
    close = df["close"]
    atr = df[atr_col]
    
    tp_R = [float(x) for x in risk.get("atr_tp", [1.2, 1.3, 1.5])]
    sl_R = float(risk.get("atr_sl", 1.0))
    max_hold_bars = int(risk.get("max_hold_min", 720) // 15)
    
    mdd_threshold = float(risk.get("mdd_breaker", 0.05))
    enable_mdd = bool(risk.get("enable_mdd_breaker", False))
    
    rows = []
    i = 0
    n = len(df)
    
    cumulative_R = 0.0
    peak_R = 0.0
    
    while i < n:
        if enable_mdd and peak_R > 0:
            dd_R = peak_R - cumulative_R
            dd_pct = dd_R / peak_R
            
            if dd_pct >= mdd_threshold:
                print(f"[MDD BREAKER] Triggered at bar {i}, DD={dd_R:.2f}R ({dd_pct:.1%})")
                break
        
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
                
                for k, tp in enumerate(tp_levels, start=1):
                    if hi >= tp:
                        rr = float(tp_R[k - 1])
                        reason = f"tp{k}"
                        break
                
                if reason != "timeout":
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
            
            cumulative_R += rr
            if cumulative_R > peak_R:
                peak_R = cumulative_R
            
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
        metrics = {
            "winrate": 0.0,
            "pf": 0.0,
            "expR": 0.0,
            "mdd_R": 0.0,
            "avg_hold_bars": 0,
        }
        return trades, metrics
    
    wins = trades.filter(pl.col("rr") > 0)
    losses = trades.filter(pl.col("rr") < 0)
    
    gross_win = float(wins["rr"].sum()) if wins.height else 0.0
    gross_loss = abs(float(losses["rr"].sum())) if losses.height else 0.0
    
    pf = (gross_win / gross_loss) if gross_loss > 1e-9 else float("inf")
    winrate = float((trades["rr"] > 0).mean())
    expR = float(trades["rr"].mean())
    avg_hold = int(float(trades["bars"].mean()))
    
    eq = trades["rr"].cum_sum()
    peak = eq.cum_max()
    dd = peak - eq
    mdd_R = float(dd.max()) if dd.len() > 0 else 0.0
    
    metrics = {
        "winrate": round(winrate, 4),
        "pf": round(pf, 3) if pf != float("inf") else pf,
        "expR": round(expR, 4),
        "mdd_R": round(mdd_R, 3),
        "avg_hold_bars": avg_hold,
    }
    
    return trades, metrics


# ========== Main Entry Point (버그 3, 5, 6 수정) ==========
def run(
    cfg_path: Path,
    quiet: bool = False,
    use_polars_engine: bool = True
) -> Dict[str, Any]:
    """Backtest execution (integrated)"""
    t0 = time.time()
    
    # 1. Config load
    cfg = load_cfg(cfg_path)
    
    # 2. Data load
    df = load_ohlcv(cfg)
    
    # 3. Indicators (always)
    df = add_indicators(df, cfg)
    
    # ========== 핵심 수정: 엔진별 분기 ==========
    if use_polars_engine and POLARS_ENGINE_AVAILABLE:
        # Polars 엔진: score_and_gate 스킵 (엔진 내부에서 처리)
        if not quiet:
            print(f"\n[ENGINE] Using backtest_polars (v9.4)")
        
        # cfg에 gate_mode 설정
        cfg["gate_mode"] = cfg.get("gate_mode", "fixed")
        cfg["fixed_threshold"] = cfg.get("fixed_threshold", 0.02)
        
        # 엔진 실행 (indicators만 있는 df 전달)
        trades, result = backtest_with_engine(df, cfg, skip_indicators=True)
        
        # threshold info (엔진 내부 값 사용)
        thr_c = cfg["fixed_threshold"]
        thr_e = thr_c
        thr_info = {"thr_mode": "engine_internal"}
        pool_tag = "engine"
        
    else:
        # Loop 방식: 기존 로직 유지
        if not quiet:
            print(f"\n[ENGINE] Using simple loop (fallback)")
        
        # 4. Score & Gate
        df = score_and_gate(df, cfg)
        
        # 5. Threshold 계산
        dbg = cfg.get("debug", {})
        use_ewq = bool(cfg.get("use_ewq", False))
        use_gate = not bool(dbg.get("no_gate", False))
        
        if use_ewq:
            thr_c, thr_e, thr_info = compute_thresholds_ewq(df, cfg, use_gate=use_gate)
            pool_tag = thr_info.get("pool", "ewq")
            pool_size = thr_info.get("pool_size", int(df.height))
        else:
            if use_gate and "gate_ok" in df.columns:
                score_pool = df.filter(pl.col("gate_ok"))["score"]
                pool_tag = "gate_ok"
            else:
                score_pool = df["score"]
                pool_tag = "all"
            
            pool_size = int(score_pool.len())
            thr_c, thr_e, thr_info = compute_thresholds_quantile(score_pool, cfg)
            thr_info.update({"pool": pool_tag, "pool_size": pool_size})
        
        # 6. Mask 생성
        if use_gate and "gate_ok" in df.columns:
            base_gate = df["gate_ok"]
        else:
            base_gate = pl.Series([True] * len(df), dtype=pl.Boolean)
        
        cand_mask = (df["score"] >= thr_c)
        enter_mask = (df["score"] >= thr_e)
        
        # Cooloff
        g = cfg["gates"]
        if int(g.get("cooloff_bars", 0)) > 0:
            cooloff_bars = int(g["cooloff_bars"])
            cand_mask = cooloff_mask(cand_mask, cooloff_bars)
            enter_mask = cooloff_mask(enter_mask, cooloff_bars)
        
        # Gate AND
        cand_mask = cand_mask & base_gate
        enter_mask = enter_mask & base_gate
        
        df = df.with_columns([
            pl.Series("cand_mask", cand_mask),
            pl.Series("enter_mask", enter_mask),
        ])
        
        if not quiet:
            print(f"\n[DEBUG] Threshold info:")
            print(f"  pool: {pool_tag}, size: {pool_size}")
            print(f"  thr_c: {thr_c:.6f}, thr_e: {thr_e:.6f}")
            print(f"\n[DEBUG] Mask stats:")
            print(f"  cand_mask: {int(cand_mask.sum())} / {len(df)}")
            print(f"  enter_mask: {int(enter_mask.sum())} / {len(df)}")
        
        # 7. Loop backtest
        trades, result = simple_backtest_loop(df, cfg)
    
    elapsed = time.time() - t0
    
    # 8. Stats (방어적 접근)
    stats = {
        "score_q25": float(df["score"].quantile(0.25)) if "score" in df.columns else 0.0,
        "score_q50": float(df["score"].quantile(0.50)) if "score" in df.columns else 0.0,
        "score_q75": float(df["score"].quantile(0.75)) if "score" in df.columns else 0.0,
        "score_q90": float(df["score"].quantile(0.90)) if "score" in df.columns else 0.0,
        "rows": df.height,
        "gate_ok_rows": int(df["gate_ok"].sum()) if "gate_ok" in df.columns else 0,
        "gate_ok_rate": float(df["gate_ok"].mean()) if "gate_ok" in df.columns else 0.0,
    }
    
    counts = {
        "total_rows": df.height,
        "score_ge_cand": int((df["score"] >= thr_c).sum()) if "score" in df.columns else 0,
        "score_ge_enter": int((df["score"] >= thr_e).sum()) if "score" in df.columns else 0,
        "gate_ok": int(df["gate_ok"].sum()) if "gate_ok" in df.columns else 0,
        "cand_mask_true": int(df["cand_mask"].sum()) if "cand_mask" in df.columns else 0,
        "trades": trades.height,
    }
    
    # 9. Result dict
    out = {
        "winrate": result["winrate"],
        "pf": result["pf"],
        "expR": result["expR"],
        "mdd_R": result["mdd_R"],
        "avg_hold_bars": result["avg_hold_bars"],
        "sharpe": result.get("sharpe", 0.0),
        "cand_ratio": result.get("cand_ratio", 0.0),
        "avg_gap_bars": result.get("avg_gap_bars", 0.0),
        "elapsed_sec": round(elapsed, 3),
        "thr_used": {
            "cand": round(float(thr_c), 6),
            "enter": round(float(thr_e), 6),
        },
        "counts": counts,
        "breakdown": {
            "stats": stats,
            "gate_used": pool_tag,
            **thr_info,
        },
    }
    
    # 10. Save
    trades_path = _ROOT / cfg["output"]["trades_csv"]
    dbg_path = _ROOT / cfg["output"]["dbg_json"]
    
    save_csv(trades, trades_path)
    save_json(out, dbg_path)
    
    if not quiet:
        print(f"\n[RESULT]")
        print(f"  Trades: {counts['trades']}")
        print(f"  WR: {out['winrate']:.2%}")
        print(f"  PF: {out['pf']:.2f}")
        print(f"  ExpR: {out['expR']:.3f}R")
        print(f"  MDD: {out['mdd_R']:.2f}R")
        print(f"  Sharpe: {out['sharpe']:.2f}")
        print(f"  Cand Ratio: {out['cand_ratio']:.2%}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"\n[SAVE]")
        print(f"  Trades: {trades_path}")
        print(f"  Result: {dbg_path}")
    
    return out


# ========== CLI ==========
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="v9.4 Backtest Entry Point")
    p.add_argument("--config", type=str, required=True, help="config/settings_v9.yaml")
    p.add_argument("--quiet", action="store_true", default=False, help="Suppress output")
    p.add_argument("--use-loop", action="store_true", default=False, 
                   help="Force simple loop (disable Polars engine)")
    args = p.parse_args()
    
    result = run(
        Path(args.config),
        quiet=bool(args.quiet),
        use_polars_engine=not bool(args.use_loop)
    )
    
    print(f"\n[FINAL] WR={result['winrate']:.2%} PF={result['pf']:.2f} "
          f"ExpR={result['expR']:.3f}R Trades={result['counts']['trades']}")