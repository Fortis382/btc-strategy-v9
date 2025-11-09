# scripts/backtest/run_backtest.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import polars as pl
import yaml

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
    
_INT_DTYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}

def _is_int_dtype(dt: pl.PolarsDataType) -> bool:
    return dt in _INT_DTYPES

from src.signals.indicators import add_indicators
from src.core.scoring import score_and_gate

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
    """
    Polars quantile with empty series fallback
    
    Returns:
        분위수 값, 비어있으면 0.0 (게이트 통과 봉 없음 = 임계값 0)
    """
    if series.len() == 0:
        return 0.0
    
    result = series.quantile(pct)
    return float(result) if result is not None else 0.0

def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def save_csv(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(str(path))

# scripts/backtest/run_backtest.py 수정 (compute_thresholds 함수 교체)

def compute_thresholds_ewq(
    df: pl.DataFrame,
    cfg: Dict[str, Any]
) -> Tuple[float, float, Dict[str, Any]]:
    """
    EWQ 기반 동적 임계값 계산
    
    Returns:
        thr_c: cand 임계 (마지막 phi 값)
        thr_e: enter 임계 (phi + bias)
        info: 디버깅 정보
    """
    from src.signals.ewq_numba import ewq_batch_numba
    import numpy as np
    
    # EWQ 파라미터
    ewq_cfg = cfg.get("ewq", {})
    q_init = float(ewq_cfg.get("initial_threshold", 70.0))
    theta = float(ewq_cfg.get("theta", 0.7))
    alpha = float(ewq_cfg.get("alpha", 0.05))
    daily_cap = float(ewq_cfg.get("daily_cap", 0.03))
    tf_per_day = int(ewq_cfg.get("tf_per_day", 96))
    
    # 스코어를 [-100, 100] 범위로 스케일링 (EWQ가 이해하기 쉬운 범위)
    scores_norm = df["score"].to_numpy() * 100.0  # [-1,1] → [-100,100]
    
    # EWQ 계산
    phis = ewq_batch_numba(q_init, scores_norm, theta, alpha, daily_cap, tf_per_day)
    
    # 마지막 phi를 임계값으로 사용
    thr_c = float(phis[-1] / 100.0)  # [-100,100] → [-1,1]
    
    # Enter는 bias 추가
    bias = float(cfg["scoring"].get("bias_trend_enter", 0.04))
    thr_e = thr_c + bias
    
    return thr_c, thr_e, {
        "thr_mode": "ewq",
        "q_init": q_init,
        "theta": theta,
        "alpha": alpha,
        "daily_cap": daily_cap,
        "phi_last": phis[-1],
        "thr_c_ewq": thr_c,
        "thr_e_ewq": thr_e,
    }

# 58번째 줄 이후에 추가

def compute_thresholds(scores: pl.Series, cfg: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
    """
    기존 분위 기반 임계값 계산
    
    Args:
        scores: 스코어 Series (gate_ok 필터링된 값)
        cfg: 설정
    
    Returns:
        thr_c: cand 임계값
        thr_e: enter 임계값
        info: 디버깅 정보
    """
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
        "thr_c_auto": thr_c,
        "thr_e_auto": thr_e,
    }
    
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

    # ✅ MDD Breaker
    mdd_threshold = float(risk.get("mdd_breaker", 0.05))
    enable_mdd_breaker = bool(risk.get("enable_mdd_breaker", False))

    rows = []
    i = 0
    n = len(df)
    
    # ✅ 수정: 초기값 0.0 (첫 거래 후 업데이트)
    cumulative_R = 0.0
    peak_R = 0.0

    while i < n:
        # ✅ 수정: MDD 계산 (peak_R > 0일 때만)
        if enable_mdd_breaker and peak_R > 0:
            dd_R = peak_R - cumulative_R  # 절대값 (R 단위)
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
            
            # ✅ R 업데이트
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
        result = {"winrate": 0.0, "pf": 0.0, "expR": 0.0, "mdd_R": 0.0, "avg_hold_bars": 0}
        return trades, result

    wins = trades.filter(pl.col("rr") > 0)
    losses = trades.filter(pl.col("rr") < 0)
    gross_win = float(wins["rr"].sum()) if wins.height else 0.0
    gross_loss_abs = -float(losses["rr"].sum()) if losses.height else 0.0

    pf = (gross_win / gross_loss_abs) if gross_loss_abs > 1e-12 else float("inf")
    winrate = float((trades["rr"] > 0).mean())
    expR = float(trades["rr"].mean())
    avg_hold = int(float(trades["bars"].mean()))

    rr_series = trades["rr"].fill_null(0.0)
    eq = rr_series.cum_sum()
    peak = eq.cum_max()
    dd = (peak - eq)
    mdd_R = float(dd.max()) if dd.len() > 0 else 0.0

    result = {
        "winrate": round(winrate, 4),
        "pf": round(pf, 3) if pf != float("inf") else float("inf"),
        "expR": round(expR, 4),
        "mdd_R": round(mdd_R, 3),
        "avg_hold_bars": int(avg_hold),
    }
    return trades, result

def run(cfg_path: Path, quiet: bool = False) -> None:
    cfg = load_cfg(cfg_path)

    df = load_ohlcv(cfg)
    df = add_indicators(df, cfg)
    df = score_and_gate(df, cfg)

    # ===== 임계값 계산 =====
    dbg = cfg.get("debug", {})
    use_ewq = bool(cfg.get("use_ewq", False))
    use_gate = not bool(dbg.get("no_gate", False))  # ✅ 항상 정의
    
    if use_ewq:
        thr_c, thr_e, thr_info = compute_thresholds_ewq(df, cfg)
        pool_tag = "ewq"
        pool_size = int(df.height)
    else:
        if use_gate:
            score_pool = df.filter(pl.col("gate_ok"))["score"]
            pool_tag = "gate_ok"
            pool_size = int(score_pool.len())
        else:
            score_pool = df["score"]
            pool_tag = "all"
            pool_size = int(score_pool.len())
        
        thr_c, thr_e, thr_info = compute_thresholds(score_pool, cfg)
    
    thr_info.update({
        "pool": pool_tag,
        "pool_size": pool_size,
    })

    df = df.with_columns([
        pl.col("score").alias("score_enter"),
        pl.col("score").alias("score_cand"),
    ])

    g = cfg["gates"]

    # ✅ use_gate 기반 base_gate 생성 (EWQ 무관)
    if use_gate:
        base_gate = df["gate_ok"]
    else:
        base_gate = pl.Series([True] * len(df), dtype=pl.Boolean)

    cand_mask = (df["score_cand"] >= thr_c)
    enter_mask = (df["score_enter"] >= thr_e)

    if int(g.get("cooloff_bars", 0)) > 0:
        cand_mask = cooloff_mask(cand_mask, int(g["cooloff_bars"]))

    cand_mask = cand_mask & base_gate
    enter_mask = enter_mask & base_gate

    df = df.with_columns([
        pl.Series("cand_mask", cand_mask),
        pl.Series("enter_mask", enter_mask),
    ])
    
    # ✅ 신규: SLO 검증
    slo = cfg.get("slo", {})
    if slo.get("enable_validation", False):
        # 처리 시간 측정 (백테스트 시간 / 데이터 길이)
        import time
        t0 = time.perf_counter()
        trades, result = simple_backtest(df, cfg)
        t1 = time.perf_counter()
        
        processing_time_per_bar = (t1 - t0) / len(df)
        slo_target = float(slo.get("processing_time_per_bar_s", 0.001))  # 1ms
        
        if processing_time_per_bar > slo_target:
            print(f"[SLO WARN] Processing time: {processing_time_per_bar*1000:.2f}ms > {slo_target*1000:.0f}ms")
        else:
            print(f"[SLO OK] Processing time: {processing_time_per_bar*1000:.2f}ms")
            
    stats = {
        "score_q25": float(df["score"].quantile(0.25)),
        "score_q50": float(df["score"].quantile(0.50)),
        "score_q75": float(df["score"].quantile(0.75)),
        "score_q90": float(df["score"].quantile(0.90)),
        "ema21_slope_n_mean": float(df["ema21_slope_n"].mean()),
        "adx_n_mean": float(df["adx_n"].mean()),
        "rows": df.height,
        "gate_ok_rows": int(df["gate_ok"].sum()),
        "gate_ok_rate": float(df["gate_ok"].mean()),
    }
    counts = {
        "total_rows": df.height,
        "score_ge_cand": int((df["score"] >= thr_c).sum()),
        "score_ge_enter": int((df["score"] >= thr_e).sum()),
        "gate_ok": int(df["gate_ok"].sum()),
        "cand_mask_true": int(df["cand_mask"].sum()),
    }

    trades, result = simple_backtest(df, cfg)

    out = {
        "winrate": result["winrate"],
        "pf": result["pf"],
        "expR": result["expR"],
        "mdd_R": result["mdd_R"],
        "avg_hold_bars": result["avg_hold_bars"],
        "thr_used": {"cand": round(float(thr_c), 6), "enter": round(float(thr_e), 6)},
        "counts": {"trades": trades.height},
        "breakdown": {
            "stats": stats,
            "counts": counts,
            "gate_used": "normal",
            **thr_info,
        }
    }

    trades_path = _ROOT / cfg["output"]["trades_csv"]
    dbg_path = _ROOT / cfg["output"]["dbg_json"]
    save_csv(trades, trades_path)
    save_json(out, dbg_path)

    if not quiet:
        print("[RESULT]", out)
        print("[SAVE]", trades_path)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--quiet", action="store_true", default=False)
    args = p.parse_args()
    run(Path(args.config), quiet=bool(args.quiet))