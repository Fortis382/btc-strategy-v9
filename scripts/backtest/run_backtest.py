# scripts/backtest/run_backtest.py
"""
v9.4 Backtest Entry Point — 최종 통합판
- 기존: loader, cooloff, MDD breaker 유지
- 신규: backtest_polars 엔진 통합 (EWM, Sharpe, cache, adaptive gate)
- 호환: 기존 YAML 구조 유지 (data, gates, risk, output)
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

# 프로젝트 루트 추가
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 기존 import (유지)
from src.signals.indicators import add_indicators
from src.core.scoring import score_and_gate

# 신규 import (backtest_polars 엔진)
try:
    from scripts.backtest.backtest_polars import BacktestEngine, _read_parquet_any
    POLARS_ENGINE_AVAILABLE = True
except ImportError:
    POLARS_ENGINE_AVAILABLE = False
    print("[WARN] backtest_polars not available, falling back to simple loop")


# ========== 유틸 (기존 유지) ==========
def load_cfg(path: Path) -> Dict[str, Any]:
    """YAML 설정 로드"""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    # 필수 키 검증
    for key in ["data", "gates", "risk", "output"]:
        if key not in cfg:
            raise KeyError(f"Missing config key: {key}")
    
    return cfg


def load_ohlcv(cfg: Dict[str, Any]) -> pl.DataFrame:
    """OHLCV 데이터 로드 (기존 loader 유지)"""
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
        # fallback: 직접 parquet 로드
        data_path = Path(cfg["data"].get("path_primary", "data/processed/BTCUSDT_15m_cleaned.parquet"))
        if not data_path.is_absolute():
            data_path = _ROOT / data_path
        return pl.read_parquet(data_path)


def quantile(series: pl.Series, pct: float) -> float:
    """Polars quantile (빈 Series 방어)"""
    if series.len() == 0:
        return 0.0
    result = series.quantile(pct)
    return float(result) if result is not None else 0.0


def save_json(obj: Dict[str, Any], path: Path) -> None:
    """JSON 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(df: pl.DataFrame, path: Path) -> None:
    """CSV 저장"""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(str(path))


# ========== Cooloff (기존 로직 유지, 버그 수정) ==========
def cooloff_mask(mask: pl.Series, bars: int) -> pl.Series:
    """
    Cooloff 마스크 (신호 후 N바 차단)
    
    Args:
        mask: 원본 시그널 (bool Series)
        bars: 쿨다운 봉 수
    
    Returns:
        쿨다운 적용된 마스크
    
    예시:
        input:  [F, T, T, F, T, F]  (bars=2)
        output: [F, T, F, F, T, F]
                     ↑ 신호   ↑ 2봉 후 재개
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


# ========== 임계값 계산 (기존 + EWQ 통합) ==========
def compute_thresholds_ewq(
    df: pl.DataFrame,
    cfg: Dict[str, Any],
    use_gate: bool = True
) -> Tuple[float, float, Dict[str, Any]]:
    """
    EWQ 기반 동적 임계값 (기존 로직 유지)
    
    Args:
        df: 전체 DataFrame
        cfg: 설정
        use_gate: gate_ok 필터링 여부
    
    Returns:
        thr_c: cand 임계값
        thr_e: enter 임계값
        info: 디버깅 정보
    """
    from src.signals.ewq_numba import ewq_batch_numba
    import numpy as np
    
    ewq_cfg = cfg.get("ewq", {})
    q_init = float(ewq_cfg.get("initial_threshold", 70.0))
    theta = float(ewq_cfg.get("theta", 0.7))
    alpha = float(ewq_cfg.get("alpha", 0.05))
    daily_cap = float(ewq_cfg.get("daily_cap", 0.03))
    tf_per_day = int(ewq_cfg.get("tf_per_day", 96))
    
    # gate 필터링
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
    """
    분위수 기반 임계값 (기존 로직)
    
    Args:
        scores: 스코어 Series (gate 필터링된 값)
        cfg: 설정
    
    Returns:
        thr_c, thr_e, info
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
        "thr_c": thr_c,
        "thr_e": thr_e,
    }


# ========== 백테스트 (신규: Polars 엔진 통합) ==========
def backtest_with_engine(
    df: pl.DataFrame,
    cfg: Dict[str, Any],
    skip_indicators: bool = False
) -> Tuple[pl.DataFrame, Dict[str, float]]:
    """
    backtest_polars 엔진 사용 (v9.4 방식)
    
    Args:
        df: OHLCV + indicators DataFrame
        cfg: 설정 (v9.4 형식으로 변환)
        skip_indicators: indicators 재계산 스킵
    
    Returns:
        trades: 거래 DataFrame
        metrics: {winrate, pf, expR, mdd_R, sharpe, ...}
    """
    if not POLARS_ENGINE_AVAILABLE:
        raise ImportError("backtest_polars not available")
    
    # cfg 변환 (기존 → v9.4)
    cfg_v9 = {
        "warmup_bars": cfg.get("warmup_bars", 300),
        "max_hold_bars": int(cfg["risk"].get("max_hold_min", 720) // 15),
        "side": cfg.get("side", "both"),
        
        "gate": {
            "mode": cfg.get("gate_mode", "fixed"),  # fixed | adaptive_ewm_z
            "fixed_threshold": cfg.get("fixed_threshold", 0.02),
            "phi": cfg.get("phi", 0.75),
            "ewm_alpha": cfg.get("ewm_alpha", 0.05),
            "min_sigma": 1e-6,
        },
        
        "risk": {
            "max_risk_per_trade": cfg["risk"].get("atr_sl", 1.0) * 0.01,  # R 정규화
        },
        
        "cols": {
            "ts": "ts",
            "close": "close",
            "atr": f"atr{cfg['indicators']['atr']}_abs",
        },
        
        "weights": cfg.get("weights", {}),
        
        "cache": {
            "enabled": cfg.get("cache_enabled", False),
            "dir": cfg.get("cache_dir", "cache"),
        },
    }
    
    # 엔진 실행
    engine = BacktestEngine(cfg_v9)
    result = engine.run(df, skip_indicators=skip_indicators)
    
    # 결과 변환 (v9.4 → 기존 형식)
    trades_v9 = result["trades"]
    metrics_v9 = result["metrics"]
    
    # trades DataFrame 변환
    if len(trades_v9) > 0:
        trades_df = pl.DataFrame(trades_v9)
        trades_df = trades_df.rename({"R": "rr"})
        # 기존 형식에 맞게 추가 칼럼
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
    
    # metrics 변환
    metrics = {
        "winrate": metrics_v9["winrate"],
        "pf": metrics_v9["profit_factor"],
        "expR": metrics_v9["expectancy_R"],
        "mdd_R": metrics_v9["max_dd_R"],
        "avg_hold_bars": metrics_v9["avg_hold_bars"],
        "sharpe": metrics_v9.get("sharpe", 0.0),
        "cand_ratio": metrics_v9.get("cand_ratio", 0.0),
        "avg_gap_bars": metrics_v9.get("avg_gap_bars", 0.0),
    }
    
    return trades_df, metrics


def simple_backtest_loop(
    df: pl.DataFrame,
    cfg: Dict[str, Any]
) -> Tuple[pl.DataFrame, Dict[str, float]]:
    """
    기존 loop 백테스트 (fallback)
    
    Args:
        df: enter_mask가 있는 DataFrame
        cfg: 설정
    
    Returns:
        trades, metrics
    """
    risk = cfg["risk"]
    atr_len = int(cfg["indicators"]["atr"])
    atr_col = f"atr{atr_len}_abs"
    
    close = df["close"]
    atr = df[atr_col]
    
    tp_R = [float(x) for x in risk.get("atr_tp", [1.2, 1.3, 1.5])]
    sl_R = float(risk.get("atr_sl", 1.0))
    max_hold_bars = int(risk.get("max_hold_min", 720) // 15)
    
    # MDD breaker
    mdd_threshold = float(risk.get("mdd_breaker", 0.05))
    enable_mdd = bool(risk.get("enable_mdd_breaker", False))
    
    rows = []
    i = 0
    n = len(df)
    
    cumulative_R = 0.0
    peak_R = 0.0
    
    while i < n:
        # MDD 체크
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
    
    # trades DataFrame
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
    
    # metrics
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
    
    # MDD
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


# ========== 메인 진입점 ==========
def run(
    cfg_path: Path,
    quiet: bool = False,
    use_polars_engine: bool = True
) -> Dict[str, Any]:
    """
    백테스트 실행 (통합)
    
    Args:
        cfg_path: 설정 파일 경로
        quiet: 출력 억제
        use_polars_engine: True면 backtest_polars 엔진, False면 loop
    
    Returns:
        결과 dict
    """
    t0 = time.time()
    
    # 1. 설정 로드
    cfg = load_cfg(cfg_path)
    
    # 2. 데이터 로드
    df = load_ohlcv(cfg)
    
    # 3. Indicators
    df = add_indicators(df, cfg)
    
    # 4. Score & Gate (기존 로직 유지)
    df = score_and_gate(df, cfg)
    
    # 5. 임계값 계산
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
    
    # 6. Mask 생성 (기존 로직)
    if use_gate and "gate_ok" in df.columns:
        base_gate = df["gate_ok"]
    else:
        base_gate = pl.Series([True] * len(df), dtype=pl.Boolean)
    
    cand_mask = (df["score"] >= thr_c)
    enter_mask = (df["score"] >= thr_e)
    
    # Cooloff 적용
    g = cfg["gates"]
    if int(g.get("cooloff_bars", 0)) > 0:
        cand_mask = cooloff_mask(cand_mask, int(g["cooloff_bars"]))
    
    # Gate AND
    cand_mask = cand_mask & base_gate
    enter_mask = enter_mask & base_gate
    
    df = df.with_columns([
        pl.Series("cand_mask", cand_mask),
        pl.Series("enter_mask", enter_mask),
    ])
    
    # 디버깅 출력
    if not quiet:
        print(f"\n[DEBUG] Threshold info:")
        print(f"  pool: {pool_tag}, size: {pool_size}")
        print(f"  thr_c: {thr_c:.6f}, thr_e: {thr_e:.6f}")
        print(f"\n[DEBUG] Mask stats:")
        print(f"  cand_mask: {int(cand_mask.sum())} / {len(df)}")
        print(f"  enter_mask: {int(enter_mask.sum())} / {len(df)}")
    
    # 7. 백테스트 실행
    if use_polars_engine and POLARS_ENGINE_AVAILABLE:
        if not quiet:
            print(f"\n[ENGINE] Using backtest_polars (v9.4)")
        
        # cfg에 gate_mode 추가 (adaptive_ewm_z 또는 fixed)
        cfg["gate_mode"] = "fixed"  # 기본값
        cfg["fixed_threshold"] = thr_c
        
        trades, result = backtest_with_engine(df, cfg, skip_indicators=True)
    else:
        if not quiet:
            print(f"\n[ENGINE] Using simple loop (fallback)")
        trades, result = simple_backtest_loop(df, cfg)
    
    elapsed = time.time() - t0
    
    # 8. 통계
    stats = {
        "score_q25": float(df["score"].quantile(0.25)),
        "score_q50": float(df["score"].quantile(0.50)),
        "score_q75": float(df["score"].quantile(0.75)),
        "score_q90": float(df["score"].quantile(0.90)),
        "rows": df.height,
        "gate_ok_rows": int(base_gate.sum()),
        "gate_ok_rate": float(base_gate.mean()),
    }
    
    counts = {
        "total_rows": df.height,
        "score_ge_cand": int((df["score"] >= thr_c).sum()),
        "score_ge_enter": int((df["score"] >= thr_e).sum()),
        "gate_ok": int(base_gate.sum()),
        "cand_mask_true": int(cand_mask.sum()),
        "trades": trades.height,
    }
    
    # 9. 결과 dict
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
    
    # 10. 저장
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