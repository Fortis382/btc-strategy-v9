# scripts/backtest/run_backtest.py
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import polars as pl
import yaml

# --- project root on sys.path (robust) ---
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]  # .../btc-v9
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
    
# 버전 무관 dtype 체크 유틸
_INT_DTYPES = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}

def _is_int_dtype(dt: pl.PolarsDataType) -> bool:
    return dt in _INT_DTYPES

# Now safe to import our modules
from src.signals.indicators import add_indicators
from src.core.scoring import score_and_gate

from typing import Optional, List
import polars as pl
from pathlib import Path

def _collect_parquet_files(path: Path) -> List[str]:
    """path가 디렉터리면 재귀로 *.parquet/*.parq/*.pq 수집, 파일이면 그대로."""
    if path.is_dir():
        exts = ("*.parquet", "*.parq", "*.pq")
        files: List[str] = []
        for ext in exts:
            files += [str(p) for p in path.rglob(ext)]
        return files
    elif path.is_file():
        return [str(path)]
    return []

def _read_parquet_any(path: Path) -> Optional[pl.DataFrame]:
    files = _collect_parquet_files(path)
    if not files:
        return None
    # 여러 파일을 한 번에 로드 (스키마 자동 합치기)
    return pl.read_parquet(files, use_pyarrow=True)

def _standardize_ohlcv(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return df

    # 1) 전체 소문자
    lower_map = {c: c.lower() for c in df.columns}
    df = df.rename(lower_map)
    cols = set(df.columns)

    # 2) 동의어 매핑
    rename_map = {}
    for c in ("ts", "timestamp", "time", "date", "datetime"):
        if c in cols:
            rename_map[c] = "ts"
            break
    for tgt, alts in {
        "open":  ["open", "o"],
        "high":  ["high", "h", "hi"],
        "low":   ["low", "l", "lo"],
        "close": ["close", "c", "adj_close", "close_price"],
    }.items():
        for a in alts:
            if a in cols:
                rename_map[a] = tgt
                break
    for a in ("volume", "vol", "base_volume", "volume_usdt", "quote_volume"):
        if a in cols:
            rename_map[a] = "volume"
            break

    df = df.rename(rename_map)
    cols = set(df.columns)

    # 3) ts → pl.Datetime 정규화 (정수 에폭 ms/s 및 문자열 지원)
    if "ts" in cols:
        ts_s = df["ts"]
        dt = ts_s.dtype
        if _is_int_dtype(dt):
            mx = int(ts_s.max()) if ts_s.len() else 0
            if mx > 1_000_000_000_000:  # epoch ms
                df = df.with_columns(pl.from_epoch(pl.col("ts"), unit="ms").alias("ts"))
            elif mx > 1_000_000_000:    # epoch s
                df = df.with_columns(pl.from_epoch(pl.col("ts"), unit="s").alias("ts"))
            else:
                # 이미 초 단위보다 작은 값이거나 특수케이스면 그대로 둠
                pass
        elif dt == pl.Utf8:
            # 엄격하지 않은 문자열 → datetime 파싱
            df = df.with_columns(pl.col("ts").str.to_datetime(strict=False).alias("ts"))
        # Datetime/Date면 그대로 둠

    return df.sort("ts")

# ---------- utils ----------
def load_cfg(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def load_ohlcv(cfg: Dict[str, Any]) -> pl.DataFrame:
    data_cfg = cfg["data"]
    p1 = (_ROOT / data_cfg["path_primary"]).resolve()
    p2 = (_ROOT / data_cfg["path_fallback"]).resolve()

    df: Optional[pl.DataFrame] = None

    # 1) 1순위: partitioned 디렉터리(재귀)
    if p1.exists():
        df = _read_parquet_any(p1)

    # 2) 2순위: fallback 파일/디렉터리
    if df is None and p2.exists():
        df = _read_parquet_any(p2)

    if df is None or df.width == 0:
        raise FileNotFoundError(
            f"No readable parquet found. Checked:\n  - {p1}\n  - {p2}"
        )

    # 표준화(ts/ohlc/volume + ts 타입)
    df = _standardize_ohlcv(df)

    # 날짜 슬라이싱
    start = data_cfg.get("start") or None
    end = data_cfg.get("end") or None
    if start:
        df = df.filter(pl.col("ts") >= pl.lit(start))
    if end:
        df = df.filter(pl.col("ts") <= pl.lit(end))

    # 필수 컬럼 검사
    need = {"ts", "open", "high", "low", "close", "volume"}
    missing = need - set(df.columns)
    if missing:
        # 문제 상황 디버그용: 현재 컬럼 리스트를 같이 보여줌
        raise KeyError(
            f"Missing columns in OHLCV: {missing}\n"
            f"Existing columns: {list(df.columns)[:30]}"
        )

    return df.select(["ts", "open", "high", "low", "close", "volume"] + 
                     [c for c in df.columns if c not in ("ts","open","high","low","close","volume")])

def quantile(series: pl.Series, pct: float) -> float:
    return float(series.quantile(pct))

def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))

def save_csv(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(str(path))

def compute_thresholds(scores: pl.Series, cfg: Dict[str, Any]) -> Tuple[float, float, Dict[str, Any]]:
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
    """Zero out `bars` bars after every True in mask."""
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

    tp_R = risk.get("atr_tp", [1.2, 1.3, 1.5])
    sl_R = float(risk.get("atr_sl", 1.0))
    max_hold_bars = int(risk.get("max_hold_min", 720) // 15)  # TF=15m

    rows = []
    i = 0
    n = len(df)
    while i < n:
        if df["enter_mask"][i]:
            entry = float(close[i])
            atr_i = float(atr[i])
            tp_levels = [entry + k * atr_i for k in tp_R]
            sl_level = entry - sl_R * atr_i

            j = i + 1
            rr = 0.0
            reason = "timeout"
            while j < n and (j - i) <= max_hold_bars:
                hi = float(df["high"][j])
                lo = float(df["low"][j])
                if lo <= sl_level:
                    rr = -sl_R
                    reason = "sl"
                    break
                hit = None
                for k, tp in enumerate(tp_levels, start=1):
                    if hi >= tp:
                        hit = k
                        break
                if hit is not None:
                    rr = float(tp_R[hit - 1] / sl_R)
                    reason = f"tp{hit}"
                    break
                j += 1
            rows.append((df["ts"][i], entry, df["ts"][j - 1] if j > i else df["ts"][i],
                         float(close[j - 1]) if j > i else entry, reason, j - i, rr))
            i = j
        else:
            i += 1

    trades = pl.DataFrame(rows, schema=["entry_ts", "entry", "exit_ts", "exit", "reason", "bars", "rr"]) \
        if rows else pl.DataFrame(schema=["entry_ts", "entry", "exit_ts", "exit", "reason", "bars", "rr"])

    if len(trades) == 0:
        result = {"winrate": 0.0, "pf": 0.0, "expR": 0.0, "mdd_R": 0.0, "avg_hold_bars": 0}
        return trades, result

    wins = trades.filter(pl.col("rr") > 0)
    losses = trades.filter(pl.col("rr") < 0)
    gross_win = float(wins["rr"].sum()) if wins.height else 0.0
    gross_loss = -float(losses["rr"].sum()) if losses.height else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 1e-12 else float("inf")
    winrate = float((trades["rr"] > 0).mean())
    expR = float(trades["rr"].mean())
    avg_hold = int(float(trades["bars"].mean()))

    eq = trades["rr"].cumsum()
    peak = eq.cum_max()
    dd = peak - eq
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

    # thresholds
    thr_c, thr_e, thr_info = compute_thresholds(df["score"], cfg)

    # masks
    g = cfg["gates"]
    dbg = cfg.get("debug", {})
    base_gate = df["gate_ok"] | pl.Series([dbg.get("no_gate", False)] * len(df))
    cand_mask = (df["score"] >= thr_c)
    enter_mask = (df["score"] >= thr_e)
    if g.get("cooloff_bars", 0) > 0:
        cand_mask = cooloff_mask(cand_mask, int(g["cooloff_bars"]))

    if not dbg.get("no_gate", False):
        cand_mask = cand_mask & base_gate
        enter_mask = enter_mask & base_gate

    df = df.with_columns([
        pl.Series("cand_mask", cand_mask),
        pl.Series("enter_mask", enter_mask),
    ])

    # reporting
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
