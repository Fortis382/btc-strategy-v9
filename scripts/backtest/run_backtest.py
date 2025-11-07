# scripts/backtest/run_backtest.py
from __future__ import annotations

# ---- robust path bootstrap (must be first) ----
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
# repo root: .../btc-v9
PROJECT_ROOT = FILE.parents[2]

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    # put src at the very front — ensure imports work no matter cwd
    sys.path.insert(0, str(SRC_PATH))

# (optional) also make PROJECT_ROOT importable (ex: config helpers later)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
# -----------------------------------------------

import argparse, json, math
import polars as pl
import yaml

from src.core.loader_polars import load_ohlcv
from src.signals.indicators import add_indicators
from src.core.scoring import score_and_gate

def load_cfg(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _q(df: pl.DataFrame, col: str, p: float) -> float:
    return float(df.select(pl.col(col).quantile(p)).to_series().item())

def _stats(df: pl.DataFrame) -> dict:
    if df.height == 0:
        return {}
    q = df.select([
        pl.col("score").quantile(0.25).alias("score_q25"),
        pl.col("score").quantile(0.50).alias("score_q50"),
        pl.col("score").quantile(0.75).alias("score_q75"),
        pl.col("score").quantile(0.90).alias("score_q90"),
        pl.col("ema21_slope_n").mean().alias("ema21_slope_n_mean"),
        pl.col("adx_n").mean().alias("adx_n_mean"),
    ]).to_dicts()[0]
    total = df.height
    gate_ok = df.filter(pl.col("gate_ok")).height
    return {**{k: (float(v) if v is not None else None) for k, v in q.items()},
            "rows": int(total),
            "gate_ok_rows": int(gate_ok),
            "gate_ok_rate": float(gate_ok/total) if total else 0.0}

def run(cfg_path: Path, quiet: bool = False):
    def log(*a, **k):
        if not quiet:
            print(*a, **k)

    cfg = load_cfg(cfg_path)
    debug = cfg.get("debug", {})

    # 1) 데이터 + 피쳐
    df = load_ohlcv(PROJECT_ROOT,
                    cfg["data"]["path_primary"],
                    cfg["data"]["path_fallback"],
                    cfg["data"]["start"], cfg["data"]["end"])
    df = add_indicators(df, cfg)
    df = score_and_gate(df, cfg)

    # 2) 임계(자동/수동)
    thr_c = float(cfg["thresholds"]["cand_score_min"])
    thr_e = float(cfg["thresholds"]["enter_score_min"])
    thr_mode = "manual"
    thr_note = {}

    auto = debug.get("auto_thresholds", None)
    if auto:
        cand_pct  = float(auto.get("cand_pct", 0.70))   # cand = 상위 30% 기준치
        enter_pct = float(auto.get("enter_pct", 0.85))  # enter = 상위 15% 기준치
        thr_c = _q(df, "score", cand_pct)
        thr_e = _q(df, "score", enter_pct)
        thr_mode = "auto_pct"
        thr_note = {"cand_pct": cand_pct, "enter_pct": enter_pct,
                    "thr_c_auto": thr_c, "thr_e_auto": thr_e}

    cool  = cfg["gates"]["cooloff_bars"]
    atr_col = f"atr{cfg['indicators']['atr']}"

    # 3) cand 생성 로직
    gate_used = "normal"
    gate_note = {}

    if bool(debug.get("no_gate", False)):
        df = df.with_columns([(pl.col("score") >= thr_c).alias("is_candidate")])
        gate_used = "no_gate"
    elif bool(debug.get("force_cand_from_score", False)):
        pct = float(debug.get("force_pct", 0.75))
        qthr = _q(df, "score", pct)
        df = df.with_columns([(pl.col("score") >= qthr).alias("is_candidate")])
        gate_used = "forced_pct"
        gate_note = {"force_pct": pct, "score_qthr": float(qthr)}
    else:
        df = df.with_columns([((pl.col("score") >= thr_c) & pl.col("gate_ok")).alias("is_candidate")])

    # 4) 엔진(스모크)
    trades = []
    in_pos = False
    entry_i = None
    entry_price = None
    cool_left = 0

    for i in range(1, len(df)):
        row = df.row(i, named=True)

        if in_pos:
            atr = row[atr_col]
            if atr is None or math.isnan(atr):
                continue
            base = entry_price
            tp1 = base + cfg["risk"]["atr_tp"][0]*atr
            tp2 = base + cfg["risk"]["atr_tp"][1]*atr
            tp3 = base + cfg["risk"]["atr_tp"][2]*atr
            sl  = base - cfg["risk"]["atr_sl"]*atr

            hit_tp1 = row["high"] >= tp1
            hit_tp2 = row["high"] >= tp2
            hit_tp3 = row["high"] >= tp3
            hit_sl  = row["low"]  <= sl

            exit_reason = None
            exit_price  = None
            if hit_sl:
                exit_reason, exit_price = "SL", sl
            else:
                if   hit_tp3: exit_reason, exit_price = "TP3", tp3
                elif hit_tp2: exit_reason, exit_price = "TP2", tp2
                elif hit_tp1: exit_reason, exit_price = "TP1", tp1

            if exit_reason is None:
                hold = i - entry_i
                max_hold = max(1, int(cfg["risk"]["max_hold_min"] // 15))
                if hold >= max_hold:
                    exit_reason, exit_price = "TIME", row["close"]

            if exit_reason:
                rr = (exit_price - entry_price) / (cfg["risk"]["atr_sl"]*atr + 1e-12)
                trades.append({
                    "entry_ts": df["ts"][entry_i], "entry": float(entry_price),
                    "exit_ts": row["ts"],          "exit":  float(exit_price),
                    "reason":  exit_reason,        "bars":  i - entry_i,
                    "rr":      float(rr),
                })
                in_pos = False
                entry_i = None
                cool_left = cool
                continue

        if not in_pos and cool_left == 0:
            if row["is_candidate"] and row["score"] >= thr_e:
                entry_price = row["close"]; entry_i = i; in_pos = True

        if cool_left > 0:
            cool_left -= 1

    # 5) 결과/로그
    trades_df = (pl.from_dicts(trades) if trades else
                 pl.DataFrame(schema={"entry_ts":pl.Datetime,"entry":pl.Float64,"exit_ts":pl.Datetime,
                                      "exit":pl.Float64,"reason":pl.Utf8,"bars":pl.Int64,"rr":pl.Float64}))

    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    trades_df.write_csv(PROJECT_ROOT / cfg["output"]["trades_csv"])

    if len(trades_df) > 0:
        win = (trades_df["rr"] > 0).sum()
        winrate = win / len(trades_df)

        pos_sum = trades_df.filter(pl.col("rr") > 0)["rr"].sum()
        neg_sum = trades_df.filter(pl.col("rr") <= 0)["rr"].sum()
        pf  = (pos_sum / abs(neg_sum)) if neg_sum != 0 else 0.0
        exp = trades_df["rr"].mean()

        # ★ MDD 계산: cum_sum / cum_max 사용
        eq = trades_df["rr"].cum_sum()
        dd = eq - eq.cum_max()
        mdd_abs = float(abs(dd.min()))
        avg_hold = int(trades_df["bars"].mean())
    else:
        winrate = pf = exp = 0.0
        mdd_abs = 0.0
        avg_hold = 0

    breakdown = {}
    if bool(debug.get("log_breakdown", False)):
        breakdown = {
            "stats": _stats(df),
            "counts": {
                "total_rows": int(df.height),
                "score_ge_cand": int(df.filter(pl.col("score") >= thr_c).height),
                "score_ge_enter": int(df.filter(pl.col("score") >= thr_e).height),
                "gate_ok": int(df.filter(pl.col("gate_ok")).height),
                "cand_mask_true": int(df.filter(pl.col("is_candidate")).height),
            },
            "gate_used": "no_gate" if bool(debug.get("no_gate", False)) else "normal",
            "thr_mode": thr_mode,
            **thr_note
        }

    dbg = {
        "winrate": round(winrate, 4),
        "pf": round(pf, 3) if pf != 0 else 0,
        "expR": round(exp, 4),
        "mdd_R": round(mdd_abs, 3),
        "avg_hold_bars": avg_hold,
        "thr_used": {"cand": round(thr_c, 6), "enter": round(thr_e, 6)},
        "counts": {"trades": len(trades_df)},
        "breakdown": breakdown
    }
    (PROJECT_ROOT / cfg["output"]["dbg_json"]).write_text(
        json.dumps(dbg, ensure_ascii=False, indent=2), encoding="utf-8")

    log("[RESULT]", dbg)
    log(f"[SAVE] {PROJECT_ROOT / cfg['output']['trades_csv']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config/settings_v9.yaml"))
    ap.add_argument("--quiet", action="store_true", help="suppress console prints")
    args = ap.parse_args()
    run(Path(args.config), quiet=bool(args.quiet))
