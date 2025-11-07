# scripts/backtest/run_backtest.py
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import polars as pl
import yaml, sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))

from core.loader_polars import load_ohlcv
from signals.indicators import add_indicators
from core.scoring import score_and_gate

def load_cfg(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def run(cfg_path: Path):
    cfg = load_cfg(cfg_path)

    df = load_ohlcv(PROJECT_ROOT,
                    cfg["data"]["path_primary"],
                    cfg["data"]["path_fallback"],
                    cfg["data"]["start"], cfg["data"]["end"])
    df = add_indicators(df, cfg)
    df = score_and_gate(df, cfg)

    thr_c = cfg["thresholds"]["cand_score_min"]
    thr_e = cfg["thresholds"]["enter_score_min"]
    cool  = cfg["gates"]["cooloff_bars"]
    atr_col = f"atr{cfg['indicators']['atr']}"

    df = df.with_columns([
        ((pl.col("score") >= thr_c) & pl.col("gate_ok")).alias("is_candidate")
    ])

    trades = []
    in_pos = False
    entry_i = None
    entry_price = None
    cool_left = 0

    for i in range(1, len(df)):
        row = df.row(i, named=True)

        if in_pos:
            atr = row[atr_col]
            if atr is None or math.isnan(atr):  # 보수 가드
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
        mdd = float((trades_df["rr"].cumsum().cummin().min()))
        avg_hold = int(trades_df["bars"].mean())
    else:
        winrate = pf = exp = 0.0; mdd = 0.0; avg_hold = 0

    dbg = {
        "winrate": round(winrate,4),
        "pf": round(pf,3) if pf!=0 else 0,
        "expR": round(exp,4),
        "mdd_R": round(mdd,3),
        "avg_hold_bars": avg_hold,
        "thr_used": {"cand":thr_c,"enter":thr_e},
        "counts": {"trades": len(trades_df)}
    }
    (PROJECT_ROOT / cfg["output"]["dbg_json"]).write_text(
        json.dumps(dbg, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[RESULT]", dbg)
    print(f"[SAVE] {PROJECT_ROOT / cfg['output']['trades_csv']}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(PROJECT_ROOT / "config/settings_v9.yaml"))
    args = ap.parse_args()
    run(Path(args.config))
