# scripts/data/download_binance_vision.py
from __future__ import annotations
import io, sys
from pathlib import Path
from datetime import date
import requests, zipfile
import polars as pl
from tqdm import tqdm

# ---- 설정 ----
MARKET = "futures/um"  # "spot" 또는 "futures/um"
SYMBOL = "BTCUSDT"
TF     = "15m"
START  = (2023, 1)     # (YYYY, MM)부터
END    = None          # None이면 오늘 월까지

BASE_URL = "https://data.binance.vision/data"

def month_iter(start_y, start_m, end_y, end_m):
    y, m = start_y, start_m
    while (y < end_y) or (y == end_y and m <= end_m):
        yield y, m
        m += 1
        if m == 13:
            y += 1
            m = 1

def build_url(market: str, symbol: str, tf: str, y: int, m: int) -> str:
    # 예) futures/um/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2025-11.zip
    yymm = f"{y:04d}-{m:02d}"
    return f"{BASE_URL}/{market}/monthly/klines/{symbol}/{tf}/{symbol}-{tf}-{yymm}.zip"

def download_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise FileNotFoundError(f"not found: {url} ({r.status_code})")
    return r.content

def csv_to_parquet_from_zip(zb: bytes, out_path: Path):
    with zipfile.ZipFile(io.BytesIO(zb)) as zf:
        # zip 안에 CSV 하나만 존재. 이름 탐색
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise ValueError("zip has no csv")
        with zf.open(names[0]) as f:
            # binance csv 스키마(열 순서 고정)
            # open_time, open, high, low, close, volume, close_time, quote_vol, trades, taker_buy_base, taker_buy_quote, ignore
            df = pl.read_csv(f, has_header=False)
            df = df.rename({
                "column_1":"open_time","column_2":"open","column_3":"high","column_4":"low",
                "column_5":"close","column_6":"volume","column_7":"close_time"
            })
            df = df.select([
                pl.col("open_time").cast(pl.Int64).alias("ts_ms"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Float64),
            ])
            df = df.with_columns([
                pl.from_epoch(pl.col("ts_ms"), unit="ms").alias("ts")
            ]).drop("ts_ms")
            # 정렬/중복 제거
            df = df.sort("ts").unique(subset=["ts"], keep="last")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(out_path)

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    part_root = root / "data" / "partitioned" / f"{SYMBOL}_{TF}"

    today = date.today()
    if END is None:
        end_y, end_m = today.year, today.month
    else:
        end_y, end_m = END
    start_y, start_m = START

    print(f"[INFO] download {SYMBOL} {TF} {MARKET} {start_y:04d}-{start_m:02d}..{end_y:04d}-{end_m:02d}")

    errs = 0
    for y, m in tqdm(list(month_iter(start_y, start_m, end_y, end_m))):
        url = build_url(MARKET, SYMBOL, TF, y, m)
        out = part_root / f"{y:04d}-{m:02d}" / "part.parquet"
        try:
            if out.exists():
                continue
            zb = download_bytes(url)
            csv_to_parquet_from_zip(zb, out)
        except Exception as e:
            errs += 1
            print(f"[WARN] skip {y}-{m:02d}: {e}")

    print(f"[DONE] partitioned path: {part_root} (errs={errs})")
