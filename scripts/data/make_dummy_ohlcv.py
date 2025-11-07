# scripts/data/make_dummy_ohlcv.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import polars as pl
from datetime import datetime, timedelta

def make_ohlcv(n_minutes: int = 90*24*60, tf_min: int = 15, seed: int = 42) -> pl.DataFrame:
    """
    90일치(기본) 15분봉 더미 OHLCV 생성.
    - ts: Datetime[KST naive 가정]
    - open/high/low/close/volume 컬럼 포함
    """
    np.random.seed(seed)
    steps = n_minutes // tf_min

    # 시간축(KST 로컬 naive)
    end = datetime.now()
    start = end - timedelta(minutes=n_minutes)
    # polars는 start/end 키워드 사용
    ts = pl.datetime_range(start=start, end=end, interval=f"{tf_min}m", eager=True).head(steps)

    # 랜덤워크 가격
    price0 = 30_000.0
    ret = np.random.normal(loc=0.0001, scale=0.005, size=len(ts))
    close = price0 * (1 + ret).cumprod()

    # OHLC 구성(고/저는 스프레드로 생성)
    spread = np.abs(np.random.normal(0.0, 0.003, size=len(ts))) * close
    open_  = np.concatenate([[price0], close[:-1]])
    high   = np.maximum.reduce([open_, close, close + spread])
    low    = np.minimum.reduce([open_, close, close - spread])
    vol    = np.random.lognormal(mean=10.0, sigma=0.3, size=len(ts))

    df = pl.DataFrame({
        "ts": ts,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol
    })
    return df

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    out = root / "data" / "processed" / "BTCUSDT_15m.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df = make_ohlcv()
    df.write_parquet(out)
    print(f"[OK] dummy written: {out}  rows={df.height}")
