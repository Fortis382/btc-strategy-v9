# src/core/loader_polars.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import polars as pl
from datetime import datetime
import duckdb

MIN_PARQUET_BYTES = 12  # parquet 최소 헤더/푸터 크기

def _is_valid_parquet(p: Path) -> bool:
    try:
        if not p.is_file() or p.stat().st_size < MIN_PARQUET_BYTES:
            return False
        # 파일별 1행만 검사(느려지지 않도록)
        pl.scan_parquet(str(p)).limit(1).collect()
        return True
    except Exception:
        return False

def _collect_valid_files(root: Path) -> List[str]:
    files = list(root.rglob("*.parquet"))
    valids = [str(f) for f in files if _is_valid_parquet(f)]
    return valids

def _read_any(p: Path) -> pl.DataFrame:
    """DuckDB로 partitioned Parquet 직접 로드"""
    if p.is_dir():
        # ✅ Windows 안전 경로 (forward slash + glob 명시)
        pattern = str(p).replace("\\", "/") + "/**/*.parquet"
        
        con = duckdb.connect(":memory:")
        try:
            result = con.execute(
                f"SELECT * FROM read_parquet('{pattern}') ORDER BY ts"
            ).arrow()
            con.close()
            return pl.from_arrow(result)
        except Exception as e:
            con.close()
            # DuckDB 실패 시 Polars 직접 로드로 폴백
            files = _collect_valid_files(p)
            if not files:
                raise FileNotFoundError(f"No valid parquet in {p}")
            return pl.read_parquet(files)
    
    # 단일 파일
    if not _is_valid_parquet(p):
        raise FileNotFoundError(f"Invalid parquet: {p}")
    return pl.read_parquet(str(p))

def _parse_iso_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s)

def _normalize_ts(df: pl.DataFrame) -> pl.DataFrame:
    ts_dtype = df["ts"].dtype
    if ts_dtype == pl.Datetime:
        return df
    if ts_dtype in (pl.Int64, pl.Int32, pl.UInt64, pl.UInt32):
        vmax = int(df["ts"].max()) if df.height > 0 else 0
        unit = "s"
        if vmax > 10_000_000_000: unit = "ms"
        if vmax > 10_000_000_000_000: unit = "us"
        if vmax > 10_000_000_000_000_000: unit = "ns"
        return df.with_columns(pl.from_epoch(pl.col("ts").cast(pl.Int64), unit=unit).alias("ts"))
    if ts_dtype == pl.Utf8:
        # ✅ 수정: strptime → str.to_datetime (Polars 0.20+)
        return df.with_columns(
            pl.col("ts").str.to_datetime(strict=False).alias("ts")
        )
    return df.with_columns(pl.col("ts").cast(pl.Datetime))

def load_ohlcv(project_root: Path,
               path_primary: str,
               path_fallback: str,
               start: Optional[str],
               end: Optional[str]) -> pl.DataFrame:
    p1, p2 = project_root / path_primary, project_root / path_fallback

    df = None
    # 1) partitioned 우선
    if p1.exists() and p1.is_dir():
        try:
            df = _read_any(p1)
        except FileNotFoundError:
            df = None  # 비어있거나 전부 불량 → fallback 시도
    # 2) fallback 단일
    if df is None and p2.exists():
        df = _read_any(p2)

    if df is None:
        raise FileNotFoundError(
            f"데이터 없음 또는 손상: {p1} (dir) / {p2} (file). "
            f"partitioned에 유효 파켓이 없고 fallback 파일도 유효하지 않음."
        )

    df = _normalize_ts(df).sort("ts").unique(subset=["ts"], keep="last")

    s_dt, e_dt = _parse_iso_dt(start), _parse_iso_dt(end)
    if s_dt: df = df.filter(pl.col("ts") >= pl.lit(s_dt))
    if e_dt: df = df.filter(pl.col("ts") <= pl.lit(e_dt))

    needed = {"ts","open","high","low","close","volume"}
    have = set(df.columns)
    missing = needed - have
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing} (있는 컬럼: {sorted(have)})")

    return df
