# src/core/scoring.py
from __future__ import annotations
import polars as pl

_EPS = 1e-12

def score_and_gate(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """
    v9.4 완전 호환 스코어링 + 게이트
    - 5-factor weighted sum
    - 동적 ADX/slope/range 게이트
    """
    w = cfg["scoring"]["weights"]
    g = cfg["gates"]
    ind = cfg["indicators"]

    atr_len = int(ind.get("atr", 14))
    ema_fast = int(ind.get("ema", [21, 55])[0])

    # Factor 정규화 값 로드
    slope_n = pl.col("ema21_slope_n").clip(-1, 1)
    rsi_n   = pl.col("rsi_n").clip(-1, 1)
    adx_n   = pl.col("adx_n").clip(-1, 1)
    participation_n = pl.col("participation_n").clip(-1, 1)  # ✅ 신규
    location_n = pl.col("location_n").clip(-1, 1)            # ✅ 신규

    # Volatility 스코어 (낮을수록 좋음)
    vol_ref = float(df["atr_p"].quantile(0.90)) if df.height else 1.0
    vol_sc  = (1.0 - (pl.col("atr_p") / (vol_ref + _EPS))).clip(-1.0, 1.0)

    # ✅ 5-factor 가중합
    w_tr = float(w.get("trend",         0.25))
    w_mo = float(w.get("momentum",      0.25))
    w_vo = float(w.get("volatility",    0.15))
    w_pa = float(w.get("participation", 0.20))  # ✅ 신규
    w_lo = float(w.get("location",      0.15))  # ✅ 신규
    w_sum = (w_tr + w_mo + w_vo + w_pa + w_lo) or 1.0

    score = (
        w_tr * slope_n +
        w_mo * rsi_n +
        w_vo * vol_sc +
        w_pa * participation_n +  # ✅ 신규
        w_lo * location_n         # ✅ 신규
    ) / w_sum

    # ---------- 게이트 ----------
    conds = []

    # ADX (✅ 수정된 정규화 기준)
    if g.get("use_adx_gate", True):
        adx_min = float(g.get("adx_min", 25.0))
        # adx_n = (ADX - 25) / 25 범위 [-1, 1]
        # ADX 20 → (-5/25) = -0.2
        # ADX 25 → 0
        # ADX 30 → 0.2
        adx_thr = (adx_min - 25.0) / 25.0
        conds.append(pl.col("adx_n") >= pl.lit(adx_thr))

    # 동적 slope (촙/트렌드 구분)
    if g.get("use_trend_gate", True):
        slope_min_base = float(g.get("ema_slope_min", 0.06))
        slope_min_chop = float(g.get("ema_slope_min_chop", 0.10))
        slope_thr = pl.when(adx_n < 0.0).then(pl.lit(slope_min_chop)).otherwise(pl.lit(slope_min_base))
        conds.append(pl.col("ema21_slope_n") >= slope_thr)

    # Range 히스테리시스
    if g.get("use_range_gate", True):
        min_range_atr = float(g.get("min_range_atr", 0.60))
        range_on_bars = int(g.get("range_on_bars", 3))
        atr_abs_col = pl.col(f"atr{atr_len}_abs")

        df = df.with_columns(
            ((pl.col("high") - pl.col("low")) / (atr_abs_col + _EPS)).alias("range_atr")
        )
        df = df.with_columns(
            (pl.col("range_atr") >= pl.lit(min_range_atr)).cast(pl.Int32).alias("range_flag")
        )
        df = df.with_columns(
            pl.col("range_flag").rolling_sum(window_size=max(1, range_on_bars))
              .fill_null(0)
              .ge(range_on_bars)
              .alias("range_on")
        )
        conds.append(pl.col("range_on"))

    # 가격–EMA 정렬 (롱 기준)
    if g.get("align_price_ema", True):
        conds.append(pl.col("close") >= pl.col(f"ema{ema_fast}"))

    # 과열乖리 가드
    if g.get("use_dev_guard", True):
        atr_abs_col = pl.col(f"atr{atr_len}_abs")
        df = df.with_columns(
            ((pl.col("close") - pl.col(f"ema{ema_fast}")).abs() / (atr_abs_col + _EPS)).alias("dev_atr")
        )
        max_dev = float(g.get("max_dev_atr", 0.60))
        conds.append(pl.col("dev_atr") <= pl.lit(max_dev))

    gate_ok = pl.all_horizontal(conds) if conds else pl.lit(True)

    return df.with_columns([
        score.clip(-1.0, 1.0).alias("score"),
        gate_ok.alias("gate_ok"),
    ])