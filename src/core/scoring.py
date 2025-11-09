# src/core/scoring.py
from __future__ import annotations
import polars as pl

_EPS = 1e-12

def score_and_gate(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    """
    v9.4 완전 호환 스코어링 + 게이트 (버그 수정 v2)
    - 5-factor weighted sum
    - 방어 코드 추가 (Expression → Series 변환 수정)
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
    participation_n = pl.col("participation_n").clip(-1, 1)
    location_n = pl.col("location_n").clip(-1, 1)

    # Volatility 스코어 (낮을수록 좋음)
    vol_ref = float(df["atr_p"].quantile(0.90)) if df.height else 1.0
    vol_sc  = (1.0 - (pl.col("atr_p") / (vol_ref + _EPS))).clip(-1.0, 1.0)

    # 5-factor 가중합
    w_tr = float(w.get("trend",         0.25))
    w_mo = float(w.get("momentum",      0.25))
    w_vo = float(w.get("volatility",    0.15))
    w_pa = float(w.get("participation", 0.20))
    w_lo = float(w.get("location",      0.15))
    w_sum = (w_tr + w_mo + w_vo + w_pa + w_lo) or 1.0

    score = (
        w_tr * slope_n +
        w_mo * rsi_n +
        w_vo * vol_sc +
        w_pa * participation_n +
        w_lo * location_n
    ) / w_sum

    # ---------- 게이트 ----------
    conds = []

    # ADX
    if g.get("use_adx_gate", True):
        adx_min = float(g.get("adx_min", 25.0))
        adx_thr = (adx_min - 25.0) / 25.0
        conds.append(pl.col("adx_n") >= pl.lit(adx_thr))

    # 동적 slope
    if g.get("use_trend_gate", True):
        slope_min_base = float(g.get("ema_slope_min", 0.06))
        slope_min_chop = float(g.get("ema_slope_min_chop", 0.10))
        slope_thr = pl.when(adx_n < 0.0).then(pl.lit(slope_min_chop)).otherwise(pl.lit(slope_min_base))
        conds.append(pl.col("ema21_slope_n") >= slope_thr)

    # Range 게이트 (히스테리시스 제거)
    if g.get("use_range_gate", False):
        min_range_atr = float(g.get("min_range_atr", 0.60))
        atr_abs_col = pl.col(f"atr{atr_len}_abs")
        range_ok = (pl.col("high") - pl.col("low")) / (atr_abs_col + _EPS) >= pl.lit(min_range_atr)
        conds.append(range_ok)

    # EMA Bias 정량적
    if g.get("use_ema_bias_gate", True):
        ema_bias_norm = float(ind.get("ema_bias_norm", 0.010))
        df = df.with_columns(
            ((pl.col("close") - pl.col(f"ema{ema_fast}")) / (pl.col(f"ema{ema_fast}") + _EPS)).alias("ema_bias")
        )
        conds.append(pl.col("ema_bias") >= pl.lit(ema_bias_norm))
    
    # 가격-EMA 정렬 (기존)
    if g.get("align_price_ema", False):
        conds.append(pl.col("close") >= pl.col(f"ema{ema_fast}"))

    # 과열 편차 가드
    if g.get("use_dev_guard", True):
        atr_abs_col = pl.col(f"atr{atr_len}_abs")
        df = df.with_columns(
            ((pl.col("close") - pl.col(f"ema{ema_fast}")).abs() / (atr_abs_col + _EPS)).alias("dev_atr")
        )
        max_dev = float(g.get("max_dev_atr", 0.60))
        conds.append(pl.col("dev_atr") <= pl.lit(max_dev))
    
    # 변동성 필터
    if g.get("use_vol_filter", False):
        max_atr_p = float(g.get("max_atr_p", 2.5))
        conds.append(pl.col("atr_p") <= pl.lit(max_atr_p))

    # ✅ 1. gate_ok Expression 생성
    gate_ok = pl.all_horizontal(conds) if conds else pl.lit(True)

    # ✅ 2. DataFrame에 먼저 추가
    result_df = df.with_columns([
        score.clip(-1.0, 1.0).alias("score"),
        gate_ok.alias("gate_ok"),
    ])

    # ✅ 3. 방어 코드: result_df에서 Series 추출 후 계산
    if conds:
        gate_ok_series = result_df["gate_ok"]  # Series 추출
        gate_ok_count = int(gate_ok_series.sum())
        gate_ok_rate = float(gate_ok_series.mean())
        
        if gate_ok_rate < 0.01:  # 1% 미만
            print(f"\n[WARN] Gate pass rate too low: {gate_ok_rate:.2%} ({gate_ok_count} / {len(result_df)} rows)")
            active_gates = [k for k in g.keys() if k.startswith('use_') and g.get(k)]
            print(f"[WARN] Active gates: {active_gates}")
            print(f"[WARN] Consider:")
            print(f"  1. Relax thresholds:")
            print(f"     - adx_min: {g.get('adx_min', 25)} → 18")
            print(f"     - ema_slope_min: {g.get('ema_slope_min', 0.06)} → 0.04")
            print(f"     - max_dev_atr: {g.get('max_dev_atr', 0.60)} → 0.80")
            print(f"  2. Disable strict gates:")
            print(f"     - use_dev_guard: false")
            print(f"     - use_ema_bias_gate: false")
            print(f"  3. Debug mode: debug.no_gate: true\n")

    # ✅ 4. 반환
    return result_df