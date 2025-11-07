# src/core/scoring.py
from __future__ import annotations
import polars as pl

_EPS = 1e-12

def score_and_gate(df: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    w = cfg["scoring"]["weights"]
    g = cfg["gates"]
    ind = cfg["indicators"]

    # ---- 설정에서 길이/스케일 읽기 ----
    atr_len = int(ind.get("atr", 14))

    # ---- 계약 컬럼 직접 소비(이미 indicators.py에서 생성됨) ----
    # ema21_slope_n, rsi_n, adx_n, atr_p는 -1..+1 또는 % 스케일로 정리되어 있음
    slope_n = pl.col("ema21_slope_n").clip(-1, 1)
    rsi_n   = pl.col("rsi_n").clip(-1, 1)
    adx_n   = pl.col("adx_n").clip(-1, 1)
    atr_p   = pl.col("atr_p")  # % 단위(0..∞). 가중치에서만 쓰는 보조항

    # ---- 변동성 보정 점수(극단치 과대 가중 방지) ----
    # 상위 분위수 기준으로 - (과대 변동시 감점) 형태, [-1,1]로 클램프
    vol_ref = float(df["atr_p"].quantile(0.90)) if df.height else 1.0
    vol_sc  = (1.0 - (pl.col("atr_p") / (vol_ref + _EPS))).clip(-1.0, 1.0)

    # ---- 종합 스코어: 선형결합 → [-1,1]
    # weights: trend/momentum/volatility/regime
    w_tr = float(w.get("trend",      0.40))
    w_mo = float(w.get("momentum",   0.25))
    w_vo = float(w.get("volatility", 0.10))
    w_re = float(w.get("regime",     0.25))
    w_sum = (w_tr + w_mo + w_vo + w_re) or 1.0

    score = (
        w_tr * slope_n +
        w_mo * rsi_n   +
        w_vo * vol_sc  +
        w_re * adx_n
    ) / w_sum
    score = score.clip(-1.0, 1.0).alias("score")

    # ---- 게이트 구성 ----
    conds = []

    # 1) ADX 게이트(0..100 → -1..+1 맵은 indicators에서 이미 처리됨 → 여기선 adx_n 직접 비교)
    #    설정의 adx_min(0..100)은 내부적으로 adx_n 임계로 변환
    if g.get("use_adx_gate", True):
        adx_min = float(g.get("adx_min", 20.0))
        adx_thr = (adx_min / 50.0) - 1.0           # 0..100 → -1..+1
        conds.append(adx_n >= pl.lit(adx_thr))

    # 2) 동적 트렌드 게이트: 촙(ADX<0)일 때 slope 요구 상향
    if g.get("use_trend_gate", True):
        slope_min_base = float(g.get("ema_slope_min", 0.06))
        slope_min_chop = float(g.get("ema_slope_min_chop", 0.10))
        slope_thr = pl.when(adx_n < 0.0).then(pl.lit(slope_min_chop)).otherwise(pl.lit(slope_min_base))
        conds.append(pl.col("ema21_slope_n") >= slope_thr)

    # --- 레인지 히스테리시스: (high-low)/ATR >= θ가 연속 N바 이상일 때만 ON
    if g.get("use_range_gate", False):
        min_range_atr = float(g.get("min_range_atr", 0.60))
        range_on_bars = int(g.get("range_on_bars", 3))
        atr_abs_col = pl.col(f"atr{atr_len}_abs")   # 설정 길이 동기화

        # 1단계: range_atr 물리 컬럼 먼저 생성
        df = df.with_columns(
            ((pl.col("high") - pl.col("low")) / (atr_abs_col + _EPS)).alias("range_atr")
        )

        # 2단계: 임계 통과 플래그 (정수형으로 고정)
        df = df.with_columns(
            (pl.col("range_atr") >= pl.lit(min_range_atr)).cast(pl.Int32).alias("range_flag")
        )

        # 3단계: 연속성 히스테리시스 (rolling_sum ≥ N 이면 ON)
        df = df.with_columns(
            pl.col("range_flag")
              .rolling_sum(window_size=max(1, range_on_bars))
              .fill_null(0)
              .ge(range_on_bars)
              .alias("range_on")
        )

        conds.append(pl.col("range_on"))

    gate_ok = pl.all_horizontal(conds) if conds else pl.lit(True)

    return df.with_columns([
        score,
        gate_ok.alias("gate_ok"),
    ])
