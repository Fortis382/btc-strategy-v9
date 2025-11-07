# tests/test_scoring_min.py
import polars as pl
from src.core.scoring import score_and_gate

def _df():
    return pl.DataFrame({
        "ts":   pl.datetime_range(start=pl.datetime(2025,1,1,0,0), end=pl.datetime(2025,1,1,5,0), interval="15m", eager=True),
        "open": [100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120],
        "high": [x+2 for x in range(100,121)],
        "low":  [x-2 for x in range(100,121)],
        "close":[x+0.5 for x in range(100,121)],
        "ema21_slope_n":[0.06]*21, "rsi_n":[0.2]*21, "adx_n":[-0.4]*21, "atr_p":[0.03]*21, "atr14":[10]*21
    })

def test_no_type_error():
    df=score_and_gate(_df(), {
        "scoring":{"weights":{"trend":1,"momentum":1,"volatility":1,"regime":1}},
        "gates":{"use_adx_gate":True,"use_trend_gate":True,"use_range_gate":True,"adx_min":26,"ema_slope_min":0.05,"min_range_atr":0.58,"adx_strict_buffer":0.20},
        "indicators":{"atr":14}
    })
    assert "score" in df.columns and "gate_ok" in df.columns

def test_adaptive_kofn():
    # adx_n=-0.4이면 엄격모드(3-of-3) → gate_ok는 대부분 False
    df=score_and_gate(_df(), {
        "scoring":{"weights":{"trend":1,"momentum":1,"volatility":1,"regime":1}},
        "gates":{"use_adx_gate":True,"use_trend_gate":True,"use_range_gate":True,"adx_min":26,"ema_slope_min":0.05,"min_range_atr":0.58,"adx_strict_buffer":0.20},
        "indicators":{"atr":14}
    })
    assert df["gate_ok"].sum() <= 1
