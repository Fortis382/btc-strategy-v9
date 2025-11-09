# tests/test_scoring_min.py
import polars as pl
from src.core.scoring import score_and_gate

def _df():
    """5-factor 호환 더미 데이터"""
    return pl.DataFrame({
        "ts": pl.datetime_range(
            start=pl.datetime(2025,1,1,0,0), 
            end=pl.datetime(2025,1,1,5,0), 
            interval="15m", 
            eager=True
        ),
        "open": [100+i for i in range(21)],
        "high": [102+i for i in range(21)],
        "low": [98+i for i in range(21)],
        "close": [100.5+i for i in range(21)],
        "volume": [1000+i*10 for i in range(21)],  # ✅ 추가
        # 5-factor 정규화 값 (indicators.py 출력 가정)
        "ema21_slope_n": [0.06]*21,
        "rsi_n": [0.2]*21,
        "adx_n": [-0.4]*21,
        "atr_p": [0.03]*21,
        "atr14_abs": [10.0]*21,
        "participation_n": [0.1]*21,  # ✅ 신규
        "location_n": [0.5]*21,       # ✅ 신규
        "ema21": [100.0]*21,
    })

def test_no_type_error():
    """타입 에러 없이 실행되는지 확인"""
    df = score_and_gate(_df(), {
        "scoring": {
            "weights": {
                "trend": 0.25,
                "momentum": 0.25,
                "volatility": 0.15,
                "participation": 0.20,  # ✅ 신규
                "location": 0.15,       # ✅ 신규
            }
        },
        "gates": {
            "use_adx_gate": True,
            "adx_min": 20.0,
            "use_trend_gate": True,
            "ema_slope_min": 0.06,
            "use_range_gate": False,
            "use_ema_bias_gate": False,
            "use_dev_guard": False,
        },
        "indicators": {"atr": 14, "ema": [21, 55]}
    })
    assert "score" in df.columns
    assert "gate_ok" in df.columns

def test_score_range():
    """스코어가 [-1, 1] 범위인지 확인"""
    df = score_and_gate(_df(), {
        "scoring": {
            "weights": {
                "trend": 0.25, "momentum": 0.25, "volatility": 0.15,
                "participation": 0.20, "location": 0.15,
            }
        },
        "gates": {"use_adx_gate": False, "use_trend_gate": False},
        "indicators": {"atr": 14, "ema": [21, 55]}
    })
    assert df["score"].min() >= -1.0
    assert df["score"].max() <= 1.0