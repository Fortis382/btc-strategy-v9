# scripts/backtest/backtest_polars.py
"""
v9.4 Backtest Engine - Final Hardened Version
- EWM variance correction (Welford's algorithm)
- Cache fingerprint with data hash
- adaptive_ewm_z warmup handling fix
- indicators recalculation prevention option
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import polars as pl
import numpy as np
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).parents[2]))
from src.signals.indicators import add_indicators
from src.signals.factors import compute_factors_polars, score_weighted_sum


# ========== IO ==========
def _read_parquet_any(path: Path, cfg: Dict[str, Any]) -> pl.DataFrame:
    """File/folder/glob + DuckDB option"""
    s = str(path)
    pattern = s if any(c in s for c in "*?[") else (s if path.is_file() else f"{path}/*.parquet")
    
    if cfg.get("io", {}).get("duckdb", False):
        try:
            import duckdb
            return duckdb.sql(f"SELECT * FROM read_parquet('{pattern}')").pl()
        except Exception:
            pass
    
    if Path(pattern).is_file():
        return pl.read_parquet(pattern)
    return pl.scan_parquet(pattern).collect(streaming=True)


# ========== Utils ==========
def _require_cols(df: pl.DataFrame, cols: List[str]) -> None:
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise KeyError(f"Missing: {miss}")


def _weights_from_cfg(cfg: Dict[str, Any]) -> Dict[str, float]:
    base = {"trend": 0.25, "momentum": 0.25, "volatility": 0.15, 
            "participation": 0.20, "location": 0.15}
    w = cfg.get("weights") or {}
    base.update({k: float(v) for k, v in w.items()})
    return base


def _phi_to_z(phi: float) -> float:
    """Standard normal z approximation (linear interpolation)"""
    table = [
        (0.70, 0.5244), (0.72, 0.5832), (0.75, 0.6745), (0.78, 0.7722),
        (0.80, 0.8416), (0.85, 1.0364), (0.90, 1.2816), (0.95, 1.6449)
    ]
    if phi <= table[0][0]:
        return table[0][1]
    if phi >= table[-1][0]:
        return table[-1][1]
    
    for (p0, z0), (p1, z1) in zip(table, table[1:]):
        if p0 <= phi <= p1:
            t = (phi - p0) / (p1 - p0)
            return z0 + t * (z1 - z0)
    return 0.6745


def _ewm_mean_std_correct(x: np.ndarray, alpha: float, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    EWM mean/std (Welford's algorithm)
    
    Correct variance formula:
    V_t = (1-a) * V_{t-1} + a * (x_t - mu_{t-1})^2
    """
    m = np.empty_like(x)
    v = np.empty_like(x)
    mean = 0.0
    var = 0.0
    init = False
    
    for i, val in enumerate(x):
        if np.isnan(val):
            m[i] = np.nan
            v[i] = np.nan
            continue
        
        if not init:
            mean = val
            var = 0.0
            init = True
        else:
            delta = val - mean
            mean = mean + alpha * delta
            var = (1 - alpha) * (var + alpha * delta * delta)
        
        m[i] = mean
        v[i] = var
    
    std = np.sqrt(np.maximum(v, 0.0)) + eps
    return m, std


def _fingerprint_safe(df: pl.DataFrame, cfg: Dict, factor_cols: List[str], ts_col: str = "ts") -> str:
    """Safe cache key (data hash + config)"""
    # Data signature
    data_sig = {
        "cols": df.columns,
        "rows": df.height,
        "first_ts": int(df[ts_col][0]) if df.height > 0 and ts_col in df.columns else 0,
        "last_ts": int(df[ts_col][-1]) if df.height > 0 and ts_col in df.columns else 0,
    }
    
    # Config signature
    cfg_sig = {
        "warmup": cfg.get("warmup_bars", 300),
        "weights": cfg.get("weights", {}),
        "factors": sorted(factor_cols),
    }
    
    combined = json.dumps({**data_sig, **cfg_sig}, sort_keys=True)
    return hashlib.sha256(combined.encode()).hexdigest()[:16]


# ========== Engine ==========
class BacktestEngine:
    """Polars vectorized backtest (fully hardened)"""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.warmup = int(cfg.get("warmup_bars", 300))
        self.hold = int(cfg.get("max_hold_bars", 60))
        self.side = str(cfg.get("side", "both"))
        
        gate = cfg.get("gate", {}) or {}
        self.gate_mode = str(gate.get("mode", "fixed"))
        self.fixed_thr = float(gate.get("fixed_threshold", 0.02))
        self.phi = float(gate.get("phi", 0.75))
        self.ewm_alpha = float(gate.get("ewm_alpha", 0.05))
        self.min_sigma = float(gate.get("min_sigma", 1e-6))
        
        risk = cfg.get("risk", {}) or {}
        self.risk_pct = float(risk.get("max_risk_per_trade", 0.02))
        
        cols = cfg.get("cols", {}) or {}
        self.col_ts = cols.get("ts", "ts")
        self.col_close = cols.get("close", "close")
        self.col_atr = cols.get("atr", "atr")
        
        self.weights = _weights_from_cfg(cfg)
        
        cache = cfg.get("cache", {}) or {}
        self.cache_enabled = bool(cache.get("enabled", False))
        self.cache_dir = Path(cache.get("dir", "cache"))
        if self.cache_enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self, df: pl.DataFrame, skip_indicators: bool = False) -> Dict[str, Any]:
        """Full backtest
        
        Args:
            skip_indicators: True to skip add_indicators (for CPCV)
        """
        df = df.sort(self.col_ts).with_row_count("_idx")
        
        # 1. Indicators
        if not skip_indicators:
            df = add_indicators(df, self.cfg)
        _require_cols(df, [self.col_close, self.col_atr])
        
        # 2. Factors (cache)
        df = self._attach_factors(df)
        
        # 3. Score
        df = self._add_score(df)
        
        # 4. Warmup null
        df = df.with_columns(
            pl.when(pl.col("_idx") < self.warmup)
            .then(None)
            .otherwise(pl.col("score"))
            .alias("score")
        )
        
        # 5. Gate
        df = self._gate(df)
        
        # 6. Trades
        df = self._simulate(df)
        
        # 7. Metrics
        metrics = self._metrics(df)
        trades = df.filter(pl.col("R").is_not_null()) \
                   .select([c for c in [self.col_ts, "score", "R"] if c in df.columns]) \
                   .to_dicts()
        
        return {"df": df, "metrics": metrics, "trades": trades}
    
    def _attach_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """Factor computation (cache support)"""
        factor_cols = ["trend", "momentum", "volatility", "participation", "location"]
        fp = _fingerprint_safe(df, self.cfg, factor_cols, self.col_ts)
        cache_path = self.cache_dir / f"factors_{fp}.parquet"
        
        if self.cache_enabled and cache_path.exists():
            fdf = pl.read_parquet(cache_path)
            if self.col_ts in fdf.columns:
                return df.join(fdf, on=self.col_ts, how="left")
        
        # compute
        df2 = compute_factors_polars(df, self.cfg)
        for f in factor_cols:
            if f not in df2.columns:
                df2 = df2.with_columns(pl.lit(0.0).alias(f))
        
        if self.cache_enabled:
            to_save = df2.select([self.col_ts] + factor_cols)
            to_save.write_parquet(cache_path)
        
        return df2
    
    def _add_score(self, df: pl.DataFrame) -> pl.DataFrame:
        """Score computation"""
        s = score_weighted_sum(df, self.weights)
        if isinstance(s, pl.Expr):
            return df.with_columns(s.alias("score"))
        if hasattr(s, "alias"):
            return df.with_columns(s.alias("score"))
        
        # fallback
        parts = [pl.col(k) * float(v) for k, v in self.weights.items() 
                 if k in df.columns and v != 0]
        return df.with_columns(pl.sum_horizontal(parts).alias("score"))
    
    def _gate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Gate application (warmup handling fix)"""
        if self.gate_mode == "fixed":
            long_ok = pl.col("score") > self.fixed_thr
            short_ok = pl.col("score") < -self.fixed_thr
        
        elif self.gate_mode == "adaptive_ewm_z":
            # Exclude warmup for EWM calculation
            valid_mask_np = (df["_idx"] >= self.warmup).to_numpy()
            score_all = df["score"].to_numpy()
            score_valid = score_all[valid_mask_np]
            
            mu, sig = _ewm_mean_std_correct(score_valid, self.ewm_alpha, self.min_sigma)
            z = (score_valid - mu) / np.maximum(sig, self.min_sigma)
            z_thr = _phi_to_z(self.phi)
            
            # Restore to full length
            long_ok_full = np.zeros(df.height, dtype=bool)
            short_ok_full = np.zeros(df.height, dtype=bool)
            
            long_ok_full[valid_mask_np] = (z > +z_thr)
            short_ok_full[valid_mask_np] = (z < -z_thr)
            
            df = df.with_columns([
                pl.Series("long_ok", long_ok_full).cast(pl.Boolean),
                pl.Series("short_ok", short_ok_full).cast(pl.Boolean)
            ])
            
            long_ok = pl.col("long_ok")
            short_ok = pl.col("short_ok")
        
        else:
            raise ValueError(f"Unknown gate.mode: {self.gate_mode}")
        
        # side handling
        if self.side == "long":
            return df.with_columns([
                long_ok.alias("entry"),
                pl.lit(True).alias("is_long")
            ])
        elif self.side == "short":
            return df.with_columns([
                short_ok.alias("entry"),
                pl.lit(False).alias("is_long")
            ])
        else:
            return df.with_columns([
                (long_ok | short_ok).alias("entry"),
                long_ok.alias("is_long")
            ])
    
    def _simulate(self, df: pl.DataFrame) -> pl.DataFrame:
        """Trade simulation (ATR-R)"""
        c, a = self.col_close, self.col_atr
        df = df.with_columns([
            pl.col(c).shift(-self.hold).alias("exit_close"),
            pl.col(a).alias("atr_entry")
        ])
        
        denom = (pl.col("atr_entry") * self.risk_pct) \
                .fill_null(1e-9) \
                .clip(lower_bound=1e-9)
        
        r_expr = (
            pl.when(pl.col("entry") & pl.col("is_long"))
            .then((pl.col("exit_close") - pl.col(c)) / denom)
            .when(pl.col("entry") & (~pl.col("is_long")))
            .then((pl.col(c) - pl.col("exit_close")) / denom)
            .otherwise(None)
        )
        
        return df.with_columns([
            r_expr.alias("R")
        ]).with_columns([
            pl.col("R").fill_nan(None).clip(-100.0, 100.0)
        ])
    
    def _metrics(self, df: pl.DataFrame) -> Dict[str, float]:
        """Metrics (correct Sharpe)"""
        trades = df.filter(pl.col("R").is_not_null())
        n = int(trades.height)
        
        # cand_ratio (exclude warmup)
        live = df.filter(pl.col("_idx") >= self.warmup)
        total_bars = int(live.height)
        cand_count = int(live["entry"].sum()) if "entry" in live.columns else 0
        cand_ratio = cand_count / total_bars if total_bars > 0 else 0.0
        
        if n == 0:
            return {
                "total_trades": 0, "winrate": 0.0, "avg_R": 0.0, "std_R": 0.0,
                "profit_factor": 0.0, "expectancy_R": 0.0, "max_dd_R": 0.0,
                "sharpe": 0.0, "avg_hold_bars": float(self.hold),
                "cand_ratio": cand_ratio, "avg_gap_bars": float(self.hold)
            }
        
        R = trades["R"].to_numpy()
        wins = (R > 0).sum()
        losses = (R < 0).sum()
        
        win_R = float(R[R > 0].sum()) if wins else 0.0
        loss_R = abs(float(R[R < 0].sum())) if losses else 0.0
        pf = (win_R / loss_R) if loss_R > 0 else 999.0
        
        # DD
        cum = np.cumsum(R)
        dd = cum - np.maximum.accumulate(cum)
        max_dd_R = abs(float(dd.min())) if dd.size else 0.0
        
        # Sharpe (inter-trade gap)
        avg_R = float(R.mean())
        std_R = float(R.std()) if R.size > 1 else 1e-9
        
        trade_idx = trades["_idx"].to_numpy()
        if n > 1:
            gaps = np.diff(trade_idx)
            avg_gap = float(gaps.mean())
        else:
            avg_gap = float(self.hold)
        
        bars_per_year = 252 * 96
        trades_per_year = bars_per_year / max(avg_gap, 1.0)
        sharpe = (avg_R / std_R) * np.sqrt(trades_per_year)
        
        return {
            "total_trades": n,
            "winrate": float(wins / n),
            "avg_R": avg_R,
            "std_R": std_R,
            "profit_factor": pf,
            "expectancy_R": avg_R,
            "max_dd_R": max_dd_R,
            "sharpe": float(sharpe),
            "avg_hold_bars": float(self.hold),
            "cand_ratio": cand_ratio,
            "avg_gap_bars": avg_gap
        }


def run_backtest(data_path: Path, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Public API"""
    df = _read_parquet_any(data_path, cfg)
    return BacktestEngine(cfg).run(df)


if __name__ == "__main__":
    import yaml
    cfg = yaml.safe_load(Path("config/settings_v9.yaml").read_text("utf-8"))
    result = run_backtest(Path(cfg["data"]["path"]), cfg)
    m = result["metrics"]
    print(f"[ENGINE] n={m['total_trades']} wr={m['winrate']:.2%} "
          f"pf={m['profit_factor']:.2f} exp={m['expectancy_R']:.3f}R "
          f"sharpe={m['sharpe']:.2f} cand={m['cand_ratio']:.2%} "
          f"gap={m['avg_gap_bars']:.1f}")