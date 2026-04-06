#!/usr/bin/env python3
"""
STAT ARB ML BOT v1

What this bot does
------------------
1) Builds a universe of crypto pairs grouped by token sectors/themes.
2) Downloads OHLCV data from Binance via ccxt.
3) Computes stat-arb features per pair on rolling windows.
4) Generates entry signals when spread dislocates and pair quality is acceptable.
5) Applies an ML filter if enough historical labeled trades exist.
6) Simulates positions with demo money using hedge-ratio-weighted notionals.
7) Exits by system rules:
   - Take profit when |zscore| contracts below tp_z_abs
   - Stop loss when |zscore| exceeds sl_z_abs
   - Cointegration quality breaks
   - Max holding time exceeded
8) Logs everything to CSV for research and future model training.
9) Sends Telegram updates for entries, exits, and errors.
10) Auto-builds a training dataset from closed trades and retrains the model.

What to expect
--------------
- This is a research / demo execution engine, not broker execution code.
- It is designed to produce a clean dataset and a disciplined signal-to-trade loop.
- Early on, ML will be inactive until enough closed trades exist.
- The first goal is not profits. The first goal is clean, consistent data.
- Then you can measure whether filters, regimes, sectors, and pair types actually add edge.

Core idea
---------
Each trade is treated as an experiment. The bot records:
- pair features at entry
- simulated position sizing
- exit reason
- pnl / return / MFE / MAE / holding time
This creates a row-based dataset for self-improvement.

Dependencies
------------
pip install ccxt pandas numpy scikit-learn scipy requests joblib

Optional
--------
Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID to get alerts.

Notes
-----
- Uses Binance spot/perpetual-style market data naming conventions like BTC/USDT.
- Hedge sizing is based on normalized dollar weighting from hedge ratio.
- Spread is modeled using log prices and an OLS hedge ratio.
- Cointegration proxy uses ADF on spread. Strict Engle-Granger is approximated for speed.
- This script is intentionally verbose and heavily logged for transparency.
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import zscore
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.tsa.stattools import adfuller

try:
    import ccxt
except Exception as e:
    raise RuntimeError("ccxt is required. Install with: pip install ccxt") from e


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    "exchange_id": "binance",
    "quote_asset": "USDT",
    "timeframe": "1h",
    "history_limit": 600,
    "poll_seconds": 60,
    "lookback": 120,
    "entry_z_abs": 2.2,
    "tp_z_abs": 0.5,
    "sl_z_abs": 3.0,
    "min_corr": 0.75,
    "min_rolling_corr": 0.70,
    "max_adf_p": 0.05,
    "recent_adf_window": 60,
    "half_life_min": 2,
    "half_life_max": 72,
    "max_holding_bars": 48,
    "demo_equity": 10000.0,
    "risk_fraction_per_trade": 0.08,
    "max_open_positions": 6,
    "cooldown_bars_same_pair": 8,
    "min_ml_rows": 80,
    "ml_retrain_every_closed_trades": 20,
    "ml_prob_threshold": 0.56,
    "ml_min_auc_to_activate": 0.52,
    "data_dir": "./stat_arb_bot_data",
    "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),
    "pair_groups": {
        "L1": ["BTC", "ETH", "SOL", "AVAX", "ADA", "SUI", "ATOM", "NEAR"],
        "L2": ["ARB", "OP", "STRK", "MNT"],
        "GAMING": ["IMX", "GALA", "AXS", "RONIN", "BEAM"],
        "MEME": ["DOGE", "SHIB", "PEPE", "BONK"],
        "AI": ["FET", "TAO", "RENDER"],
        "DEFI": ["AAVE", "UNI", "MKR", "INJ"],
    },
}


# ============================================================
# Files / Directories
# ============================================================

DATA_DIR = Path(CONFIG["data_dir"])
DATA_DIR.mkdir(parents=True, exist_ok=True)
SIGNALS_CSV = DATA_DIR / "signals_log.csv"
POSITIONS_CSV = DATA_DIR / "positions_log.csv"
CLOSED_TRADES_CSV = DATA_DIR / "closed_trades_log.csv"
DATASET_CSV = DATA_DIR / "ml_dataset.csv"
MODEL_PATH = DATA_DIR / "ml_model.joblib"
META_PATH = DATA_DIR / "ml_meta.json"
BOT_STATE_PATH = DATA_DIR / "bot_state.json"


# ============================================================
# Helpers
# ============================================================

def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


def safe_float(x, default=np.nan):
    try:
        return float(x)
    except Exception:
        return default


def append_row_csv(path: Path, row: Dict):
    df = pd.DataFrame([row])
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def send_telegram(message: str):
    token = CONFIG["telegram_bot_token"]
    chat_id = CONFIG["telegram_chat_id"]
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": message}, timeout=10)
    except Exception:
        pass


def load_state() -> Dict:
    if BOT_STATE_PATH.exists():
        return json.loads(BOT_STATE_PATH.read_text())
    return {"open_positions": [], "last_pair_entry_bar": {}, "closed_trades_since_retrain": 0}


def save_state(state: Dict):
    BOT_STATE_PATH.write_text(json.dumps(state, indent=2))


def linear_half_life(spread: pd.Series) -> float:
    spread = spread.dropna()
    if len(spread) < 20:
        return np.nan
    lag = spread.shift(1).dropna()
    delta = spread.diff().dropna()
    aligned = pd.concat([lag, delta], axis=1).dropna()
    if aligned.empty:
        return np.nan
    y = aligned.iloc[:, 1].values
    x = add_constant(aligned.iloc[:, 0].values)
    try:
        beta = OLS(y, x).fit().params[1]
        if beta >= 0:
            return np.nan
        return -np.log(2) / beta
    except Exception:
        return np.nan


def calc_hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> float:
    aligned = pd.concat([log_a, log_b], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan
    y = aligned.iloc[:, 0].values
    x = add_constant(aligned.iloc[:, 1].values)
    try:
        model = OLS(y, x).fit()
        return float(model.params[1])
    except Exception:
        return np.nan


def adf_pvalue(series: pd.Series) -> float:
    series = series.dropna()
    if len(series) < 30:
        return np.nan
    try:
        return float(adfuller(series, autolag="AIC")[1])
    except Exception:
        return np.nan


def rolling_corr(a: pd.Series, b: pd.Series, window: int = 60) -> float:
    aligned = pd.concat([a, b], axis=1).dropna()
    if len(aligned) < window:
        return np.nan
    return float(aligned.iloc[:, 0].rolling(window).corr(aligned.iloc[:, 1]).iloc[-1])


def classify_pair_type(group_a: str, group_b: str) -> str:
    if group_a == group_b:
        return "same_sector"
    if {group_a, group_b} <= {"L1", "L2"}:
        return "infrastructure"
    return "cross_sector"


def long_short_labels(z: float, base_a: str, base_b: str) -> Tuple[str, str, str]:
    # spread = logA - hr*logB
    # z > 0 => A rich / B cheap => short A, long B
    if z > 0:
        return "short_spread", f"SHORT {base_a} / LONG {base_b}", "mean_reversion_down"
    return "long_spread", f"LONG {base_a} / SHORT {base_b}", "mean_reversion_up"


# ============================================================
# Exchange / Data Layer
# ============================================================

class MarketData:
    def __init__(self, exchange_id: str):
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({"enableRateLimit": True})
        self.markets = self.exchange.load_markets()

    def symbol_exists(self, base: str, quote: str) -> bool:
        return f"{base}/{quote}" in self.markets

    def fetch_ohlcv_df(self, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not raw:
            raise ValueError(f"No OHLCV data for {symbol}")
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df


# ============================================================
# Universe Builder
# ============================================================

@dataclass
class PairSpec:
    symbol_a: str
    symbol_b: str
    base_a: str
    base_b: str
    group_a: str
    group_b: str
    pair_type: str
    sector: str


def build_pair_universe(md: MarketData) -> List[PairSpec]:
    quote = CONFIG["quote_asset"]
    groups = CONFIG["pair_groups"]
    members = []
    for g, assets in groups.items():
        for asset in assets:
            if md.symbol_exists(asset, quote):
                members.append((asset, g))

    pairs: List[PairSpec] = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            a, ga = members[i]
            b, gb = members[j]
            pair_type = classify_pair_type(ga, gb)
            sector = f"{ga.lower()}_vs_{gb.lower()}" if ga != gb else ga.lower()
            pairs.append(
                PairSpec(
                    symbol_a=f"{a}/{quote}",
                    symbol_b=f"{b}/{quote}",
                    base_a=a,
                    base_b=b,
                    group_a=ga,
                    group_b=gb,
                    pair_type=pair_type,
                    sector=sector,
                )
            )
    return pairs


# ============================================================
# Feature Engineering
# ============================================================

@dataclass
class PairFeatures:
    timestamp: str
    pair: str
    pair_type: str
    sector: str
    signal: str
    side_text: str
    score: float
    zscore: float
    corr: float
    rolling_corr: float
    adf_p: float
    recent_adf_p: float
    half_life: float
    hedge_ratio: float
    vol_ratio: float
    spread: float
    spread_mean: float
    spread_std: float
    spread_percentile: float
    z_change_1: float
    z_change_3: float
    ret_a_1: float
    ret_b_1: float
    ret_a_6: float
    ret_b_6: float
    price_a: float
    price_b: float
    market_regime: str
    session: str
    group_a: str
    group_b: str


def classify_session(ts: pd.Timestamp) -> str:
    h = ts.hour
    if 0 <= h < 8:
        return "asia"
    if 8 <= h < 16:
        return "london"
    return "new_york"


def regime_from_btc(btc_close: pd.Series) -> str:
    if len(btc_close) < 60:
        return "unknown"
    ma_fast = btc_close.rolling(20).mean().iloc[-1]
    ma_slow = btc_close.rolling(50).mean().iloc[-1]
    vol = btc_close.pct_change().rolling(20).std().iloc[-1]
    if pd.isna(ma_fast) or pd.isna(ma_slow) or pd.isna(vol):
        return "unknown"
    if ma_fast > ma_slow and vol > btc_close.pct_change().rolling(60).std().iloc[-1]:
        return "trend_high_vol"
    if ma_fast > ma_slow:
        return "trend_low_vol"
    if ma_fast <= ma_slow and vol > btc_close.pct_change().rolling(60).std().iloc[-1]:
        return "range_high_vol"
    return "range_low_vol"


def compute_pair_features(df_a: pd.DataFrame, df_b: pd.DataFrame, pair: PairSpec, btc_df: pd.DataFrame) -> Optional[PairFeatures]:
    lookback = CONFIG["lookback"]
    recent_window = CONFIG["recent_adf_window"]

    merged = pd.merge(df_a[["timestamp", "close"]], df_b[["timestamp", "close"]], on="timestamp", suffixes=("_a", "_b"))
    if len(merged) < max(lookback, recent_window) + 5:
        return None

    merged = merged.tail(max(lookback, recent_window) + 10).copy()
    close_a = merged["close_a"].astype(float)
    close_b = merged["close_b"].astype(float)
    log_a = np.log(close_a)
    log_b = np.log(close_b)

    corr = close_a.pct_change().corr(close_b.pct_change())
    r_corr = rolling_corr(close_a.pct_change(), close_b.pct_change(), window=min(60, len(merged) - 1))
    hedge_ratio = calc_hedge_ratio(log_a.tail(lookback), log_b.tail(lookback))
    if pd.isna(hedge_ratio):
        return None

    spread = log_a - hedge_ratio * log_b
    spread_lb = spread.tail(lookback)
    spread_mean = spread_lb.mean()
    spread_std = spread_lb.std(ddof=0)
    if spread_std <= 0 or pd.isna(spread_std):
        return None
    z = (spread_lb.iloc[-1] - spread_mean) / spread_std
    z_hist = (spread_lb - spread_mean) / spread_std
    z_chg_1 = z_hist.iloc[-1] - z_hist.iloc[-2] if len(z_hist) >= 2 else np.nan
    z_chg_3 = z_hist.iloc[-1] - z_hist.iloc[-4] if len(z_hist) >= 4 else np.nan

    adf_p = adf_pvalue(spread_lb)
    recent_adf = adf_pvalue(spread.tail(recent_window))
    hl = linear_half_life(spread_lb)

    ret_a = close_a.pct_change()
    ret_b = close_b.pct_change()
    vol_ratio = (ret_a.tail(24).std() / ret_b.tail(24).std()) if ret_b.tail(24).std() not in [0, np.nan] else np.nan

    side, side_text, _ = long_short_labels(z, pair.base_a, pair.base_b)
    percentile = float((spread_lb.rank(pct=True).iloc[-1]))

    ts = merged["timestamp"].iloc[-1]
    btc_regime = regime_from_btc(btc_df["close"])
    session = classify_session(ts)

    # Score: transparent weighted heuristic; not magic.
    score = 0.0
    score += min(abs(z) / 3.0, 1.0) * 25
    score += max(min((corr - 0.6) / 0.4, 1.0), 0.0) * 15 if not pd.isna(corr) else 0
    score += max(min((r_corr - 0.6) / 0.4, 1.0), 0.0) * 15 if not pd.isna(r_corr) else 0
    score += max(min((0.05 - adf_p) / 0.05, 1.0), 0.0) * 20 if not pd.isna(adf_p) else 0
    if not pd.isna(hl):
        if CONFIG["half_life_min"] <= hl <= CONFIG["half_life_max"]:
            score += 15
        elif hl < CONFIG["half_life_min"] * 1.5 or hl < CONFIG["half_life_max"] * 1.5:
            score += 7
    if pair.group_a == pair.group_b:
        score += 10
    score = round(score, 2)

    return PairFeatures(
        timestamp=str(ts),
        pair=f"{pair.base_a}-{pair.base_b}",
        pair_type=pair.pair_type,
        sector=pair.sector,
        signal=side,
        side_text=side_text,
        score=score,
        zscore=float(z),
        corr=safe_float(corr),
        rolling_corr=safe_float(r_corr),
        adf_p=safe_float(adf_p),
        recent_adf_p=safe_float(recent_adf),
        half_life=safe_float(hl),
        hedge_ratio=safe_float(hedge_ratio),
        vol_ratio=safe_float(vol_ratio),
        spread=safe_float(spread_lb.iloc[-1]),
        spread_mean=safe_float(spread_mean),
        spread_std=safe_float(spread_std),
        spread_percentile=safe_float(percentile),
        z_change_1=safe_float(z_chg_1),
        z_change_3=safe_float(z_chg_3),
        ret_a_1=safe_float(ret_a.iloc[-1]),
        ret_b_1=safe_float(ret_b.iloc[-1]),
        ret_a_6=safe_float(close_a.pct_change(6).iloc[-1]),
        ret_b_6=safe_float(close_b.pct_change(6).iloc[-1]),
        price_a=safe_float(close_a.iloc[-1]),
        price_b=safe_float(close_b.iloc[-1]),
        market_regime=btc_regime,
        session=session,
        group_a=pair.group_a,
        group_b=pair.group_b,
    )


# ============================================================
# ML Layer
# ============================================================

FEATURE_COLUMNS_NUMERIC = [
    "score", "zscore", "corr", "rolling_corr", "adf_p", "recent_adf_p", "half_life",
    "hedge_ratio", "vol_ratio", "spread_percentile", "z_change_1", "z_change_3",
    "ret_a_1", "ret_b_1", "ret_a_6", "ret_b_6",
]
FEATURE_COLUMNS_CATEGORICAL = ["pair_type", "sector", "market_regime", "session", "group_a", "group_b"]
TARGET_COLUMN = "target_win"


class MLFilter:
    def __init__(self):
        self.pipeline = None
        self.meta = {"active": False, "auc": None, "trained_rows": 0, "trained_at": None}
        self.load()

    def load(self):
        if MODEL_PATH.exists() and META_PATH.exists():
            self.pipeline = joblib.load(MODEL_PATH)
            self.meta = json.loads(META_PATH.read_text())

    def save(self):
        if self.pipeline is not None:
            joblib.dump(self.pipeline, MODEL_PATH)
            META_PATH.write_text(json.dumps(self.meta, indent=2))

    def train_if_possible(self):
        if not DATASET_CSV.exists():
            return
        df = pd.read_csv(DATASET_CSV)
        if len(df) < CONFIG["min_ml_rows"]:
            self.meta.update({"active": False, "auc": None, "trained_rows": len(df)})
            self.save()
            return

        df = df.dropna(subset=[TARGET_COLUMN]).copy()
        if len(df) < CONFIG["min_ml_rows"]:
            return

        X = df[FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL]
        y = df[TARGET_COLUMN].astype(int)

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), FEATURE_COLUMNS_NUMERIC),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]), FEATURE_COLUMNS_CATEGORICAL),
            ]
        )

        base_model = RandomForestClassifier(
            n_estimators=250,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            class_weight="balanced_subsample",
        )

        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", CalibratedClassifierCV(base_model, method="sigmoid", cv=3)),
        ])

        aucs = []
        tscv = TimeSeriesSplit(n_splits=4)
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline.fit(X_train, y_train)
            proba = pipeline.predict_proba(X_test)[:, 1]
            try:
                aucs.append(roc_auc_score(y_test, proba))
            except Exception:
                pass

        auc = float(np.mean(aucs)) if aucs else np.nan
        pipeline.fit(X, y)
        self.pipeline = pipeline
        self.meta = {
            "active": bool(not pd.isna(auc) and auc >= CONFIG["ml_min_auc_to_activate"]),
            "auc": None if pd.isna(auc) else round(auc, 4),
            "trained_rows": int(len(df)),
            "trained_at": utc_now_str(),
        }
        self.save()

    def predict_trade_quality(self, feature_row: Dict) -> Tuple[Optional[float], bool]:
        if self.pipeline is None or not self.meta.get("active", False):
            return None, True
        row = pd.DataFrame([{k: feature_row.get(k) for k in FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL}])
        prob = float(self.pipeline.predict_proba(row)[:, 1][0])
        return prob, prob >= CONFIG["ml_prob_threshold"]


# ============================================================
# Position Logic
# ============================================================

@dataclass
class OpenPosition:
    position_id: str
    entry_time: str
    pair: str
    base_a: str
    base_b: str
    symbol_a: str
    symbol_b: str
    pair_type: str
    sector: str
    group_a: str
    group_b: str
    signal: str
    side_text: str
    score: float
    zscore_entry: float
    corr_entry: float
    rolling_corr_entry: float
    adf_p_entry: float
    recent_adf_p_entry: float
    half_life_entry: float
    hedge_ratio_entry: float
    vol_ratio_entry: float
    spread_entry: float
    spread_mean_entry: float
    spread_std_entry: float
    spread_percentile_entry: float
    z_change_1_entry: float
    z_change_3_entry: float
    ret_a_1_entry: float
    ret_b_1_entry: float
    ret_a_6_entry: float
    ret_b_6_entry: float
    price_a_entry: float
    price_b_entry: float
    market_regime_entry: str
    session_entry: str
    ml_prob: Optional[float]
    ml_passed: bool
    demo_equity_at_entry: float
    gross_notional: float
    notional_a: float
    notional_b: float
    qty_a: float
    qty_b: float
    tp_z_abs: float
    sl_z_abs: float
    max_holding_bars: int
    bars_held: int
    max_favorable_pnl: float
    max_adverse_pnl: float


def position_sizing(demo_equity: float, risk_fraction: float, hedge_ratio: float, price_a: float, price_b: float) -> Tuple[float, float, float, float, float]:
    gross_notional = demo_equity * risk_fraction
    hr_abs = abs(hedge_ratio)
    if hr_abs <= 0 or np.isnan(hr_abs):
        w_a = 0.5
        w_b = 0.5
    else:
        w_a = 1.0 / (1.0 + hr_abs)
        w_b = hr_abs / (1.0 + hr_abs)

    notional_a = gross_notional * w_a
    notional_b = gross_notional * w_b
    qty_a = notional_a / price_a if price_a > 0 else 0.0
    qty_b = notional_b / price_b if price_b > 0 else 0.0
    return gross_notional, notional_a, notional_b, qty_a, qty_b


def compute_position_pnl(pos: OpenPosition, price_a: float, price_b: float) -> float:
    # Signal based on spread direction
    # long_spread = long A short B
    if pos.signal == "long_spread":
        pnl_a = pos.qty_a * (price_a - pos.price_a_entry)
        pnl_b = pos.qty_b * (pos.price_b_entry - price_b)
    else:
        pnl_a = pos.qty_a * (pos.price_a_entry - price_a)
        pnl_b = pos.qty_b * (price_b - pos.price_b_entry)
    return float(pnl_a + pnl_b)


# ============================================================
# Dataset Builder
# ============================================================

def rebuild_dataset_from_closed_trades() -> pd.DataFrame:
    if not CLOSED_TRADES_CSV.exists():
        return pd.DataFrame()
    closed = pd.read_csv(CLOSED_TRADES_CSV)
    if closed.empty:
        return closed

    closed[TARGET_COLUMN] = (closed["pnl_usd"] > 0).astype(int)
    dataset_cols = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL + [TARGET_COLUMN, "pair", "exit_reason", "pnl_usd", "holding_bars"]
    keep = [c for c in dataset_cols if c in closed.columns]
    dataset = closed[keep].copy()
    dataset.to_csv(DATASET_CSV, index=False)
    return dataset


# ============================================================
# Signal Evaluation
# ============================================================

def passes_base_filters(f: PairFeatures) -> Tuple[bool, str]:
    if abs(f.zscore) < CONFIG["entry_z_abs"]:
        return False, "zscore_too_small"
    if pd.isna(f.corr) or f.corr < CONFIG["min_corr"]:
        return False, "corr_too_low"
    if pd.isna(f.rolling_corr) or f.rolling_corr < CONFIG["min_rolling_corr"]:
        return False, "rolling_corr_too_low"
    if pd.isna(f.adf_p) or f.adf_p > CONFIG["max_adf_p"]:
        return False, "adf_fail"
    if pd.isna(f.recent_adf_p) or f.recent_adf_p > 0.10:
        return False, "recent_adf_fail"
    if pd.isna(f.half_life) or not (CONFIG["half_life_min"] <= f.half_life <= CONFIG["half_life_max"]):
        return False, "half_life_out_of_range"
    return True, "ok"


# ============================================================
# Main Bot Engine
# ============================================================

class StatArbMLBot:
    def __init__(self):
        self.md = MarketData(CONFIG["exchange_id"])
        self.pairs = build_pair_universe(self.md)
        self.state = load_state()
        self.ml = MLFilter()
        self.demo_equity = CONFIG["demo_equity"]
        send_telegram(f"STAT ARB ML BOT started\nPairs loaded: {len(self.pairs)}\nTimeframe: {CONFIG['timeframe']}\nUTC: {utc_now_str()}")

    def maybe_retrain_model(self):
        if self.state.get("closed_trades_since_retrain", 0) >= CONFIG["ml_retrain_every_closed_trades"]:
            rebuild_dataset_from_closed_trades()
            self.ml.train_if_possible()
            self.state["closed_trades_since_retrain"] = 0
            save_state(self.state)
            send_telegram(f"ML retrained\nActive: {self.ml.meta.get('active')}\nAUC: {self.ml.meta.get('auc')}\nRows: {self.ml.meta.get('trained_rows')}")

    def fetch_all_needed(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        symbols = sorted({p.symbol_a for p in self.pairs} | {p.symbol_b for p in self.pairs} | {"BTC/USDT"})
        data = {}
        for s in symbols:
            try:
                data[s] = self.md.fetch_ohlcv_df(s, CONFIG["timeframe"], CONFIG["history_limit"])
            except Exception as e:
                print(f"[WARN] fetch failed for {s}: {e}")
        btc_df = data.get("BTC/USDT")
        if btc_df is None:
            raise RuntimeError("BTC/USDT data unavailable; needed for regime classification")
        return data, btc_df

    def open_positions(self) -> List[OpenPosition]:
        return [OpenPosition(**p) for p in self.state.get("open_positions", [])]

    def persist_open_positions(self, positions: List[OpenPosition]):
        self.state["open_positions"] = [asdict(p) for p in positions]
        save_state(self.state)

    def evaluate_new_entries(self, market_data: Dict[str, pd.DataFrame], btc_df: pd.DataFrame):
        open_positions = self.open_positions()
        open_pairs = {p.pair for p in open_positions}
        if len(open_positions) >= CONFIG["max_open_positions"]:
            return

        candidates: List[Tuple[PairSpec, PairFeatures, Optional[float], bool]] = []

        for pair in self.pairs:
            if f"{pair.base_a}-{pair.base_b}" in open_pairs:
                continue
            df_a = market_data.get(pair.symbol_a)
            df_b = market_data.get(pair.symbol_b)
            if df_a is None or df_b is None:
                continue
            f = compute_pair_features(df_a, df_b, pair, btc_df)
            if f is None:
                continue

            ok, reason = passes_base_filters(f)
            signal_row = asdict(f)
            signal_row.update({"filter_pass": ok, "filter_reason": reason, "logged_at": utc_now_str()})
            append_row_csv(SIGNALS_CSV, signal_row)
            if not ok:
                continue

            last_pair_bar = self.state.get("last_pair_entry_bar", {}).get(f.pair)
            current_bar = f.timestamp
            if last_pair_bar == current_bar:
                continue

            ml_prob, ml_pass = self.ml.predict_trade_quality(signal_row)
            candidates.append((pair, f, ml_prob, ml_pass))

        # rank by heuristic score then ML probability when available
        candidates.sort(key=lambda x: ((x[2] if x[2] is not None else 0.5) * 100 + x[1].score), reverse=True)

        for pair, f, ml_prob, ml_pass in candidates:
            open_positions = self.open_positions()
            if len(open_positions) >= CONFIG["max_open_positions"]:
                break
            if not ml_pass:
                continue

            gross_notional, notional_a, notional_b, qty_a, qty_b = position_sizing(
                self.demo_equity,
                CONFIG["risk_fraction_per_trade"],
                f.hedge_ratio,
                f.price_a,
                f.price_b,
            )

            pos = OpenPosition(
                position_id=f"{f.pair}_{int(pd.Timestamp(f.timestamp).timestamp())}",
                entry_time=f.timestamp,
                pair=f.pair,
                base_a=pair.base_a,
                base_b=pair.base_b,
                symbol_a=pair.symbol_a,
                symbol_b=pair.symbol_b,
                pair_type=f.pair_type,
                sector=f.sector,
                group_a=f.group_a,
                group_b=f.group_b,
                signal=f.signal,
                side_text=f.side_text,
                score=f.score,
                zscore_entry=f.zscore,
                corr_entry=f.corr,
                rolling_corr_entry=f.rolling_corr,
                adf_p_entry=f.adf_p,
                recent_adf_p_entry=f.recent_adf_p,
                half_life_entry=f.half_life,
                hedge_ratio_entry=f.hedge_ratio,
                vol_ratio_entry=f.vol_ratio,
                spread_entry=f.spread,
                spread_mean_entry=f.spread_mean,
                spread_std_entry=f.spread_std,
                spread_percentile_entry=f.spread_percentile,
                z_change_1_entry=f.z_change_1,
                z_change_3_entry=f.z_change_3,
                ret_a_1_entry=f.ret_a_1,
                ret_b_1_entry=f.ret_b_1,
                ret_a_6_entry=f.ret_a_6,
                ret_b_6_entry=f.ret_b_6,
                price_a_entry=f.price_a,
                price_b_entry=f.price_b,
                market_regime_entry=f.market_regime,
                session_entry=f.session,
                ml_prob=ml_prob,
                ml_passed=ml_pass,
                demo_equity_at_entry=self.demo_equity,
                gross_notional=gross_notional,
                notional_a=notional_a,
                notional_b=notional_b,
                qty_a=qty_a,
                qty_b=qty_b,
                tp_z_abs=CONFIG["tp_z_abs"],
                sl_z_abs=CONFIG["sl_z_abs"],
                max_holding_bars=CONFIG["max_holding_bars"],
                bars_held=0,
                max_favorable_pnl=0.0,
                max_adverse_pnl=0.0,
            )
            open_positions.append(pos)
            self.persist_open_positions(open_positions)
            self.state.setdefault("last_pair_entry_bar", {})[f.pair] = f.timestamp
            save_state(self.state)

            pos_row = asdict(pos)
            pos_row.update({"event": "OPEN", "logged_at": utc_now_str()})
            append_row_csv(POSITIONS_CSV, pos_row)

            msg = (
                f"🚨 STAT ARB OPEN\n"
                f"Pair: {f.pair}\n"
                f"Pair Type: {f.pair_type}\n"
                f"Sector: {f.sector}\n"
                f"Signal: {f.side_text}\n"
                f"Score: {f.score}\n"
                f"Z: {f.zscore:.3f}\n"
                f"Corr: {f.corr:.3f} | RollCorr: {f.rolling_corr:.3f}\n"
                f"ADF p: {f.adf_p:.4f} | Recent: {f.recent_adf_p:.4f}\n"
                f"HL: {f.half_life:.2f}\n"
                f"Hedge Ratio: {f.hedge_ratio:.4f}\n"
                f"Notional A: ${notional_a:.2f}\n"
                f"Notional B: ${notional_b:.2f}\n"
                f"ML Prob: {('n/a' if ml_prob is None else round(ml_prob, 3))}\n"
                f"TP when |Z| < {CONFIG['tp_z_abs']}\n"
                f"SL when |Z| > {CONFIG['sl_z_abs']} or cointegration breaks"
            )
            print(msg)
            send_telegram(msg)

    def manage_open_positions(self, market_data: Dict[str, pd.DataFrame], btc_df: pd.DataFrame):
        positions = self.open_positions()
        if not positions:
            return

        survivors: List[OpenPosition] = []

        for pos in positions:
            # reconstruct pair spec for fresh feature computation
            pair_spec = PairSpec(
                symbol_a=pos.symbol_a,
                symbol_b=pos.symbol_b,
                base_a=pos.base_a,
                base_b=pos.base_b,
                group_a=pos.group_a,
                group_b=pos.group_b,
                pair_type=pos.pair_type,
                sector=pos.sector,
            )
            df_a = market_data.get(pos.symbol_a)
            df_b = market_data.get(pos.symbol_b)
            if df_a is None or df_b is None:
                survivors.append(pos)
                continue
            f = compute_pair_features(df_a, df_b, pair_spec, btc_df)
            if f is None:
                survivors.append(pos)
                continue

            pos.bars_held += 1
            pnl = compute_position_pnl(pos, f.price_a, f.price_b)
            pos.max_favorable_pnl = max(pos.max_favorable_pnl, pnl)
            pos.max_adverse_pnl = min(pos.max_adverse_pnl, pnl)

            exit_reason = None
            if abs(f.zscore) < pos.tp_z_abs:
                exit_reason = "tp_z_reversion"
            elif abs(f.zscore) > pos.sl_z_abs:
                exit_reason = "sl_z_expansion"
            elif pd.isna(f.adf_p) or f.adf_p > 0.10 or pd.isna(f.recent_adf_p) or f.recent_adf_p > 0.15:
                exit_reason = "cointegration_break"
            elif pd.isna(f.rolling_corr) or f.rolling_corr < 0.55:
                exit_reason = "corr_break"
            elif pos.bars_held >= pos.max_holding_bars:
                exit_reason = "max_holding_time"

            if exit_reason is None:
                survivors.append(pos)
                continue

            pnl_pct = pnl / max(pos.gross_notional, 1e-9)
            r_multiple = pnl / max(pos.demo_equity_at_entry * CONFIG["risk_fraction_per_trade"], 1e-9)
            self.demo_equity += pnl

            closed_row = {
                **asdict(pos),
                "exit_time": f.timestamp,
                "exit_reason": exit_reason,
                "zscore_exit": f.zscore,
                "corr_exit": f.corr,
                "rolling_corr_exit": f.rolling_corr,
                "adf_p_exit": f.adf_p,
                "recent_adf_p_exit": f.recent_adf_p,
                "half_life_exit": f.half_life,
                "price_a_exit": f.price_a,
                "price_b_exit": f.price_b,
                "spread_exit": f.spread,
                "pnl_usd": pnl,
                "pnl_pct": pnl_pct,
                "R_multiple": r_multiple,
                "mfe_usd": pos.max_favorable_pnl,
                "mae_usd": pos.max_adverse_pnl,
                "holding_bars": pos.bars_held,
                "logged_at": utc_now_str(),
                # feature mirror for dataset builder
                "score": pos.score,
                "zscore": pos.zscore_entry,
                "corr": pos.corr_entry,
                "rolling_corr": pos.rolling_corr_entry,
                "adf_p": pos.adf_p_entry,
                "recent_adf_p": pos.recent_adf_p_entry,
                "half_life": pos.half_life_entry,
                "hedge_ratio": pos.hedge_ratio_entry,
                "vol_ratio": pos.vol_ratio_entry,
                "spread_percentile": pos.spread_percentile_entry,
                "z_change_1": pos.z_change_1_entry,
                "z_change_3": pos.z_change_3_entry,
                "ret_a_1": pos.ret_a_1_entry,
                "ret_b_1": pos.ret_b_1_entry,
                "ret_a_6": pos.ret_a_6_entry,
                "ret_b_6": pos.ret_b_6_entry,
                "market_regime": pos.market_regime_entry,
                "session": pos.session_entry,
                "group_a": pos.group_a,
                "group_b": pos.group_b,
            }
            append_row_csv(CLOSED_TRADES_CSV, closed_row)
            append_row_csv(POSITIONS_CSV, {**closed_row, "event": "CLOSE"})

            self.state["closed_trades_since_retrain"] = self.state.get("closed_trades_since_retrain", 0) + 1
            save_state(self.state)

            msg = (
                f"✅ STAT ARB CLOSE\n"
                f"Pair: {pos.pair}\n"
                f"Reason: {exit_reason}\n"
                f"Entry: {pos.side_text}\n"
                f"PnL: ${pnl:.2f} ({pnl_pct*100:.2f}%)\n"
                f"R: {r_multiple:.2f}\n"
                f"Bars held: {pos.bars_held}\n"
                f"MFE: ${pos.max_favorable_pnl:.2f} | MAE: ${pos.max_adverse_pnl:.2f}\n"
                f"Demo Equity: ${self.demo_equity:.2f}"
            )
            print(msg)
            send_telegram(msg)

        self.persist_open_positions(survivors)

    def run_cycle(self):
        market_data, btc_df = self.fetch_all_needed()
        self.manage_open_positions(market_data, btc_df)
        self.evaluate_new_entries(market_data, btc_df)
        self.maybe_retrain_model()

    def run_forever(self):
        print(f"[START] UTC {utc_now_str()} | pairs={len(self.pairs)} | timeframe={CONFIG['timeframe']}")
        while True:
            try:
                self.run_cycle()
                print(f"[SLEEP] {CONFIG['poll_seconds']} sec | equity=${self.demo_equity:.2f} | open={len(self.open_positions())}")
            except KeyboardInterrupt:
                print("Stopped by user")
                break
            except Exception as e:
                err = f"[ERROR] {utc_now_str()}\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(err)
                send_telegram(err[:3500])
            time.sleep(CONFIG["poll_seconds"])


# ============================================================
# Optional one-shot utilities
# ============================================================

def summary_report() -> str:
    if not CLOSED_TRADES_CSV.exists():
        return "No closed trades yet."
    df = pd.read_csv(CLOSED_TRADES_CSV)
    if df.empty:
        return "No closed trades yet."
    wr = (df["pnl_usd"] > 0).mean() * 100
    avg_pnl = df["pnl_usd"].mean()
    pf_num = df.loc[df["pnl_usd"] > 0, "pnl_usd"].sum()
    pf_den = abs(df.loc[df["pnl_usd"] < 0, "pnl_usd"].sum())
    pf = pf_num / pf_den if pf_den > 0 else np.nan
    by_sector = df.groupby("sector")["pnl_usd"].agg(["count", "mean", "sum"]).sort_values("sum", ascending=False)
    lines = [
        "STAT ARB BOT SUMMARY",
        f"Closed trades: {len(df)}",
        f"Win rate: {wr:.2f}%",
        f"Avg pnl: ${avg_pnl:.2f}",
        f"Profit factor: {pf:.2f}" if not pd.isna(pf) else "Profit factor: n/a",
        "Top sectors:",
        by_sector.head(10).to_string(),
    ]
    return "\n".join(lines)


def run_once_for_test():
    bot = StatArbMLBot()
    bot.run_cycle()
    print(summary_report())


# ============================================================
# Entrypoint
# ============================================================

if __name__ == "__main__":
    mode = os.getenv("BOT_MODE", "live_loop").lower()
    if mode == "test_once":
        run_once_for_test()
    elif mode == "summary":
        print(summary_report())
    else:
        bot = StatArbMLBot()
        bot.run_forever()
