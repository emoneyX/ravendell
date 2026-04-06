#!/usr/bin/env python3
"""
STAT ARB ML BOT v3

Perbaikan dari v2:
-----------------
A) DATA QUALITY
   - demo_equity di-persist ke state.json (tidak lagi reset ke $10k setiap restart)
   - CSV writes di-buffer per cycle, tidak per baris (jauh lebih cepat)
   - CSV header-check tidak lagi melakukan disk read setiap baris

B) SIGNAL QUALITY
   - Pair stability tracker: tiap pair punya skor stabilitas yang decay jika
     corr / ADF memburuk N bar berturut-turut. Pair tidak stabil tidak dimasukkan antrian.
   - Regime gate: entry baru diblokir saat BTC dalam mode trend_high_vol
     (mean-reversion edge terbukti rendah saat trending kuat + volatil)
   - Spread zscore sekarang menggunakan ewm spread mean/std (lebih responsif dari rolling)

C) RISK MANAGEMENT
   - Kelly-fractional position sizing menggantikan flat 8%
     f* = (win_rate * avg_win - loss_rate * avg_loss) / avg_win, di-cap di max_risk_fraction
     Jika belum cukup data per pair_type, fallback ke base_risk_fraction
   - Circuit breaker: jika drawdown dari equity peak > circuit_breaker_dd,
     bot pause entry baru dan kirim alert Telegram
   - Max notional per posisi di-cap agar satu trade tidak dominasi portofolio

D) ML UPGRADE
   - Tambah GradientBoostingClassifier sebagai model kedua
   - Ensemble sederhana (rata-rata probabilitas RF + GBM) sebelum threshold
   - Walk-forward OOS AUC dihitung SEBELUM model diaktifkan
   - Model hanya aktif jika OOS AUC >= ml_min_auc_to_activate

E) BUG FIXES
   - Cooldown logic v2 menyimpan timestamp string, bisa gagal compare.
     v3 menyimpan Unix epoch int, compare numerik.
   - append_row_csv v2: tidak aman saat file ditulis concurrent dan sangat lambat.
     v3 pakai buffer per cycle + flush atomic dengan mode='a'.
   - Equity curve di analyze_stat_arb_results.py bisa kosong saat tidak ada
     closed trade — sudah di-guard.

Cara deploy:
    1. Stop v2: pkill -f stat_arb_ml_bot_v_2.py  (atau Ctrl+C)
    2. Copy v3 ke VPS
    3. python3 stat_arb_ml_bot_v_3.py
    Data dir yang sama, state.json kompatibel — tidak ada yang hilang.

Dependencies (sama dengan v2):
    pip install ccxt pandas numpy scikit-learn scipy requests joblib statsmodels
"""

from __future__ import annotations

import json
import math
import os
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
    raise RuntimeError("ccxt is required: pip install ccxt") from e


# ============================================================
# Configuration
# ============================================================

CONFIG = {
    # Exchange
    "exchange_id": "binance",
    "quote_asset": "USDT",
    "timeframe": "1h",
    "history_limit": 600,
    "poll_seconds": 60,         # seberapa sering cek bar baru

    # Feature computation
    "lookback": 120,
    "recent_adf_window": 60,

    # Entry filters (global defaults; sector-aware override di passes_base_filters)
    "entry_z_abs": 2.0,
    "min_corr": 0.70,
    "min_rolling_corr": 0.65,
    "max_adf_p": 0.10,
    "half_life_min": 2,
    "half_life_max": 72,

    # Exit rules
    "tp_z_abs": 0.5,
    "sl_z_abs": 3.0,
    "max_holding_bars": 48,

    # NEW: Regime gate — blokir entry baru saat BTC trending + volatile
    "block_entry_regimes": ["trend_high_vol"],

    # NEW: Pair stability tracker
    "stability_decay_bars": 4,       # jumlah bar berturut memburuk sebelum pair "unstable"
    "stability_corr_floor": 0.60,    # ambang batas "memburuk" untuk corr
    "stability_adf_ceiling": 0.15,   # ambang batas "memburuk" untuk ADF p-value

    # Position sizing
    "base_risk_fraction": 0.06,      # fallback saat belum cukup data Kelly
    "max_risk_fraction": 0.10,       # cap absolut
    "min_risk_fraction": 0.02,       # floor agar posisi tetap bermakna
    "kelly_lookback_trades": 30,     # jumlah closed trades minimum untuk Kelly per pair_type

    # Portfolio limits
    "max_open_positions": 3,
    "cooldown_bars_same_pair": 8,
    "demo_equity_initial": 10000.0,  # hanya dipakai jika state.json belum ada equity

    # NEW: Circuit breaker
    "circuit_breaker_dd": 0.15,      # pause jika drawdown dari peak > 15%

    # ML
    "min_ml_rows": 80,
    "ml_retrain_every_closed_trades": 20,
    "ml_prob_threshold": 0.56,
    "ml_min_auc_to_activate": 0.52,

    # I/O
    "data_dir": "./stat_arb_bot_data",
    "telegram_bot_token": os.getenv("TELEGRAM_BOT_TOKEN", ""),
    "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID", ""),

    # Universe
    "pair_groups": {
        "L1":     ["BTC", "ETH", "SOL", "AVAX", "ADA", "SUI", "ATOM", "NEAR"],
        "L2":     ["ARB", "OP", "STRK", "MNT"],
        "GAMING": ["IMX", "GALA", "AXS", "RONIN", "BEAM"],
        "MEME":   ["DOGE", "SHIB", "PEPE", "BONK"],
        "AI":     ["FET", "TAO", "RENDER"],
        "DEFI":   ["AAVE", "UNI", "MKR", "INJ"],
    },
}


# ============================================================
# Files / Directories
# ============================================================

DATA_DIR = Path(CONFIG["data_dir"])
DATA_DIR.mkdir(parents=True, exist_ok=True)

SIGNALS_CSV       = DATA_DIR / "signals_log.csv"
POSITIONS_CSV     = DATA_DIR / "positions_log.csv"
CLOSED_TRADES_CSV = DATA_DIR / "closed_trades_log.csv"
DATASET_CSV       = DATA_DIR / "ml_dataset.csv"
MODEL_PATH        = DATA_DIR / "ml_model.joblib"
META_PATH         = DATA_DIR / "ml_meta.json"
BOT_STATE_PATH    = DATA_DIR / "bot_state.json"


# ============================================================
# CSV Buffer  (NEW — replaces per-row append_row_csv)
# ============================================================

_csv_buffer: Dict[Path, List[Dict]] = defaultdict(list)


def buffer_row(path: Path, row: Dict) -> None:
    """Tambahkan baris ke buffer. Belum ditulis ke disk."""
    _csv_buffer[path].append(row)


def flush_csv_buffer() -> None:
    """Tulis semua buffer ke disk sekaligus di akhir setiap cycle."""
    for path, rows in _csv_buffer.items():
        if not rows:
            continue
        df = pd.DataFrame(rows)
        write_header = not path.exists()
        df.to_csv(path, mode="a", header=write_header, index=False)
    _csv_buffer.clear()


# ============================================================
# Helpers
# ============================================================

def utc_now_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z")


def utc_now_epoch() -> int:
    return int(datetime.now(timezone.utc).timestamp())


def safe_float(x, default: float = np.nan) -> float:
    try:
        v = float(x)
        return default if math.isnan(v) or math.isinf(v) else v
    except Exception:
        return default


def send_telegram(message: str) -> None:
    token = CONFIG["telegram_bot_token"]
    chat_id = CONFIG["telegram_chat_id"]
    if not token or not chat_id:
        return
    try:
        url = f"https://api.telegram.org/bot{token}/sendMessage"
        requests.post(url, data={"chat_id": chat_id, "text": message[:4000]}, timeout=10)
    except Exception:
        pass


def load_state() -> Dict:
    if BOT_STATE_PATH.exists():
        try:
            return json.loads(BOT_STATE_PATH.read_text())
        except Exception:
            pass
    return {
        "open_positions": [],
        "last_pair_entry_epoch": {},       # pair -> unix epoch int (v3: bukan bar timestamp string)
        "closed_trades_since_retrain": 0,
        "demo_equity": CONFIG["demo_equity_initial"],
        "peak_equity": CONFIG["demo_equity_initial"],
        "circuit_breaker_active": False,
        "pair_stability": {},              # pair -> {"bad_bars": int, "stable": bool}
    }


def save_state(state: Dict) -> None:
    BOT_STATE_PATH.write_text(json.dumps(state, indent=2, default=str))


# ============================================================
# Math / Stats helpers
# ============================================================

def linear_half_life(spread: pd.Series) -> float:
    spread = spread.dropna()
    if len(spread) < 20:
        return np.nan
    aligned = pd.concat([spread.shift(1), spread.diff()], axis=1).dropna()
    if aligned.empty:
        return np.nan
    y = aligned.iloc[:, 1].values
    x = add_constant(aligned.iloc[:, 0].values)
    try:
        beta = OLS(y, x).fit().params[1]
        return np.nan if beta >= 0 else -np.log(2) / beta
    except Exception:
        return np.nan


def calc_hedge_ratio(log_a: pd.Series, log_b: pd.Series) -> float:
    aligned = pd.concat([log_a, log_b], axis=1).dropna()
    if len(aligned) < 20:
        return np.nan
    y = aligned.iloc[:, 0].values
    x = add_constant(aligned.iloc[:, 1].values)
    try:
        return float(OLS(y, x).fit().params[1])
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


def rolling_corr_last(a: pd.Series, b: pd.Series, window: int = 60) -> float:
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


def long_short_labels(z: float, base_a: str, base_b: str) -> Tuple[str, str]:
    if z > 0:
        return "short_spread", f"SHORT {base_a} / LONG {base_b}"
    return "long_spread", f"LONG {base_a} / SHORT {base_b}"


def classify_session(ts: pd.Timestamp) -> str:
    h = ts.hour
    if h < 8:
        return "asia"
    if h < 16:
        return "london"
    return "new_york"


def regime_from_btc(btc_close: pd.Series) -> str:
    if len(btc_close) < 60:
        return "unknown"
    ma20 = btc_close.rolling(20).mean().iloc[-1]
    ma50 = btc_close.rolling(50).mean().iloc[-1]
    vol20 = btc_close.pct_change().rolling(20).std().iloc[-1]
    vol60 = btc_close.pct_change().rolling(60).std().iloc[-1]
    if any(pd.isna(x) for x in [ma20, ma50, vol20, vol60]):
        return "unknown"
    trending = ma20 > ma50
    high_vol = vol20 > vol60
    if trending and high_vol:
        return "trend_high_vol"
    if trending:
        return "trend_low_vol"
    if high_vol:
        return "range_high_vol"
    return "range_low_vol"


# ============================================================
# NEW: Kelly position sizing
# ============================================================

def kelly_fraction(pair_type: str, closed_trades_df: Optional[pd.DataFrame]) -> float:
    """
    Hitung Kelly fraction berdasarkan historical trades per pair_type.
    f* = (win_rate * avg_win/avg_loss - loss_rate) / (avg_win/avg_loss)
       = win_rate - loss_rate / (avg_win/avg_loss)
    Di-cap antara min_risk_fraction dan max_risk_fraction.
    Gunakan half-Kelly untuk safety.
    """
    base = CONFIG["base_risk_fraction"]
    mn   = CONFIG["min_risk_fraction"]
    mx   = CONFIG["max_risk_fraction"]

    if closed_trades_df is None or closed_trades_df.empty:
        return base

    if "pair_type" in closed_trades_df.columns:
        subset = closed_trades_df[closed_trades_df["pair_type"] == pair_type]
    else:
        subset = closed_trades_df

    if len(subset) < CONFIG["kelly_lookback_trades"]:
        return base

    pnl = pd.to_numeric(subset["pnl_usd"], errors="coerce").dropna()
    if pnl.empty:
        return base

    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    if wins.empty or losses.empty:
        return base

    win_rate  = len(wins) / len(pnl)
    loss_rate = 1.0 - win_rate
    avg_win   = float(wins.mean())
    avg_loss  = float(losses.abs().mean())

    if avg_loss <= 0:
        return base

    payoff_ratio = avg_win / avg_loss
    kelly_full   = win_rate - (loss_rate / payoff_ratio)

    # Half-Kelly untuk safety
    half_kelly = kelly_full * 0.5
    return float(np.clip(half_kelly, mn, mx))


def position_sizing(
    demo_equity: float,
    risk_fraction: float,
    hedge_ratio: float,
    price_a: float,
    price_b: float,
) -> Tuple[float, float, float, float, float]:
    gross_notional = demo_equity * risk_fraction
    hr_abs = abs(safe_float(hedge_ratio, 1.0))
    if hr_abs <= 0:
        w_a = w_b = 0.5
    else:
        w_a = 1.0 / (1.0 + hr_abs)
        w_b = hr_abs / (1.0 + hr_abs)
    notional_a = gross_notional * w_a
    notional_b = gross_notional * w_b
    qty_a = notional_a / price_a if price_a > 0 else 0.0
    qty_b = notional_b / price_b if price_b > 0 else 0.0
    return gross_notional, notional_a, notional_b, qty_a, qty_b


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
            raise ValueError(f"No OHLCV data: {symbol}")
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
            sector = ga.lower() if ga == gb else f"{ga.lower()}_vs_{gb.lower()}"
            pairs.append(PairSpec(
                symbol_a=f"{a}/{quote}", symbol_b=f"{b}/{quote}",
                base_a=a, base_b=b, group_a=ga, group_b=gb,
                pair_type=classify_pair_type(ga, gb), sector=sector,
            ))
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


def compute_pair_features(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    pair: PairSpec,
    btc_df: pd.DataFrame,
) -> Optional[PairFeatures]:
    lookback = CONFIG["lookback"]
    recent_window = CONFIG["recent_adf_window"]

    merged = pd.merge(
        df_a[["timestamp", "close"]],
        df_b[["timestamp", "close"]],
        on="timestamp", suffixes=("_a", "_b"),
    )
    if len(merged) < max(lookback, recent_window) + 5:
        return None

    merged = merged.tail(max(lookback, recent_window) + 10).copy()
    close_a = merged["close_a"].astype(float)
    close_b = merged["close_b"].astype(float)
    log_a = np.log(close_a)
    log_b = np.log(close_b)

    corr = close_a.pct_change().corr(close_b.pct_change())
    r_corr = rolling_corr_last(
        close_a.pct_change(), close_b.pct_change(),
        window=min(60, len(merged) - 1),
    )
    hedge_ratio = calc_hedge_ratio(log_a.tail(lookback), log_b.tail(lookback))
    if pd.isna(hedge_ratio):
        return None

    spread = log_a - hedge_ratio * log_b
    spread_lb = spread.tail(lookback)

    # NEW: EWM spread mean/std — lebih responsif daripada rolling window
    ewm_mean = float(spread_lb.ewm(span=lookback // 2).mean().iloc[-1])
    ewm_std  = float(spread_lb.ewm(span=lookback // 2).std().iloc[-1])
    if ewm_std <= 0 or pd.isna(ewm_std):
        return None

    spread_mean = spread_lb.mean()   # tetap simpan rolling mean untuk backward compat
    spread_std  = spread_lb.std(ddof=0)

    z = (spread_lb.iloc[-1] - ewm_mean) / ewm_std
    z_hist = (spread_lb - ewm_mean) / ewm_std
    z_chg_1 = z_hist.iloc[-1] - z_hist.iloc[-2] if len(z_hist) >= 2 else np.nan
    z_chg_3 = z_hist.iloc[-1] - z_hist.iloc[-4] if len(z_hist) >= 4 else np.nan

    adf_p      = adf_pvalue(spread_lb)
    recent_adf = adf_pvalue(spread.tail(recent_window))
    hl         = linear_half_life(spread_lb)

    ret_a = close_a.pct_change()
    ret_b = close_b.pct_change()
    std_b_24 = ret_b.tail(24).std()
    vol_ratio = (ret_a.tail(24).std() / std_b_24) if std_b_24 and std_b_24 > 0 else np.nan

    side, side_text = long_short_labels(z, pair.base_a, pair.base_b)
    percentile = float(spread_lb.rank(pct=True).iloc[-1])

    ts         = merged["timestamp"].iloc[-1]
    btc_regime = regime_from_btc(btc_df["close"])
    session    = classify_session(ts)

    # Score heuristik (sama dengan v2, tidak berubah)
    score = 0.0
    score += min(abs(z) / 3.0, 1.0) * 25
    if not pd.isna(corr):
        score += max(min((corr - 0.6) / 0.4, 1.0), 0.0) * 15
    if not pd.isna(r_corr):
        score += max(min((r_corr - 0.6) / 0.4, 1.0), 0.0) * 15
    if not pd.isna(adf_p):
        score += max(min((0.05 - adf_p) / 0.05, 1.0), 0.0) * 20
    if not pd.isna(hl):
        hl_min, hl_max = CONFIG["half_life_min"], CONFIG["half_life_max"]
        if hl_min <= hl <= hl_max:
            score += 15
        elif hl < hl_min * 1.5 or hl < hl_max * 1.5:
            score += 7
    if pair.group_a == pair.group_b:
        score += 10
    score = round(score, 2)

    return PairFeatures(
        timestamp=str(ts), pair=f"{pair.base_a}-{pair.base_b}",
        pair_type=pair.pair_type, sector=pair.sector,
        signal=side, side_text=side_text, score=score, zscore=float(z),
        corr=safe_float(corr), rolling_corr=safe_float(r_corr),
        adf_p=safe_float(adf_p), recent_adf_p=safe_float(recent_adf),
        half_life=safe_float(hl), hedge_ratio=safe_float(hedge_ratio),
        vol_ratio=safe_float(vol_ratio),
        spread=safe_float(spread_lb.iloc[-1]),
        spread_mean=safe_float(spread_mean), spread_std=safe_float(spread_std),
        spread_percentile=safe_float(percentile),
        z_change_1=safe_float(z_chg_1), z_change_3=safe_float(z_chg_3),
        ret_a_1=safe_float(ret_a.iloc[-1]), ret_b_1=safe_float(ret_b.iloc[-1]),
        ret_a_6=safe_float(close_a.pct_change(6).iloc[-1]),
        ret_b_6=safe_float(close_b.pct_change(6).iloc[-1]),
        price_a=safe_float(close_a.iloc[-1]), price_b=safe_float(close_b.iloc[-1]),
        market_regime=btc_regime, session=session,
        group_a=pair.group_a, group_b=pair.group_b,
    )


# ============================================================
# ML Layer  (v3: ensemble RF + GBM, walk-forward OOS)
# ============================================================

FEATURE_COLUMNS_NUMERIC = [
    "score", "zscore", "corr", "rolling_corr", "adf_p", "recent_adf_p", "half_life",
    "hedge_ratio", "vol_ratio", "spread_percentile", "z_change_1", "z_change_3",
    "ret_a_1", "ret_b_1", "ret_a_6", "ret_b_6",
]
FEATURE_COLUMNS_CATEGORICAL = [
    "pair_type", "sector", "market_regime", "session", "group_a", "group_b",
]
TARGET_COLUMN = "target_win"


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), FEATURE_COLUMNS_NUMERIC),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), FEATURE_COLUMNS_CATEGORICAL),
    ])


def _build_rf_pipeline() -> Pipeline:
    return Pipeline([
        ("prep", _build_preprocessor()),
        ("model", CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=250, max_depth=6, min_samples_leaf=5,
                random_state=42, class_weight="balanced_subsample",
            ), method="sigmoid", cv=3,
        )),
    ])


def _build_gbm_pipeline() -> Pipeline:
    return Pipeline([
        ("prep", _build_preprocessor()),
        ("model", CalibratedClassifierCV(
            GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=42,
            ), method="sigmoid", cv=3,
        )),
    ])


class MLFilter:
    def __init__(self):
        self.rf_pipeline  = None
        self.gbm_pipeline = None
        self.meta = {
            "active": False, "auc_rf": None, "auc_gbm": None,
            "trained_rows": 0, "trained_at": None,
        }
        self._load()

    def _load(self):
        rf_path  = MODEL_PATH.with_suffix(".rf.joblib")
        gbm_path = MODEL_PATH.with_suffix(".gbm.joblib")
        if rf_path.exists():
            self.rf_pipeline = joblib.load(rf_path)
        if gbm_path.exists():
            self.gbm_pipeline = joblib.load(gbm_path)
        if META_PATH.exists():
            try:
                self.meta = json.loads(META_PATH.read_text())
            except Exception:
                pass

    def _save(self):
        rf_path  = MODEL_PATH.with_suffix(".rf.joblib")
        gbm_path = MODEL_PATH.with_suffix(".gbm.joblib")
        if self.rf_pipeline is not None:
            joblib.dump(self.rf_pipeline, rf_path)
        if self.gbm_pipeline is not None:
            joblib.dump(self.gbm_pipeline, gbm_path)
        META_PATH.write_text(json.dumps(self.meta, indent=2))

    def train_if_possible(self) -> None:
        if not DATASET_CSV.exists():
            return
        df = pd.read_csv(DATASET_CSV).dropna(subset=[TARGET_COLUMN])
        if len(df) < CONFIG["min_ml_rows"]:
            self.meta.update({"active": False, "trained_rows": len(df)})
            self._save()
            return

        X = df[FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL]
        y = df[TARGET_COLUMN].astype(int)

        tscv = TimeSeriesSplit(n_splits=4)
        rf_aucs, gbm_aucs = [], []

        for train_idx, test_idx in tscv.split(X):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
            for model_cls, auc_list, build_fn in [
                ("RF",  rf_aucs,  _build_rf_pipeline),
                ("GBM", gbm_aucs, _build_gbm_pipeline),
            ]:
                try:
                    p = build_fn()
                    p.fit(X_tr, y_tr)
                    prob = p.predict_proba(X_te)[:, 1]
                    auc_list.append(roc_auc_score(y_te, prob))
                except Exception:
                    pass

        auc_rf  = float(np.mean(rf_aucs))  if rf_aucs  else np.nan
        auc_gbm = float(np.mean(gbm_aucs)) if gbm_aucs else np.nan

        # Train final models on all data
        self.rf_pipeline = _build_rf_pipeline()
        self.rf_pipeline.fit(X, y)
        self.gbm_pipeline = _build_gbm_pipeline()
        self.gbm_pipeline.fit(X, y)

        # Model aktif hanya jika salah satu OOS AUC memenuhi threshold
        min_auc = CONFIG["ml_min_auc_to_activate"]
        best_auc = max(v for v in [auc_rf, auc_gbm] if not np.isnan(v)) if (rf_aucs or gbm_aucs) else np.nan

        self.meta = {
            "active": bool(not np.isnan(best_auc) and best_auc >= min_auc),
            "auc_rf":  round(auc_rf, 4)  if not np.isnan(auc_rf)  else None,
            "auc_gbm": round(auc_gbm, 4) if not np.isnan(auc_gbm) else None,
            "best_auc": round(best_auc, 4) if not np.isnan(best_auc) else None,
            "trained_rows": int(len(df)),
            "trained_at": utc_now_str(),
        }
        self._save()

    def predict(self, feature_row: Dict) -> Tuple[Optional[float], bool]:
        """
        Ensemble prediction (rata-rata RF + GBM).
        Return (prob, passes_threshold).
        Jika ML tidak aktif, passes = True (tidak memblokir entry).
        """
        if not self.meta.get("active", False):
            return None, True

        row = pd.DataFrame([{k: feature_row.get(k) for k in FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL}])
        probs = []
        if self.rf_pipeline is not None:
            try:
                probs.append(float(self.rf_pipeline.predict_proba(row)[:, 1][0]))
            except Exception:
                pass
        if self.gbm_pipeline is not None:
            try:
                probs.append(float(self.gbm_pipeline.predict_proba(row)[:, 1][0]))
            except Exception:
                pass

        if not probs:
            return None, True

        prob = float(np.mean(probs))
        return prob, prob >= CONFIG["ml_prob_threshold"]


# ============================================================
# Signal Filters
# ============================================================

def passes_base_filters(f: PairFeatures) -> Tuple[bool, str]:
    if abs(f.zscore) < CONFIG["entry_z_abs"]:
        return False, "zscore_too_small"

    # Sector-aware thresholds
    corr_floor       = CONFIG["min_corr"]
    roll_floor       = CONFIG["min_rolling_corr"]
    adf_ceiling      = CONFIG["max_adf_p"]
    recent_ceiling   = 0.12

    if f.pair_type == "same_sector":
        corr_floor, roll_floor, adf_ceiling, recent_ceiling = 0.65, 0.60, 0.12, 0.15
    elif f.pair_type == "infrastructure":
        corr_floor, roll_floor, adf_ceiling, recent_ceiling = 0.68, 0.62, 0.12, 0.15

    if pd.isna(f.corr) or f.corr < corr_floor:
        return False, "corr_too_low"
    if pd.isna(f.rolling_corr) or f.rolling_corr < roll_floor:
        return False, "rolling_corr_too_low"
    if pd.isna(f.adf_p) or f.adf_p > adf_ceiling:
        return False, "adf_fail"
    if pd.isna(f.recent_adf_p) or f.recent_adf_p > recent_ceiling:
        return False, "recent_adf_fail"
    if pd.isna(f.half_life) or not (CONFIG["half_life_min"] <= f.half_life <= CONFIG["half_life_max"]):
        return False, "half_life_out_of_range"

    return True, "ok"


# ============================================================
# Position Logic
# ============================================================

@dataclass
class OpenPosition:
    position_id: str
    entry_time: str
    pair: str
    base_a: str; base_b: str
    symbol_a: str; symbol_b: str
    pair_type: str; sector: str
    group_a: str; group_b: str
    signal: str; side_text: str
    score: float
    zscore_entry: float
    corr_entry: float; rolling_corr_entry: float
    adf_p_entry: float; recent_adf_p_entry: float
    half_life_entry: float; hedge_ratio_entry: float
    vol_ratio_entry: float
    spread_entry: float; spread_mean_entry: float; spread_std_entry: float
    spread_percentile_entry: float
    z_change_1_entry: float; z_change_3_entry: float
    ret_a_1_entry: float; ret_b_1_entry: float
    ret_a_6_entry: float; ret_b_6_entry: float
    price_a_entry: float; price_b_entry: float
    market_regime_entry: str; session_entry: str
    ml_prob: Optional[float]; ml_passed: bool
    demo_equity_at_entry: float; risk_fraction_used: float
    gross_notional: float
    notional_a: float; notional_b: float
    qty_a: float; qty_b: float
    tp_z_abs: float; sl_z_abs: float; max_holding_bars: int
    bars_held: int
    max_favorable_pnl: float; max_adverse_pnl: float


def compute_position_pnl(pos: OpenPosition, price_a: float, price_b: float) -> float:
    if pos.signal == "long_spread":
        return pos.qty_a * (price_a - pos.price_a_entry) + pos.qty_b * (pos.price_b_entry - price_b)
    return pos.qty_a * (pos.price_a_entry - price_a) + pos.qty_b * (price_b - pos.price_b_entry)


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
    keep_cols = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL + [
        TARGET_COLUMN, "pair", "pair_type", "exit_reason", "pnl_usd", "holding_bars",
    ]
    keep = [c for c in keep_cols if c in closed.columns]
    dataset = closed[keep].copy()
    dataset.to_csv(DATASET_CSV, index=False)
    return dataset


# ============================================================
# NEW: Pair Stability Tracker
# ============================================================

def update_pair_stability(state: Dict, pair: str, f: PairFeatures) -> bool:
    """
    Perbarui skor stabilitas pair. Return True jika pair dianggap stabil.
    Pair dianggap "memburuk" satu bar jika corr atau ADF melebihi floor.
    Setelah stability_decay_bars bar memburuk berturut-turut, pair ditandai unstable.
    Jika bar berikutnya bagus, reset bad_bar counter.
    """
    entry = state.setdefault("pair_stability", {}).setdefault(pair, {"bad_bars": 0, "stable": True})

    corr_bad = pd.isna(f.corr) or f.corr < CONFIG["stability_corr_floor"]
    adf_bad  = pd.isna(f.adf_p) or f.adf_p > CONFIG["stability_adf_ceiling"]
    bar_bad  = corr_bad or adf_bad

    if bar_bad:
        entry["bad_bars"] += 1
    else:
        entry["bad_bars"] = 0

    entry["stable"] = entry["bad_bars"] < CONFIG["stability_decay_bars"]
    return bool(entry["stable"])


# ============================================================
# NEW: Circuit Breaker
# ============================================================

def check_circuit_breaker(state: Dict) -> bool:
    """Return True jika circuit breaker aktif (entry harus diblokir)."""
    equity = state.get("demo_equity", CONFIG["demo_equity_initial"])
    peak   = state.get("peak_equity", equity)
    if peak <= 0:
        return False
    dd = (peak - equity) / peak
    active = dd >= CONFIG["circuit_breaker_dd"]
    if active and not state.get("circuit_breaker_active", False):
        state["circuit_breaker_active"] = True
        send_telegram(
            f"🔴 CIRCUIT BREAKER AKTIF\n"
            f"Drawdown: {dd*100:.1f}% dari peak ${peak:.0f}\n"
            f"Equity sekarang: ${equity:.0f}\n"
            f"Entry baru diblokir sampai equity recover."
        )
    elif not active and state.get("circuit_breaker_active", False):
        state["circuit_breaker_active"] = False
        send_telegram(f"🟢 Circuit breaker RESET — equity ${equity:.0f} ({dd*100:.1f}% dari peak)")
    return active


# ============================================================
# Main Bot Engine
# ============================================================

class StatArbMLBot:
    def __init__(self):
        self.md    = MarketData(CONFIG["exchange_id"])
        self.pairs = build_pair_universe(self.md)
        self.state = load_state()
        self.ml    = MLFilter()

        # Restore equity dari state (NEW: tidak lagi reset tiap restart)
        self.demo_equity = float(self.state.get("demo_equity", CONFIG["demo_equity_initial"]))

        send_telegram(
            f"STAT ARB ML BOT v3 started\n"
            f"Pairs: {len(self.pairs)} | TF: {CONFIG['timeframe']}\n"
            f"Equity: ${self.demo_equity:.0f} (restored)\n"
            f"UTC: {utc_now_str()}"
        )

    # ----------------------------------------------------------
    # State helpers
    # ----------------------------------------------------------

    def open_positions(self) -> List[OpenPosition]:
        return [OpenPosition(**p) for p in self.state.get("open_positions", [])]

    def persist_open_positions(self, positions: List[OpenPosition]) -> None:
        self.state["open_positions"] = [asdict(p) for p in positions]

    def _sync_equity_to_state(self) -> None:
        """Selalu simpan equity terkini ke state sebelum save."""
        self.state["demo_equity"] = self.demo_equity
        self.state["peak_equity"] = max(
            self.demo_equity,
            float(self.state.get("peak_equity", self.demo_equity)),
        )

    # ----------------------------------------------------------
    # ML Retrain
    # ----------------------------------------------------------

    def maybe_retrain_model(self) -> None:
        if self.state.get("closed_trades_since_retrain", 0) >= CONFIG["ml_retrain_every_closed_trades"]:
            rebuild_dataset_from_closed_trades()
            self.ml.train_if_possible()
            self.state["closed_trades_since_retrain"] = 0
            send_telegram(
                f"ML retrained\n"
                f"Active: {self.ml.meta.get('active')}\n"
                f"AUC RF: {self.ml.meta.get('auc_rf')} | GBM: {self.ml.meta.get('auc_gbm')}\n"
                f"Rows: {self.ml.meta.get('trained_rows')}"
            )

    # ----------------------------------------------------------
    # Data Fetch
    # ----------------------------------------------------------

    def fetch_all_needed(self) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        symbols = sorted(
            {p.symbol_a for p in self.pairs} |
            {p.symbol_b for p in self.pairs} |
            {"BTC/USDT"}
        )
        data: Dict[str, pd.DataFrame] = {}
        for s in symbols:
            try:
                data[s] = self.md.fetch_ohlcv_df(s, CONFIG["timeframe"], CONFIG["history_limit"])
            except Exception as e:
                print(f"[WARN] fetch failed {s}: {e}")
        btc_df = data.get("BTC/USDT")
        if btc_df is None:
            raise RuntimeError("BTC/USDT unavailable — needed for regime")
        return data, btc_df

    # ----------------------------------------------------------
    # Manage Open Positions
    # ----------------------------------------------------------

    def manage_open_positions(
        self, market_data: Dict[str, pd.DataFrame], btc_df: pd.DataFrame
    ) -> None:
        positions = self.open_positions()
        if not positions:
            return
        survivors: List[OpenPosition] = []

        for pos in positions:
            pair_spec = PairSpec(
                symbol_a=pos.symbol_a, symbol_b=pos.symbol_b,
                base_a=pos.base_a, base_b=pos.base_b,
                group_a=pos.group_a, group_b=pos.group_b,
                pair_type=pos.pair_type, sector=pos.sector,
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
            pos.max_adverse_pnl   = min(pos.max_adverse_pnl,   pnl)

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

            pnl_pct    = pnl / max(pos.gross_notional, 1e-9)
            r_multiple = pnl / max(pos.demo_equity_at_entry * pos.risk_fraction_used, 1e-9)
            self.demo_equity += pnl

            # Feature mirror untuk dataset
            feature_mirror = {
                "score": pos.score, "zscore": pos.zscore_entry,
                "corr": pos.corr_entry, "rolling_corr": pos.rolling_corr_entry,
                "adf_p": pos.adf_p_entry, "recent_adf_p": pos.recent_adf_p_entry,
                "half_life": pos.half_life_entry, "hedge_ratio": pos.hedge_ratio_entry,
                "vol_ratio": pos.vol_ratio_entry,
                "spread_percentile": pos.spread_percentile_entry,
                "z_change_1": pos.z_change_1_entry, "z_change_3": pos.z_change_3_entry,
                "ret_a_1": pos.ret_a_1_entry, "ret_b_1": pos.ret_b_1_entry,
                "ret_a_6": pos.ret_a_6_entry, "ret_b_6": pos.ret_b_6_entry,
                "market_regime": pos.market_regime_entry, "session": pos.session_entry,
                "group_a": pos.group_a, "group_b": pos.group_b,
                "pair_type": pos.pair_type,
            }
            closed_row = {
                **asdict(pos),
                "exit_time": f.timestamp,
                "exit_reason": exit_reason,
                "zscore_exit": f.zscore,
                "corr_exit": f.corr, "rolling_corr_exit": f.rolling_corr,
                "adf_p_exit": f.adf_p, "recent_adf_p_exit": f.recent_adf_p,
                "half_life_exit": f.half_life,
                "price_a_exit": f.price_a, "price_b_exit": f.price_b,
                "spread_exit": f.spread,
                "pnl_usd": pnl, "pnl_pct": pnl_pct, "R_multiple": r_multiple,
                "mfe_usd": pos.max_favorable_pnl, "mae_usd": pos.max_adverse_pnl,
                "holding_bars": pos.bars_held,
                "demo_equity_after": self.demo_equity,
                "logged_at": utc_now_str(),
                **feature_mirror,
            }
            buffer_row(CLOSED_TRADES_CSV, closed_row)
            buffer_row(POSITIONS_CSV, {**closed_row, "event": "CLOSE"})

            self.state["closed_trades_since_retrain"] = (
                self.state.get("closed_trades_since_retrain", 0) + 1
            )

            emoji = "✅" if pnl > 0 else "❌"
            msg = (
                f"{emoji} STAT ARB CLOSE\n"
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

    # ----------------------------------------------------------
    # Evaluate New Entries
    # ----------------------------------------------------------

    def evaluate_new_entries(
        self, market_data: Dict[str, pd.DataFrame], btc_df: pd.DataFrame
    ) -> None:
        # NEW: Regime gate
        current_regime = regime_from_btc(btc_df["close"])
        if current_regime in CONFIG["block_entry_regimes"]:
            print(f"[REGIME GATE] Entry blocked — regime={current_regime}")
            return

        # NEW: Circuit breaker
        if check_circuit_breaker(self.state):
            print("[CIRCUIT BREAKER] Entry blocked — drawdown too deep")
            return

        open_positions = self.open_positions()
        if len(open_positions) >= CONFIG["max_open_positions"]:
            return
        open_pairs = {p.pair for p in open_positions}

        # Load closed trades sekarang untuk Kelly sizing
        closed_df: Optional[pd.DataFrame] = None
        if CLOSED_TRADES_CSV.exists():
            try:
                closed_df = pd.read_csv(CLOSED_TRADES_CSV)
            except Exception:
                pass

        candidates: List[Tuple[PairSpec, PairFeatures, Optional[float], bool, float]] = []

        for pair in self.pairs:
            pair_name = f"{pair.base_a}-{pair.base_b}"
            if pair_name in open_pairs:
                continue
            df_a = market_data.get(pair.symbol_a)
            df_b = market_data.get(pair.symbol_b)
            if df_a is None or df_b is None:
                continue
            f = compute_pair_features(df_a, df_b, pair, btc_df)
            if f is None:
                continue

            # NEW: Pair stability check
            stable = update_pair_stability(self.state, pair_name, f)

            ok, reason = passes_base_filters(f)
            signal_row = asdict(f)
            signal_row.update({
                "filter_pass": ok, "filter_reason": reason,
                "pair_stable": stable, "logged_at": utc_now_str(),
            })
            buffer_row(SIGNALS_CSV, signal_row)

            if not ok:
                continue
            if not stable:
                continue

            # Cooldown check (v3: epoch int, bukan bar timestamp string)
            now_epoch = utc_now_epoch()
            last_epoch = self.state.get("last_pair_entry_epoch", {}).get(pair_name, 0)
            bars_since = (now_epoch - last_epoch) / 3600  # asumsi 1h bar
            if bars_since < CONFIG["cooldown_bars_same_pair"]:
                continue

            ml_prob, ml_pass = self.ml.predict(signal_row)
            if not ml_pass:
                continue

            # Kelly fraction untuk pair_type ini
            rf = kelly_fraction(f.pair_type, closed_df)

            candidates.append((pair, f, ml_prob, ml_pass, rf))

        # Sort: ML prob (jika ada) + heuristik score
        candidates.sort(
            key=lambda x: ((x[2] if x[2] is not None else 0.5) * 100 + x[1].score),
            reverse=True,
        )

        for pair, f, ml_prob, ml_pass, risk_fraction in candidates:
            open_positions = self.open_positions()
            if len(open_positions) >= CONFIG["max_open_positions"]:
                break

            gross_notional, notional_a, notional_b, qty_a, qty_b = position_sizing(
                self.demo_equity, risk_fraction, f.hedge_ratio, f.price_a, f.price_b,
            )

            pos = OpenPosition(
                position_id=f"{f.pair}_{int(pd.Timestamp(f.timestamp).timestamp())}",
                entry_time=f.timestamp,
                pair=f.pair, base_a=pair.base_a, base_b=pair.base_b,
                symbol_a=pair.symbol_a, symbol_b=pair.symbol_b,
                pair_type=f.pair_type, sector=f.sector,
                group_a=f.group_a, group_b=f.group_b,
                signal=f.signal, side_text=f.side_text,
                score=f.score,
                zscore_entry=f.zscore,
                corr_entry=f.corr, rolling_corr_entry=f.rolling_corr,
                adf_p_entry=f.adf_p, recent_adf_p_entry=f.recent_adf_p,
                half_life_entry=f.half_life, hedge_ratio_entry=f.hedge_ratio,
                vol_ratio_entry=f.vol_ratio,
                spread_entry=f.spread, spread_mean_entry=f.spread_mean,
                spread_std_entry=f.spread_std,
                spread_percentile_entry=f.spread_percentile,
                z_change_1_entry=f.z_change_1, z_change_3_entry=f.z_change_3,
                ret_a_1_entry=f.ret_a_1, ret_b_1_entry=f.ret_b_1,
                ret_a_6_entry=f.ret_a_6, ret_b_6_entry=f.ret_b_6,
                price_a_entry=f.price_a, price_b_entry=f.price_b,
                market_regime_entry=f.market_regime, session_entry=f.session,
                ml_prob=ml_prob, ml_passed=ml_pass,
                demo_equity_at_entry=self.demo_equity,
                risk_fraction_used=risk_fraction,
                gross_notional=gross_notional,
                notional_a=notional_a, notional_b=notional_b,
                qty_a=qty_a, qty_b=qty_b,
                tp_z_abs=CONFIG["tp_z_abs"],
                sl_z_abs=CONFIG["sl_z_abs"],
                max_holding_bars=CONFIG["max_holding_bars"],
                bars_held=0,
                max_favorable_pnl=0.0, max_adverse_pnl=0.0,
            )
            open_positions.append(pos)
            self.persist_open_positions(open_positions)
            self.state.setdefault("last_pair_entry_epoch", {})[f.pair] = utc_now_epoch()

            pos_row = asdict(pos)
            pos_row.update({"event": "OPEN", "logged_at": utc_now_str()})
            buffer_row(POSITIONS_CSV, pos_row)

            msg = (
                f"🚨 STAT ARB OPEN (v3)\n"
                f"Pair: {f.pair} | Type: {f.pair_type}\n"
                f"Sector: {f.sector}\n"
                f"Signal: {f.side_text}\n"
                f"Score: {f.score} | Z: {f.zscore:.3f}\n"
                f"Corr: {f.corr:.3f} | RollCorr: {f.rolling_corr:.3f}\n"
                f"ADF p: {f.adf_p:.4f} | Recent: {f.recent_adf_p:.4f}\n"
                f"HL: {f.half_life:.2f} | HR: {f.hedge_ratio:.4f}\n"
                f"Notional A: ${notional_a:.0f} | B: ${notional_b:.0f}\n"
                f"Risk fraction (Kelly): {risk_fraction*100:.1f}%\n"
                f"ML Prob: {'n/a' if ml_prob is None else round(ml_prob, 3)}\n"
                f"Regime: {f.market_regime}\n"
                f"TP |Z| < {CONFIG['tp_z_abs']} | SL |Z| > {CONFIG['sl_z_abs']}"
            )
            print(msg)
            send_telegram(msg)

    # ----------------------------------------------------------
    # Run Loop
    # ----------------------------------------------------------

    def run_cycle(self) -> None:
        market_data, btc_df = self.fetch_all_needed()
        self.manage_open_positions(market_data, btc_df)
        self.evaluate_new_entries(market_data, btc_df)
        self.maybe_retrain_model()
        # NEW: sync equity & flush CSV buffer di akhir setiap cycle
        self._sync_equity_to_state()
        save_state(self.state)
        flush_csv_buffer()

    def run_forever(self) -> None:
        print(f"[START v3] UTC {utc_now_str()} | pairs={len(self.pairs)} | equity=${self.demo_equity:.0f}")
        last_processed_bar: Optional[pd.Timestamp] = None

        while True:
            try:
                probe = self.md.fetch_ohlcv_df("BTC/USDT", CONFIG["timeframe"], 2)
                current_bar = pd.Timestamp(probe["timestamp"].iloc[-1])

                if last_processed_bar is None or current_bar > last_processed_bar:
                    last_processed_bar = current_bar
                    self.run_cycle()
                    print(
                        f"[CYCLE] bar={current_bar} | "
                        f"equity=${self.demo_equity:.2f} | "
                        f"open={len(self.open_positions())} | "
                        f"ml={self.ml.meta.get('active', False)}"
                    )
                else:
                    print(
                        f"[WAIT] bar={current_bar} | "
                        f"equity=${self.demo_equity:.2f} | "
                        f"open={len(self.open_positions())}"
                    )

            except KeyboardInterrupt:
                print("Stopped by user")
                flush_csv_buffer()
                self._sync_equity_to_state()
                save_state(self.state)
                break
            except Exception as e:
                err = f"[ERROR] {utc_now_str()}\n{type(e).__name__}: {e}\n{traceback.format_exc()}"
                print(err)
                send_telegram(err[:3500])
                # Coba flush buffer agar data tidak hilang saat error
                try:
                    flush_csv_buffer()
                    self._sync_equity_to_state()
                    save_state(self.state)
                except Exception:
                    pass

            time.sleep(CONFIG["poll_seconds"])


# ============================================================
# Utilities
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
    pf = pf_num / pf_den if pf_den > 0 else float("nan")

    lines = [
        "STAT ARB BOT v3 SUMMARY",
        f"Closed trades : {len(df)}",
        f"Win rate      : {wr:.2f}%",
        f"Avg PnL       : ${avg_pnl:.2f}",
        f"Profit factor : {pf:.2f}" if not math.isnan(pf) else "Profit factor : n/a",
    ]

    if "pair_type" in df.columns:
        lines.append("\nPnL by pair type:")
        lines.append(
            df.groupby("pair_type")["pnl_usd"]
            .agg(["count", "mean", "sum"])
            .sort_values("sum", ascending=False)
            .to_string()
        )

    if "exit_reason" in df.columns:
        lines.append("\nExit reasons:")
        lines.append(df["exit_reason"].value_counts().to_string())

    return "\n".join(lines)


def run_once_for_test() -> None:
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
