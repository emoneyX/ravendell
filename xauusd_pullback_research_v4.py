import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from itertools import product
from typing import Optional, List, Dict, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# XAUUSD PULLBACK RESEARCH HARNESS v4
# Purpose:
#   - Freeze the best v3 candidate as baseline
#   - Expand around it CAREFULLY, not blindly
#   - Validate robustness with walk-forward tests
#   - Test local parameter stability (plateau analysis)
#   - Add more realistic friction / commission handling
#   - Save richer artifacts for decision-making
#
# Core thesis:
#   Gold intraday continuation can be traded via selective pullback
#   entries during the London/NY overlap, with strict trade frequency
#   control and risk management.
# ============================================================

CFG = {
    "symbol_candidates": [
        "XAUUSD", "XAUUSD.", "XAUUSDm", "XAUUSDmicro", "XAUUSD.a", "XAUUSD.r",
        "GOLD", "GOLD.", "GOLDm", "XAUUSD#", "XAUUSDpro"
    ],
    "htf": mt5.TIMEFRAME_H1,
    "ltf": mt5.TIMEFRAME_M15,
    "start": datetime(2022, 1, 1),
    "end": datetime(2026, 4, 3),
    "ema_fast": 50,
    "ema_slow": 200,
    "adx_period": 14,
    "adx_threshold": 20.0,
    "atr_period": 14,
    "atr_vol_lookback": 50,
    "rsi_period": 14,
    "initial_balance": 10000.0,
    "risk_per_trade": 0.01,
    "commission_per_lot": 7.0,   # round-turn placeholder; adjust to broker reality
    "contract_size": 100.0,
    "min_lot": 0.01,
    "lot_step": 0.01,
    "max_lot": 100.0,
    "base_spread_points": 30,
    "base_slippage_points": 10,
    "out_dir": "xauusd_pullback_v4_output",
}

# Frozen best v3 candidate = Candidate A
BASELINE_PARAMS = {
    "pullback_ema": 34,
    "sl_atr_mult": 1.2,
    "tp1_r_multiple": 1.5,
    "rsi_long_min": 48,
    "rsi_long_max": 65,
    "rsi_short_min": 35,
    "rsi_short_max": 53,
    "atr_med_mult": 0.8,
    "max_trades_per_day": 1,
    "session_name": "london_ny_overlap",
    "side_mode": "both",
    "trail_lookback": 6,
    "move_to_be_at_1r": True,
}

# Deliberately narrow neighborhood around Candidate A.
# This is how you test for robustness without contaminating the process.
LOCAL_SEARCH_SPACE = {
    "pullback_ema": [20, 34, 50],
    "sl_atr_mult": [1.0, 1.2, 1.4],
    "tp1_r_multiple": [1.25, 1.5, 1.75],
    "rsi_long_min": [46, 48, 50],
    "rsi_long_max": [62, 65, 68],
    "rsi_short_min": [33, 35, 38],
    "rsi_short_max": [50, 53, 56],
    "atr_med_mult": [0.7, 0.8, 0.9],
    "max_trades_per_day": [1, 2],
    "session_name": ["london_ny_overlap", "london_plus_us"],
    "side_mode": ["both", "long_only"],
    "trail_lookback": [4, 6, 8],
    "move_to_be_at_1r": [True],
}

SESSION_MAP = {
    "london_ny_overlap": (14, 21),
    "us_only": (19, 24),
    "london_plus_us": (13, 22),
}

FRICTION_SCENARIOS = {
    "base": {"spread_points": 30, "slippage_points": 10},
    "stress": {"spread_points": 45, "slippage_points": 20},
    "panic": {"spread_points": 60, "slippage_points": 30},
}

# Multiple walk-forward segments. More honest than one split.
WALK_FORWARD_WINDOWS = [
    {
        "name": "wf_2024",
        "train_start": datetime(2022, 1, 1),
        "train_end": datetime(2023, 12, 31, 23, 59, 59),
        "test_start": datetime(2024, 1, 1),
        "test_end": datetime(2024, 12, 31, 23, 59, 59),
    },
    {
        "name": "wf_2025",
        "train_start": datetime(2022, 1, 1),
        "train_end": datetime(2024, 12, 31, 23, 59, 59),
        "test_start": datetime(2025, 1, 1),
        "test_end": datetime(2025, 12, 31, 23, 59, 59),
    },
    {
        "name": "wf_2026_ytd",
        "train_start": datetime(2023, 1, 1),
        "train_end": datetime(2025, 12, 31, 23, 59, 59),
        "test_start": datetime(2026, 1, 1),
        "test_end": datetime(2026, 4, 3, 23, 59, 59),
    },
]


@dataclass
class Position:
    direction: str
    entry_time: pd.Timestamp
    entry_price: float
    stop_price: float
    initial_stop: float
    qty_total: float
    qty_open: float
    tp1_price: float
    tp1_done: bool = False
    break_even_done: bool = False
    realized_pnl: float = 0.0
    risk_amount_at_entry: float = 0.0


def initialize_mt5() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")


def shutdown_mt5() -> None:
    mt5.shutdown()


def timeframe_name(tf: int) -> str:
    names = {
        mt5.TIMEFRAME_M1: "M1",
        mt5.TIMEFRAME_M5: "M5",
        mt5.TIMEFRAME_M15: "M15",
        mt5.TIMEFRAME_M30: "M30",
        mt5.TIMEFRAME_H1: "H1",
        mt5.TIMEFRAME_H4: "H4",
        mt5.TIMEFRAME_D1: "D1",
    }
    return names.get(tf, str(tf))


def get_visible_symbols() -> List[str]:
    symbols = mt5.symbols_get()
    if symbols is None:
        return []
    return [s.name for s in symbols]


def ensure_symbol_selected(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"symbol_info() failed for {symbol}")
    if not info.visible:
        ok = mt5.symbol_select(symbol, True)
        if not ok:
            raise RuntimeError(f"Could not make symbol visible: {symbol}")


def resolve_gold_symbol(candidates: List[str]) -> str:
    visible = get_visible_symbols()
    visible_set = set(visible)

    for c in candidates:
        if c in visible_set:
            ensure_symbol_selected(c)
            return c

    fuzzy = [s for s in visible if ("XAU" in s.upper() and "USD" in s.upper()) or ("GOLD" in s.upper())]
    if fuzzy:
        fuzzy = sorted(fuzzy, key=lambda x: (len(x), x))
        ensure_symbol_selected(fuzzy[0])
        return fuzzy[0]

    all_symbols = mt5.symbols_get()
    if all_symbols:
        all_names = [s.name for s in all_symbols]
        fuzzy_all = [s for s in all_names if ("XAU" in s.upper() and "USD" in s.upper()) or ("GOLD" in s.upper())]
        if fuzzy_all:
            fuzzy_all = sorted(fuzzy_all, key=lambda x: (len(x), x))
            ensure_symbol_selected(fuzzy_all[0])
            return fuzzy_all[0]

    raise RuntimeError("No gold symbol found in MT5. Open Market Watch and enable the gold symbol.")


def fetch_mt5_direct(symbol: str, timeframe: int, start: datetime, end: datetime) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_range(symbol, timeframe, start, end)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    return df[["time", "open", "high", "low", "close", "volume"]].copy()


def fetch_mt5_chunked(symbol: str, timeframe: int, start: datetime, end: datetime, chunk_days: int = 90) -> Optional[pd.DataFrame]:
    chunks = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=chunk_days), end)
        rates = mt5.copy_rates_range(symbol, timeframe, cur, nxt)
        if rates is not None and len(rates) > 0:
            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s")
            df = df.rename(columns={"tick_volume": "volume"})
            chunks.append(df[["time", "open", "high", "low", "close", "volume"]].copy())
        cur = nxt

    if not chunks:
        return None

    out = pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return out


def fetch_mt5_from_pos(symbol: str, timeframe: int, bars: int = 180000) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    df = df[["time", "open", "high", "low", "close", "volume"]].copy()
    df = df.sort_values("time").reset_index(drop=True)
    return df


def fetch_mt5(symbol: str, timeframe: int, start: datetime, end: datetime) -> pd.DataFrame:
    ensure_symbol_selected(symbol)

    direct = fetch_mt5_direct(symbol, timeframe, start, end)
    if direct is not None and not direct.empty:
        return direct

    print(f"[WARN] direct range fetch empty for {symbol} {timeframe_name(timeframe)}. Trying chunked download...")
    chunk_days = 90 if timeframe in [mt5.TIMEFRAME_M1, mt5.TIMEFRAME_M5, mt5.TIMEFRAME_M15] else 365
    chunked = fetch_mt5_chunked(symbol, timeframe, start, end, chunk_days=chunk_days)
    if chunked is not None and not chunked.empty:
        return chunked

    print(f"[WARN] chunked fetch empty for {symbol} {timeframe_name(timeframe)}. Trying copy_rates_from_pos fallback...")
    by_pos = fetch_mt5_from_pos(symbol, timeframe, bars=180000)
    if by_pos is not None and not by_pos.empty:
        by_pos = by_pos[(by_pos["time"] >= pd.Timestamp(start)) & (by_pos["time"] <= pd.Timestamp(end))].copy()
        if not by_pos.empty:
            return by_pos

    raise RuntimeError(
        f"No data returned for {symbol} {timeframe_name(timeframe)}.\n"
        f"Fix MT5 first:\n"
        f"1. Open chart for {symbol} on {timeframe_name(timeframe)}\n"
        f"2. Press Home several times to load history\n"
        f"3. Ensure symbol is enabled in Market Watch\n"
        f"4. Re-run script"
    )


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)

    atr_sm = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_sm)
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr_sm)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def build_htf_features(htf: pd.DataFrame) -> pd.DataFrame:
    h = htf.copy()
    h["ema_fast"] = ema(h["close"], CFG["ema_fast"])
    h["ema_slow"] = ema(h["close"], CFG["ema_slow"])
    h["adx"] = adx(h, CFG["adx_period"])
    h["bull_trend"] = (
        (h["ema_fast"] > h["ema_slow"]) &
        (h["ema_fast"].diff() > 0) &
        (h["adx"] > CFG["adx_threshold"])
    )
    h["bear_trend"] = (
        (h["ema_fast"] < h["ema_slow"]) &
        (h["ema_fast"].diff() < 0) &
        (h["adx"] > CFG["adx_threshold"])
    )
    return h


def build_ltf_features(ltf: pd.DataFrame) -> pd.DataFrame:
    d = ltf.copy()
    d["atr"] = atr(d, CFG["atr_period"])
    d["atr_med"] = d["atr"].rolling(CFG["atr_vol_lookback"]).median()
    d["rsi"] = rsi(d["close"], CFG["rsi_period"])
    d["ema20"] = ema(d["close"], 20)
    d["ema34"] = ema(d["close"], 34)
    d["ema50"] = ema(d["close"], 50)
    d["prev_high"] = d["high"].shift(1)
    d["prev_low"] = d["low"].shift(1)
    d["dow"] = d["time"].dt.dayofweek
    d["hour"] = d["time"].dt.hour
    return d


def merge_htf_to_ltf(ltf: pd.DataFrame, htf: pd.DataFrame) -> pd.DataFrame:
    hsmall = htf[["time", "bull_trend", "bear_trend", "ema_fast", "ema_slow", "adx"]].copy()
    merged = pd.merge_asof(
        ltf.sort_values("time"),
        hsmall.sort_values("time"),
        on="time",
        direction="backward"
    )
    return merged


def round_lot(lot: float, min_lot: float, lot_step: float, max_lot: float) -> float:
    if lot <= 0:
        return 0.0
    steps = math.floor((lot - min_lot) / lot_step + 1e-9)
    rounded = min_lot + max(0, steps) * lot_step
    rounded = max(min_lot, min(rounded, max_lot))
    return round(rounded, 2)


def price_to_money_move(price_diff: float, lot: float, contract_size: float) -> float:
    return price_diff * contract_size * lot


def get_symbol_point(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    return info.point if info is not None else 0.01


def apply_session_filter(df: pd.DataFrame, session_name: str) -> pd.DataFrame:
    start_hour, end_hour = SESSION_MAP[session_name]
    out = df.copy()
    if end_hour == 24:
        out["in_session"] = out["time"].dt.hour >= start_hour
    else:
        out["in_session"] = out["time"].dt.hour.between(start_hour, end_hour - 1)
    return out


def add_param_columns(df: pd.DataFrame, params: Dict) -> pd.DataFrame:
    out = df.copy()
    pullback_ema = params["pullback_ema"]
    out["ema_pullback"] = out[f"ema{pullback_ema}"]

    out["dist_from_htf_fast"] = (out["close"] - out["ema_fast"]).abs()
    out["not_overextended"] = out["dist_from_htf_fast"] <= (2.0 * out["atr"])
    return out


def get_pullback_signal(row: pd.Series, params: Dict) -> Tuple[bool, bool]:
    side_mode = params["side_mode"]

    long_allowed = side_mode in ["long_only", "both"]
    short_allowed = side_mode in ["short_only", "both"]

    long_signal = False
    short_signal = False

    if long_allowed:
        long_signal = (
            bool(row["bull_trend"]) and
            bool(row["not_overextended"]) and
            (row["low"] <= row["ema_pullback"]) and
            (row["close"] > row["open"]) and
            (row["close"] > row["prev_high"]) and
            (row["rsi"] >= params["rsi_long_min"]) and
            (row["rsi"] <= params["rsi_long_max"]) and
            (row["atr"] > row["atr_med"] * params["atr_med_mult"])
        )

    if short_allowed:
        short_signal = (
            bool(row["bear_trend"]) and
            bool(row["not_overextended"]) and
            (row["high"] >= row["ema_pullback"]) and
            (row["close"] < row["open"]) and
            (row["close"] < row["prev_low"]) and
            (row["rsi"] >= params["rsi_short_min"]) and
            (row["rsi"] <= params["rsi_short_max"]) and
            (row["atr"] > row["atr_med"] * params["atr_med_mult"])
        )

    return long_signal, short_signal


def run_backtest(data: pd.DataFrame, params: Dict, friction: Dict, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    point = get_symbol_point(symbol)
    spread_price = friction["spread_points"] * point
    slippage_price = friction["slippage_points"] * point

    cash = CFG["initial_balance"]
    equity = cash
    position: Optional[Position] = None

    trades: List[Dict] = []
    equity_curve: List[Dict] = []
    trades_today: Dict[pd.Timestamp, int] = {}

    warmup = max(CFG["ema_slow"], CFG["atr_vol_lookback"], params["pullback_ema"]) + 10

    for i in range(warmup, len(data) - 1):
        row = data.iloc[i]
        nxt = data.iloc[i + 1]
        day_key = row["time"].normalize()
        trades_today.setdefault(day_key, 0)

        if position is not None:
            one_r = abs(position.entry_price - position.initial_stop)

            if params["move_to_be_at_1r"] and not position.break_even_done:
                if position.direction == "long" and row["high"] >= position.entry_price + one_r:
                    position.stop_price = max(position.stop_price, position.entry_price)
                    position.break_even_done = True
                elif position.direction == "short" and row["low"] <= position.entry_price - one_r:
                    position.stop_price = min(position.stop_price, position.entry_price)
                    position.break_even_done = True

            if not position.tp1_done:
                if position.direction == "long" and row["high"] >= position.tp1_price:
                    qty_to_close = position.qty_total * 0.5
                    fill = position.tp1_price - (spread_price / 2) - slippage_price
                    pnl = price_to_money_move(fill - position.entry_price, qty_to_close, CFG["contract_size"])
                    pnl -= CFG["commission_per_lot"] * qty_to_close
                    cash += pnl
                    position.realized_pnl += pnl
                    position.qty_open -= qty_to_close
                    position.tp1_done = True

                elif position.direction == "short" and row["low"] <= position.tp1_price:
                    qty_to_close = position.qty_total * 0.5
                    fill = position.tp1_price + (spread_price / 2) + slippage_price
                    pnl = price_to_money_move(position.entry_price - fill, qty_to_close, CFG["contract_size"])
                    pnl -= CFG["commission_per_lot"] * qty_to_close
                    cash += pnl
                    position.realized_pnl += pnl
                    position.qty_open -= qty_to_close
                    position.tp1_done = True

            if position.tp1_done:
                lb = params["trail_lookback"]
                if i - lb >= 0:
                    if position.direction == "long":
                        new_stop = data.iloc[i - lb + 1:i + 1]["low"].min()
                        position.stop_price = max(position.stop_price, new_stop)
                    else:
                        new_stop = data.iloc[i - lb + 1:i + 1]["high"].max()
                        position.stop_price = min(position.stop_price, new_stop)

            stop_hit = False
            stop_fill = None
            if position.direction == "long" and row["low"] <= position.stop_price:
                stop_hit = True
                stop_fill = position.stop_price - (spread_price / 2) - slippage_price
            elif position.direction == "short" and row["high"] >= position.stop_price:
                stop_hit = True
                stop_fill = position.stop_price + (spread_price / 2) + slippage_price

            if stop_hit:
                pnl = 0.0
                if position.qty_open > 0:
                    if position.direction == "long":
                        pnl = price_to_money_move(stop_fill - position.entry_price, position.qty_open, CFG["contract_size"])
                    else:
                        pnl = price_to_money_move(position.entry_price - stop_fill, position.qty_open, CFG["contract_size"])
                    pnl -= CFG["commission_per_lot"] * position.qty_open
                    cash += pnl
                    position.realized_pnl += pnl

                trades.append({
                    "entry_time": position.entry_time,
                    "exit_time": row["time"],
                    "direction": position.direction,
                    "entry_price": position.entry_price,
                    "exit_price": stop_fill,
                    "qty": position.qty_total,
                    "pnl": position.realized_pnl,
                    "r_multiple": position.realized_pnl / position.risk_amount_at_entry if position.risk_amount_at_entry > 0 else np.nan,
                    "exit_reason": "stop_or_trail",
                    "tp1_done": position.tp1_done,
                    "break_even_done": position.break_even_done,
                    "entry_dow": position.entry_time.dayofweek,
                    "entry_hour": position.entry_time.hour,
                })
                position = None

        unreal = 0.0
        if position is not None and position.qty_open > 0:
            if position.direction == "long":
                unreal = price_to_money_move(row["close"] - position.entry_price, position.qty_open, CFG["contract_size"])
            else:
                unreal = price_to_money_move(position.entry_price - row["close"], position.qty_open, CFG["contract_size"])

        equity = cash + unreal
        equity_curve.append({"time": row["time"], "equity": equity})

        if position is not None:
            continue
        if not row["in_session"]:
            continue
        if trades_today[day_key] >= params["max_trades_per_day"]:
            continue
        if pd.isna(row["atr"]) or pd.isna(row["atr_med"]) or row["atr"] <= 0:
            continue
        if pd.isna(row["ema_pullback"]) or pd.isna(row["prev_high"]) or pd.isna(row["prev_low"]) or pd.isna(row["rsi"]):
            continue

        long_signal, short_signal = get_pullback_signal(row, params)
        if not long_signal and not short_signal:
            continue

        risk_amount = equity * CFG["risk_per_trade"]

        if long_signal:
            raw_entry = nxt["open"] + (spread_price / 2) + slippage_price
            stop_price = raw_entry - params["sl_atr_mult"] * row["atr"]
            stop_dist = raw_entry - stop_price
            if stop_dist <= 0:
                continue
            lot = risk_amount / (stop_dist * CFG["contract_size"])
            lot = round_lot(lot, CFG["min_lot"], CFG["lot_step"], CFG["max_lot"])
            if lot <= 0:
                continue
            tp1 = raw_entry + params["tp1_r_multiple"] * stop_dist
            entry_commission = CFG["commission_per_lot"] * 0.0  # commission realized on exits to avoid double-counting here
            cash -= entry_commission
            position = Position(
                direction="long",
                entry_time=nxt["time"],
                entry_price=raw_entry,
                stop_price=stop_price,
                initial_stop=stop_price,
                qty_total=lot,
                qty_open=lot,
                tp1_price=tp1,
                risk_amount_at_entry=stop_dist * CFG["contract_size"] * lot,
            )
            trades_today[day_key] += 1

        elif short_signal:
            raw_entry = nxt["open"] - (spread_price / 2) - slippage_price
            stop_price = raw_entry + params["sl_atr_mult"] * row["atr"]
            stop_dist = stop_price - raw_entry
            if stop_dist <= 0:
                continue
            lot = risk_amount / (stop_dist * CFG["contract_size"])
            lot = round_lot(lot, CFG["min_lot"], CFG["lot_step"], CFG["max_lot"])
            if lot <= 0:
                continue
            tp1 = raw_entry - params["tp1_r_multiple"] * stop_dist
            entry_commission = CFG["commission_per_lot"] * 0.0
            cash -= entry_commission
            position = Position(
                direction="short",
                entry_time=nxt["time"],
                entry_price=raw_entry,
                stop_price=stop_price,
                initial_stop=stop_price,
                qty_total=lot,
                qty_open=lot,
                tp1_price=tp1,
                risk_amount_at_entry=stop_dist * CFG["contract_size"] * lot,
            )
            trades_today[day_key] += 1

    if position is not None:
        last = data.iloc[-1]
        if position.direction == "long":
            final_fill = last["close"] - (spread_price / 2) - slippage_price
            pnl = price_to_money_move(final_fill - position.entry_price, position.qty_open, CFG["contract_size"])
        else:
            final_fill = last["close"] + (spread_price / 2) + slippage_price
            pnl = price_to_money_move(position.entry_price - final_fill, position.qty_open, CFG["contract_size"])
        pnl -= CFG["commission_per_lot"] * position.qty_open
        cash += pnl
        position.realized_pnl += pnl

        trades.append({
            "entry_time": position.entry_time,
            "exit_time": last["time"],
            "direction": position.direction,
            "entry_price": position.entry_price,
            "exit_price": final_fill,
            "qty": position.qty_total,
            "pnl": position.realized_pnl,
            "r_multiple": position.realized_pnl / position.risk_amount_at_entry if position.risk_amount_at_entry > 0 else np.nan,
            "exit_reason": "final_close",
            "tp1_done": position.tp1_done,
            "break_even_done": position.break_even_done,
            "entry_dow": position.entry_time.dayofweek,
            "entry_hour": position.entry_time.hour,
        })
        equity_curve.append({"time": last["time"], "equity": cash})

    return pd.DataFrame(trades), pd.DataFrame(equity_curve)


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def summarize_results(trades: pd.DataFrame, eq: pd.DataFrame, initial_balance: float) -> Dict:
    if trades.empty:
        return {
            "total_trades": 0,
            "win_rate": 0.0,
            "net_profit": 0.0,
            "profit_factor": 0.0,
            "avg_trade": 0.0,
            "avg_r": 0.0,
            "max_drawdown_pct": 0.0,
            "final_equity": initial_balance,
            "long_trades": 0,
            "short_trades": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "expectancy_r": 0.0,
        }

    total_trades = len(trades)
    wins = int((trades["pnl"] > 0).sum())
    win_rate = wins / total_trades * 100
    gross_profit = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    net_profit = trades["pnl"].sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    avg_trade = trades["pnl"].mean()
    avg_r = trades["r_multiple"].mean()
    final_equity = eq["equity"].iloc[-1] if not eq.empty else initial_balance + net_profit
    dd = max_drawdown(eq["equity"]) * 100 if not eq.empty else 0.0

    return {
        "total_trades": total_trades,
        "win_rate": win_rate,
        "net_profit": net_profit,
        "profit_factor": profit_factor,
        "avg_trade": avg_trade,
        "avg_r": avg_r,
        "max_drawdown_pct": dd,
        "final_equity": final_equity,
        "long_trades": int((trades["direction"] == "long").sum()),
        "short_trades": int((trades["direction"] == "short").sum()),
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "expectancy_r": avg_r,
    }


def monthly_returns(eq: pd.DataFrame) -> pd.DataFrame:
    if eq.empty:
        return pd.DataFrame(columns=["month", "return_pct"])
    m = eq.copy()
    m["time"] = pd.to_datetime(m["time"])
    month_end = m.set_index("time")["equity"].resample("ME").last().dropna()
    ret = month_end.pct_change().fillna(0.0) * 100
    return pd.DataFrame({"month": month_end.index.astype(str), "return_pct": ret.values})


def combo_iter(search_space: Dict) -> List[Dict]:
    keys = list(search_space.keys())
    vals = [search_space[k] for k in keys]
    combos = []
    for items in product(*vals):
        d = dict(zip(keys, items))
        if d["rsi_long_min"] >= d["rsi_long_max"]:
            continue
        if d["rsi_short_min"] >= d["rsi_short_max"]:
            continue
        combos.append(d)
    return combos


def score_run(train_sum: Dict, test_base: Dict, test_stress: Dict, test_panic: Dict) -> float:
    if train_sum["total_trades"] < 80:
        return -1e9
    if test_base["total_trades"] < 30:
        return -1e9
    if test_base["profit_factor"] <= 1.0:
        return -1e9
    if test_stress["profit_factor"] <= 0.95:
        return -1e9

    score = 0.0
    score += test_base["net_profit"] * 0.0035
    score += test_base["profit_factor"] * 110
    score += test_stress["profit_factor"] * 70
    score += test_panic["profit_factor"] * 30
    score += train_sum["profit_factor"] * 15
    score -= abs(test_base["max_drawdown_pct"]) * 4.0
    score -= abs(test_stress["max_drawdown_pct"]) * 5.0
    score -= abs(test_panic["max_drawdown_pct"]) * 6.0
    score -= abs(train_sum["max_drawdown_pct"]) * 1.5
    score += min(test_base["total_trades"], 120) * 0.45
    return score


def subset_by_time(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    return df[(df["time"] >= pd.Timestamp(start)) & (df["time"] <= pd.Timestamp(end))].copy().reset_index(drop=True)


def evaluate_config_on_window(base_data: pd.DataFrame, params: Dict, symbol: str, train_start: datetime, train_end: datetime,
                              test_start: datetime, test_end: datetime) -> Dict:
    train = subset_by_time(base_data, train_start, train_end)
    test = subset_by_time(base_data, test_start, test_end)

    train = apply_session_filter(add_param_columns(train, params), params["session_name"])
    test = apply_session_filter(add_param_columns(test, params), params["session_name"])

    train_trades, train_eq = run_backtest(train, params, FRICTION_SCENARIOS["base"], symbol)
    test_trades_base, test_eq_base = run_backtest(test, params, FRICTION_SCENARIOS["base"], symbol)
    test_trades_stress, test_eq_stress = run_backtest(test, params, FRICTION_SCENARIOS["stress"], symbol)
    test_trades_panic, test_eq_panic = run_backtest(test, params, FRICTION_SCENARIOS["panic"], symbol)

    train_sum = summarize_results(train_trades, train_eq, CFG["initial_balance"])
    test_sum_base = summarize_results(test_trades_base, test_eq_base, CFG["initial_balance"])
    test_sum_stress = summarize_results(test_trades_stress, test_eq_stress, CFG["initial_balance"])
    test_sum_panic = summarize_results(test_trades_panic, test_eq_panic, CFG["initial_balance"])

    score = score_run(train_sum, test_sum_base, test_sum_stress, test_sum_panic)

    return {
        "train_sum": train_sum,
        "test_sum_base": test_sum_base,
        "test_sum_stress": test_sum_stress,
        "test_sum_panic": test_sum_panic,
        "train_trades": train_trades,
        "test_trades_base": test_trades_base,
        "test_eq_base": test_eq_base,
        "test_eq_stress": test_eq_stress,
        "test_eq_panic": test_eq_panic,
        "score": score,
    }


def aggregate_walk_forward_rows(rows: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def summarize_walk_forward(wf_df: pd.DataFrame) -> Dict:
    if wf_df.empty:
        return {
            "wf_windows": 0,
            "wf_positive_base": 0,
            "wf_median_pf": 0.0,
            "wf_median_dd": 0.0,
            "wf_total_net": 0.0,
        }
    return {
        "wf_windows": int(len(wf_df)),
        "wf_positive_base": int((wf_df["test_base_pf"] > 1.0).sum()),
        "wf_median_pf": float(wf_df["test_base_pf"].median()),
        "wf_median_dd": float(wf_df["test_base_dd"].median()),
        "wf_total_net": float(wf_df["test_base_net"].sum()),
    }


def build_walk_forward_detail(base_data: pd.DataFrame, params: Dict, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    eq_rows = []

    for window in WALK_FORWARD_WINDOWS:
        res = evaluate_config_on_window(
            base_data=base_data,
            params=params,
            symbol=symbol,
            train_start=window["train_start"],
            train_end=window["train_end"],
            test_start=window["test_start"],
            test_end=window["test_end"],
        )
        rows.append({
            "window": window["name"],
            "train_start": window["train_start"],
            "train_end": window["train_end"],
            "test_start": window["test_start"],
            "test_end": window["test_end"],
            "train_pf": res["train_sum"]["profit_factor"],
            "train_net": res["train_sum"]["net_profit"],
            "train_dd": res["train_sum"]["max_drawdown_pct"],
            "test_base_pf": res["test_sum_base"]["profit_factor"],
            "test_base_net": res["test_sum_base"]["net_profit"],
            "test_base_dd": res["test_sum_base"]["max_drawdown_pct"],
            "test_stress_pf": res["test_sum_stress"]["profit_factor"],
            "test_stress_net": res["test_sum_stress"]["net_profit"],
            "test_stress_dd": res["test_sum_stress"]["max_drawdown_pct"],
            "test_panic_pf": res["test_sum_panic"]["profit_factor"],
            "test_panic_net": res["test_sum_panic"]["net_profit"],
            "test_panic_dd": res["test_sum_panic"]["max_drawdown_pct"],
            "score": res["score"],
        })

        if not res["test_eq_base"].empty:
            eq = res["test_eq_base"].copy()
            eq["window"] = window["name"]
            eq_rows.append(eq)

    wf_df = pd.DataFrame(rows)
    wf_eq_df = pd.concat(eq_rows, ignore_index=True) if eq_rows else pd.DataFrame(columns=["time", "equity", "window"])
    return wf_df, wf_eq_df


def analyze_trade_distribution(trades: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return (
            pd.DataFrame(columns=["entry_dow", "trades", "net_profit", "avg_r"]),
            pd.DataFrame(columns=["entry_hour", "trades", "net_profit", "avg_r"]),
        )

    dow = trades.groupby("entry_dow").agg(
        trades=("pnl", "size"),
        net_profit=("pnl", "sum"),
        avg_r=("r_multiple", "mean"),
    ).reset_index()

    hour = trades.groupby("entry_hour").agg(
        trades=("pnl", "size"),
        net_profit=("pnl", "sum"),
        avg_r=("r_multiple", "mean"),
    ).reset_index()

    return dow, hour


def consecutive_loss_streak(trades: pd.DataFrame) -> int:
    if trades.empty:
        return 0
    max_streak = 0
    cur = 0
    for pnl in trades["pnl"].tolist():
        if pnl < 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return max_streak


def save_equity_plot(eq_dict: Dict[str, pd.DataFrame], out_path: Path, title: str) -> None:
    plt.figure(figsize=(12, 6))
    for label, eq in eq_dict.items():
        if eq is not None and not eq.empty:
            plt.plot(pd.to_datetime(eq["time"]), eq["equity"], label=label)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def save_bar_plot(df: pd.DataFrame, xcol: str, ycol: str, out_path: Path, title: str) -> None:
    if df.empty:
        return
    plt.figure(figsize=(10, 5))
    plt.bar(df[xcol].astype(str), df[ycol])
    plt.title(title)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def neighborhood_distance(params: Dict, baseline: Dict) -> int:
    dist = 0
    for k, v in baseline.items():
        if params.get(k) != v:
            dist += 1
    return dist


def write_summary(symbol: str, baseline_eval: Dict, wf_summary: Dict, best_row: pd.Series, plateau_df: pd.DataFrame, out_path: Path) -> None:
    lines = []
    lines.append("=" * 78)
    lines.append("XAUUSD PULLBACK RESEARCH HARNESS v4")
    lines.append("=" * 78)
    lines.append(f"Resolved symbol              : {symbol}")
    lines.append(f"HTF / LTF                    : {timeframe_name(CFG['htf'])} / {timeframe_name(CFG['ltf'])}")
    lines.append(f"Period                       : {CFG['start']} -> {CFG['end']}")
    lines.append(f"Commission per lot           : {CFG['commission_per_lot']}")
    lines.append("")
    lines.append("[BASELINE CANDIDATE A]")
    for k, v in BASELINE_PARAMS.items():
        lines.append(f"{k:28}: {v}")
    lines.append("")
    lines.append("[FULL-SAMPLE BASELINE PERFORMANCE]")
    full = baseline_eval["base_sum"]
    stress = baseline_eval["stress_sum"]
    panic = baseline_eval["panic_sum"]
    lines.append(f"base_pf                      : {full['profit_factor']:.4f}")
    lines.append(f"base_net                     : {full['net_profit']:.2f}")
    lines.append(f"base_dd                      : {full['max_drawdown_pct']:.2f}%")
    lines.append(f"stress_pf                    : {stress['profit_factor']:.4f}")
    lines.append(f"stress_net                   : {stress['net_profit']:.2f}")
    lines.append(f"stress_dd                    : {stress['max_drawdown_pct']:.2f}%")
    lines.append(f"panic_pf                     : {panic['profit_factor']:.4f}")
    lines.append(f"panic_net                    : {panic['net_profit']:.2f}")
    lines.append(f"panic_dd                     : {panic['max_drawdown_pct']:.2f}%")
    lines.append(f"max_consecutive_losses       : {baseline_eval['max_consecutive_losses']}")
    lines.append("")
    lines.append("[WALK-FORWARD SUMMARY]")
    for k, v in wf_summary.items():
        lines.append(f"{k:28}: {v}")
    lines.append("")
    lines.append("[BEST LOCAL EXPANSION RESULT]")
    for col in list(BASELINE_PARAMS.keys()):
        lines.append(f"{col:28}: {best_row[col]}")
    lines.append(f"distance_from_baseline       : {best_row['distance_from_baseline']}")
    lines.append(f"train_pf                     : {best_row['train_pf']:.4f}")
    lines.append(f"test_base_pf                 : {best_row['test_base_pf']:.4f}")
    lines.append(f"test_stress_pf               : {best_row['test_stress_pf']:.4f}")
    lines.append(f"test_panic_pf                : {best_row['test_panic_pf']:.4f}")
    lines.append(f"score                        : {best_row['score']:.2f}")
    lines.append("")
    lines.append("[PLATEAU CHECK]")
    if not plateau_df.empty:
        stable = plateau_df[(plateau_df["distance_from_baseline"] <= 2) & (plateau_df["test_base_pf"] > 1.0)]
        lines.append(f"nearby_configs_checked       : {len(plateau_df)}")
        lines.append(f"nearby_configs_pf_gt_1       : {len(stable)}")
        lines.append(f"median_nearby_pf             : {plateau_df['test_base_pf'].median():.4f}")
    else:
        lines.append("No plateau data")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    initialize_mt5()
    try:
        out_dir = Path(CFG["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)

        symbol = resolve_gold_symbol(CFG["symbol_candidates"])
        print(f"[OK] Resolved gold symbol: {symbol}")

        print(f"[DATA] Fetching {symbol} {timeframe_name(CFG['htf'])}...")
        htf = fetch_mt5(symbol, CFG["htf"], CFG["start"], CFG["end"])
        print(f"[DATA] Fetching {symbol} {timeframe_name(CFG['ltf'])}...")
        ltf = fetch_mt5(symbol, CFG["ltf"], CFG["start"], CFG["end"])
        print(f"[DATA] HTF bars: {len(htf):,}")
        print(f"[DATA] LTF bars: {len(ltf):,}")

        htf_feat = build_htf_features(htf)
        ltf_feat = build_ltf_features(ltf)
        base_data = merge_htf_to_ltf(ltf_feat, htf_feat).dropna().reset_index(drop=True)
        print(f"[DATA] merged bars: {len(base_data):,}")

        # --------------------------------------------------------
        # 1) Evaluate frozen baseline on the full sample
        # --------------------------------------------------------
        full_prepped = apply_session_filter(add_param_columns(base_data, BASELINE_PARAMS), BASELINE_PARAMS["session_name"])
        base_trades, base_eq = run_backtest(full_prepped, BASELINE_PARAMS, FRICTION_SCENARIOS["base"], symbol)
        stress_trades, stress_eq = run_backtest(full_prepped, BASELINE_PARAMS, FRICTION_SCENARIOS["stress"], symbol)
        panic_trades, panic_eq = run_backtest(full_prepped, BASELINE_PARAMS, FRICTION_SCENARIOS["panic"], symbol)

        base_sum = summarize_results(base_trades, base_eq, CFG["initial_balance"])
        stress_sum = summarize_results(stress_trades, stress_eq, CFG["initial_balance"])
        panic_sum = summarize_results(panic_trades, panic_eq, CFG["initial_balance"])
        max_losses = consecutive_loss_streak(base_trades)

        baseline_eval = {
            "base_sum": base_sum,
            "stress_sum": stress_sum,
            "panic_sum": panic_sum,
            "max_consecutive_losses": max_losses,
        }

        print("\n" + "=" * 78)
        print("BASELINE CANDIDATE A")
        print("=" * 78)
        print(pd.Series(BASELINE_PARAMS))
        print("-" * 78)
        print(f"Base   PF / Net / DD : {base_sum['profit_factor']:.2f} / {base_sum['net_profit']:.2f} / {base_sum['max_drawdown_pct']:.2f}%")
        print(f"Stress PF / Net / DD : {stress_sum['profit_factor']:.2f} / {stress_sum['net_profit']:.2f} / {stress_sum['max_drawdown_pct']:.2f}%")
        print(f"Panic  PF / Net / DD : {panic_sum['profit_factor']:.2f} / {panic_sum['net_profit']:.2f} / {panic_sum['max_drawdown_pct']:.2f}%")
        print(f"Max consecutive losses: {max_losses}")

        # --------------------------------------------------------
        # 2) Walk-forward validation of frozen baseline
        # --------------------------------------------------------
        wf_df, wf_eq_df = build_walk_forward_detail(base_data, BASELINE_PARAMS, symbol)
        wf_summary = summarize_walk_forward(wf_df)
        print("\n[WF] Summary")
        print(pd.Series(wf_summary))

        # --------------------------------------------------------
        # 3) Careful local expansion around the baseline
        # --------------------------------------------------------
        combos = combo_iter(LOCAL_SEARCH_SPACE)
        print(f"\n[SEARCH] Testing {len(combos)} local combinations around baseline...")

        # Use 2022-2024 as train and 2025+ as test for local expansion ranking.
        train_start = datetime(2022, 1, 1)
        train_end = datetime(2024, 12, 31, 23, 59, 59)
        test_start = datetime(2025, 1, 1)
        test_end = CFG["end"]

        rows = []
        for idx, params in enumerate(combos, start=1):
            if idx % 100 == 0 or idx == 1 or idx == len(combos):
                print(f"[SEARCH] {idx}/{len(combos)}")

            res = evaluate_config_on_window(
                base_data=base_data,
                params=params,
                symbol=symbol,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

            row = params.copy()
            row.update({
                "distance_from_baseline": neighborhood_distance(params, BASELINE_PARAMS),
                "train_trades": res["train_sum"]["total_trades"],
                "train_pf": res["train_sum"]["profit_factor"],
                "train_net": res["train_sum"]["net_profit"],
                "train_dd": res["train_sum"]["max_drawdown_pct"],
                "test_base_trades": res["test_sum_base"]["total_trades"],
                "test_base_pf": res["test_sum_base"]["profit_factor"],
                "test_base_net": res["test_sum_base"]["net_profit"],
                "test_base_dd": res["test_sum_base"]["max_drawdown_pct"],
                "test_stress_pf": res["test_sum_stress"]["profit_factor"],
                "test_stress_net": res["test_sum_stress"]["net_profit"],
                "test_stress_dd": res["test_sum_stress"]["max_drawdown_pct"],
                "test_panic_pf": res["test_sum_panic"]["profit_factor"],
                "test_panic_net": res["test_sum_panic"]["net_profit"],
                "test_panic_dd": res["test_sum_panic"]["max_drawdown_pct"],
                "score": res["score"],
            })
            rows.append(row)

        results = pd.DataFrame(rows)
        results = results.sort_values(
            by=["score", "test_base_pf", "test_stress_pf", "test_base_net", "distance_from_baseline"],
            ascending=[False, False, False, False, True],
        ).reset_index(drop=True)

        results.to_csv(out_dir / "local_expansion_results.csv", index=False)
        results.head(100).to_csv(out_dir / "local_expansion_top100.csv", index=False)

        plateau_df = results[results["distance_from_baseline"] <= 3].copy().reset_index(drop=True)
        plateau_df.to_csv(out_dir / "plateau_neighborhood.csv", index=False)

        best = results.iloc[0]
        best_params = {k: best[k] for k in BASELINE_PARAMS.keys()}

        print("\n" + "=" * 78)
        print("BEST LOCAL EXPANSION RESULT")
        print("=" * 78)
        print(best[list(BASELINE_PARAMS.keys())])
        print("-" * 78)
        print(f"Train PF / Net / DD       : {best['train_pf']:.2f} / {best['train_net']:.2f} / {best['train_dd']:.2f}%")
        print(f"Test base PF / Net / DD   : {best['test_base_pf']:.2f} / {best['test_base_net']:.2f} / {best['test_base_dd']:.2f}%")
        print(f"Test stress PF / Net / DD : {best['test_stress_pf']:.2f} / {best['test_stress_net']:.2f} / {best['test_stress_dd']:.2f}%")
        print(f"Test panic PF / Net / DD  : {best['test_panic_pf']:.2f} / {best['test_panic_net']:.2f} / {best['test_panic_dd']:.2f}%")
        print(f"Score                     : {best['score']:.2f}")
        print(f"Distance from baseline    : {best['distance_from_baseline']}")

        # --------------------------------------------------------
        # 4) Save rich artifacts for the frozen baseline and best local result
        # --------------------------------------------------------
        best_full_prepped = apply_session_filter(add_param_columns(base_data, best_params), best_params["session_name"])
        best_base_trades, best_base_eq = run_backtest(best_full_prepped, best_params, FRICTION_SCENARIOS["base"], symbol)
        best_stress_trades, best_stress_eq = run_backtest(best_full_prepped, best_params, FRICTION_SCENARIOS["stress"], symbol)

        base_trades.to_csv(out_dir / "baseline_fullsample_trades.csv", index=False)
        base_eq.to_csv(out_dir / "baseline_fullsample_equity.csv", index=False)
        stress_trades.to_csv(out_dir / "baseline_stress_trades.csv", index=False)
        stress_eq.to_csv(out_dir / "baseline_stress_equity.csv", index=False)
        panic_trades.to_csv(out_dir / "baseline_panic_trades.csv", index=False)
        panic_eq.to_csv(out_dir / "baseline_panic_equity.csv", index=False)
        monthly_returns(base_eq).to_csv(out_dir / "baseline_monthly_returns.csv", index=False)

        best_base_trades.to_csv(out_dir / "best_local_fullsample_trades.csv", index=False)
        best_base_eq.to_csv(out_dir / "best_local_fullsample_equity.csv", index=False)
        best_stress_trades.to_csv(out_dir / "best_local_stress_trades.csv", index=False)
        best_stress_eq.to_csv(out_dir / "best_local_stress_equity.csv", index=False)
        monthly_returns(best_base_eq).to_csv(out_dir / "best_local_monthly_returns.csv", index=False)

        wf_df.to_csv(out_dir / "walk_forward_summary.csv", index=False)
        wf_eq_df.to_csv(out_dir / "walk_forward_equity.csv", index=False)

        dow_df, hour_df = analyze_trade_distribution(base_trades)
        dow_df.to_csv(out_dir / "baseline_dayofweek_distribution.csv", index=False)
        hour_df.to_csv(out_dir / "baseline_hour_distribution.csv", index=False)

        save_equity_plot(
            {
                "baseline_base": base_eq,
                "baseline_stress": stress_eq,
                "baseline_panic": panic_eq,
            },
            out_dir / "baseline_equity_comparison.png",
            "Baseline Candidate A - Equity Comparison",
        )
        save_equity_plot(
            {
                "baseline": base_eq,
                "best_local": best_base_eq,
            },
            out_dir / "baseline_vs_bestlocal.png",
            "Baseline vs Best Local Expansion",
        )
        save_bar_plot(dow_df, "entry_dow", "net_profit", out_dir / "baseline_dayofweek_netprofit.png", "Baseline Net Profit by Day of Week")
        save_bar_plot(hour_df, "entry_hour", "net_profit", out_dir / "baseline_hour_netprofit.png", "Baseline Net Profit by Entry Hour")
        save_bar_plot(wf_df, "window", "test_base_net", out_dir / "walk_forward_test_netprofit.png", "Walk-Forward Test Net Profit")

        write_summary(symbol, baseline_eval, wf_summary, best, plateau_df, out_dir / "summary.txt")

        print(f"\n[FILES] Saved to folder: {out_dir.resolve()}")

    finally:
        shutdown_mt5()


if __name__ == "__main__":
    main()
