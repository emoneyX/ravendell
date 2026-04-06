#!/usr/bin/env python3
"""
EURUSD Combined ICT-like Live Bot (Demo / Dry-Run First) + Telegram
Fixed version: added pip_size and other execution constants to CONFIG.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import datetime, UTC
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

try:
    import MetaTrader5 as mt5
except Exception as e:
    raise RuntimeError("MetaTrader5 package is required. Install with: pip install MetaTrader5") from e

try:
    import requests
except Exception:
    requests = None


CONFIG = {
    "symbol": "EURUSD",
    "timeframe": "M15",

    "dry_run": True,
    "magic_number": 26042026,
    "deviation_points": 20,
    "poll_seconds": 10,

    # risk / money
    "risk_per_trade": 0.0025,
    "max_total_portfolio_risk": 0.0050,
    "max_open_positions_total": 2,
    "max_daily_loss_pct": 0.02,
    "max_weekly_loss_pct": 0.04,

    # pricing constants
    "pip_size": 0.0001,
    "pip_value_per_lot": 10.0,

    # execution guards
    "max_spread_pips": 1.5,
    "min_stop_pips": 2.0,
    "max_stop_pips": 30.0,

    # sessions
    "asian_start_hour": 0,
    "asian_end_hour": 6,
    "killzone_start_hour": 7,
    "killzone_end_hour": 11,
    "avoid_friday_after_hour": 10,
    "skip_monday": False,
    "use_news_safe_filter": True,
    "blocked_hours": [12, 13, 14],

    # filters
    "atr_period": 14,
    "use_atr_filter": True,
    "min_atr_pips": 4.0,
    "max_atr_pips": 25.0,

    "use_range_filter": True,
    "min_asian_range_pips": 8.0,
    "max_asian_range_pips": 35.0,

    # logic
    "sweep_buffer_pips": 1.0,
    "disp_body_atr_mult": 0.80,
    "disp_range_atr_mult": 1.20,
    "fvg_min_pips": 1.0,
    "stop_buffer_pips": 1.0,
    "tp_r": 2.0,
    "max_bars_in_trade": 20,

    # telegram
    "telegram_enabled": True,
    "telegram_bot_token": "8769071207:AAGPfN71VQS8JgL_viUKf7uLwbZvj3VlHQk",
    "telegram_chat_id": "789297530",

    # files
    "log_dir": "eurusd_combined_bot_logs",
}

STRATEGIES: Dict[str, Dict] = {
    "TOP2_v2": {
        "entry_mode": "near",
        "require_close_back_inside_range": True,
        "retest_wait_bars": 12,
        "max_sweep_bars_from_kz_open": 12,
        "max_trades_per_day": 2,
    },
    "TOP3": {
        "entry_mode": "far",
        "require_close_back_inside_range": False,
        "retest_wait_bars": 8,
        "max_sweep_bars_from_kz_open": 8,
        "max_trades_per_day": 1,
    },
}

LOG_DIR = Path(CONFIG["log_dir"])
LOG_DIR.mkdir(exist_ok=True)
TRADES_LOG = LOG_DIR / "trades_log.csv"
SIGNALS_LOG = LOG_DIR / "signals_log.csv"
STATE_FILE = LOG_DIR / "bot_state.json"


@dataclass
class Signal:
    strategy: str
    direction: str
    entry_time: pd.Timestamp
    setup_time: pd.Timestamp
    entry: float
    stop: float
    target: float
    asian_high: float
    asian_low: float
    asian_range_pips: float
    atr_pips: float
    reason: str
    signal_id: str


@dataclass
class PositionRecord:
    strategy: str
    signal_id: str
    direction: str
    entry_time: str
    entry_price: float
    stop_price: float
    target_price: float
    volume: float
    risk_amount: float
    mt5_ticket: Optional[int]
    dry_run: bool
    status: str
    exit_time: Optional[str] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_usd: Optional[float] = None


def append_csv(path: Path, row: Dict):
    df = pd.DataFrame([row])
    if path.exists():
        df.to_csv(path, mode="a", header=False, index=False)
    else:
        df.to_csv(path, index=False)


def load_state() -> Dict:
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "seen_signals": [],
        "open_positions": [],
        "daily_start_equity": {},
        "weekly_start_equity": {},
        "trade_count_by_day_strategy": {},
        "paused_days": [],
        "paused_weeks": [],
    }


def save_state(state: Dict):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def tg_send(text: str):
    if not CONFIG["telegram_enabled"] or not requests:
        return
    token = CONFIG["telegram_bot_token"]
    chat_id = CONFIG["telegram_chat_id"]
    if not token or not chat_id:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
    except Exception:
        pass


def tg_test_message():
    if CONFIG["telegram_enabled"]:
        tg_send("Telegram connected: EURUSD combined bot online.")


def initialize_mt5() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"MT5 initialize failed: {mt5.last_error()}")


def shutdown_mt5():
    try:
        mt5.shutdown()
    except Exception:
        pass


def tf_to_mt5(tf_name: str):
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
    }
    return mapping[tf_name]


def resolve_symbol(requested_symbol: str) -> str:
    symbols = mt5.symbols_get()
    if not symbols:
        return requested_symbol
    names = [s.name for s in symbols]
    if requested_symbol in names:
        mt5.symbol_select(requested_symbol, True)
        return requested_symbol
    req = requested_symbol.lower()
    candidates = [n for n in names if n.lower() == req]
    if not candidates:
        candidates = [n for n in names if req in n.lower()]
    if not candidates:
        candidates = [n for n in names if n.lower().startswith(req)]
    if not candidates and req == "eurusd":
        candidates = [n for n in names if "eurusd" in n.lower()]
    if not candidates:
        return requested_symbol
    preferred = sorted(candidates, key=lambda x: (0 if x.lower() == req else 1, len(x), x))[0]
    mt5.symbol_select(preferred, True)
    return preferred


def get_account_equity() -> float:
    info = mt5.account_info()
    if info is None:
        raise RuntimeError("Could not fetch MT5 account info.")
    return float(info.equity)


def get_symbol_tick(symbol: str):
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Could not fetch tick for {symbol}")
    return tick


def get_symbol_info(symbol: str):
    info = mt5.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Could not fetch symbol info for {symbol}")
    return info


def fetch_rates(symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
    tf = tf_to_mt5(timeframe)
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) == 0:
        raise RuntimeError(f"No rates returned for {symbol} {timeframe}")
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["time", "open", "high", "low", "close", "tick_volume"]].rename(columns={"tick_volume": "volume"})


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - prev_close).abs()
    tr3 = (df["low"] - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False).mean()


def prepare_df(m15: pd.DataFrame) -> pd.DataFrame:
    df = m15.copy()
    df["atr"] = atr(df, CONFIG["atr_period"])
    df["date"] = df["time"].dt.date
    df["hour"] = df["time"].dt.hour
    df["weekday"] = df["time"].dt.weekday
    df["body"] = (df["close"] - df["open"]).abs()
    df["range"] = df["high"] - df["low"]
    df["atr_pips"] = df["atr"] / CONFIG["pip_size"]
    asian_mask = (df["hour"] >= CONFIG["asian_start_hour"]) & (df["hour"] < CONFIG["asian_end_hour"])
    asian = (
        df.loc[asian_mask]
        .groupby("date")
        .agg(asian_high=("high", "max"), asian_low=("low", "min"))
        .reset_index()
    )
    asian["asian_range"] = asian["asian_high"] - asian["asian_low"]
    asian["asian_range_pips"] = asian["asian_range"] / CONFIG["pip_size"]
    return df.merge(asian, on="date", how="left")


def allowed_day(day_df: pd.DataFrame) -> bool:
    weekday = int(day_df["weekday"].iloc[0])
    if weekday >= 5:
        return False
    if CONFIG["skip_monday"] and weekday == 0:
        return False
    return True


def row_in_killzone(row: pd.Series, killzone_end_hour: int) -> bool:
    hour = int(row["hour"])
    weekday = int(row["weekday"])
    if weekday >= 5:
        return False
    if hour < CONFIG["killzone_start_hour"] or hour >= killzone_end_hour:
        return False
    if weekday == 4 and hour >= CONFIG["avoid_friday_after_hour"]:
        return False
    if CONFIG["use_news_safe_filter"] and hour in CONFIG["blocked_hours"]:
        return False
    return True


def daily_filters_ok(first_row: pd.Series) -> bool:
    if pd.isna(first_row["asian_high"]) or pd.isna(first_row["asian_low"]):
        return False
    if CONFIG["use_range_filter"]:
        arp = float(first_row["asian_range_pips"])
        if arp < CONFIG["min_asian_range_pips"] or arp > CONFIG["max_asian_range_pips"]:
            return False
    if CONFIG["use_atr_filter"]:
        ap = float(first_row["atr_pips"])
        if ap < CONFIG["min_atr_pips"] or ap > CONFIG["max_atr_pips"]:
            return False
    return True


def current_spread_pips(symbol: str) -> float:
    tick = get_symbol_tick(symbol)
    return abs(float(tick.ask) - float(tick.bid)) / CONFIG["pip_size"]


def portfolio_open_risk(state: Dict) -> float:
    return float(sum(pos["risk_amount"] for pos in state["open_positions"] if pos["status"] == "open"))


def fvg_from_displacement(window: pd.DataFrame, direction: str) -> Optional[Tuple[float, float]]:
    if len(window) != 3:
        return None
    a = window.iloc[0]
    c = window.iloc[2]
    if direction == "long":
        gap_low = float(a["high"])
        gap_high = float(c["low"])
    else:
        gap_low = float(c["high"])
        gap_high = float(a["low"])
    if gap_high <= gap_low:
        return None
    gap_pips = (gap_high - gap_low) / CONFIG["pip_size"]
    if gap_pips < CONFIG["fvg_min_pips"]:
        return None
    return gap_low, gap_high


def entry_price_from_fvg(fvg_low: float, fvg_high: float, entry_mode: str) -> float:
    if entry_mode == "near":
        return fvg_high
    if entry_mode == "mid":
        return (fvg_low + fvg_high) / 2.0
    if entry_mode == "far":
        return fvg_low
    raise ValueError(f"Unknown entry_mode: {entry_mode}")


def find_signals_for_strategy(df: pd.DataFrame, strategy_name: str, scfg: Dict, state: Dict) -> List[Signal]:
    signals: List[Signal] = []
    today_df = df[df["date"] == df["date"].max()].copy()
    if today_df.empty or not allowed_day(today_df):
        return signals

    kz = today_df[today_df.apply(lambda r: row_in_killzone(r, scfg["killzone_end_hour"]), axis=1)].copy()
    if kz.empty or not daily_filters_ok(kz.iloc[0]):
        return signals

    today_key = str(kz["date"].iloc[0])
    strategy_day_key = f"{today_key}|{strategy_name}"
    current_trade_count = int(state["trade_count_by_day_strategy"].get(strategy_day_key, 0))
    if current_trade_count >= scfg["max_trades_per_day"]:
        return signals

    asian_high = float(kz["asian_high"].iloc[0])
    asian_low = float(kz["asian_low"].iloc[0])
    asian_range_pips = float(kz["asian_range_pips"].iloc[0])

    sweep_buf = CONFIG["sweep_buffer_pips"] * CONFIG["pip_size"]
    stop_buf = CONFIG["stop_buffer_pips"] * CONFIG["pip_size"]
    used_signal_ids = set(state["seen_signals"])
    max_scan = min(len(kz), scfg["max_sweep_bars_from_kz_open"])

    for i in range(max_scan):
        row0 = kz.iloc[i]
        swept_high = float(row0["high"]) >= asian_high + sweep_buf
        swept_low = float(row0["low"]) <= asian_low - sweep_buf
        if swept_high and swept_low:
            continue
        if not swept_high and not swept_low:
            continue

        direction = "short" if swept_high else "long"
        sweep_idx = kz.index[i]
        sweep_time = pd.Timestamp(row0["time"])
        sweep_extreme = float(row0["high"]) if direction == "short" else float(row0["low"])

        post = today_df.loc[sweep_idx:].copy()
        if len(post) < 3:
            continue

        rows = post.reset_index()
        disp_idx = None
        disp_time = None
        atr_pips = None

        for j in range(1, len(rows) - 1):
            prev_row = rows.iloc[j - 1]
            row = rows.iloc[j]
            body = float(row["body"])
            rng = float(row["range"])
            atr_now = float(row["atr"])
            atr_pips_now = float(row["atr_pips"])
            if atr_now <= 0:
                continue

            body_ok = body >= CONFIG["disp_body_atr_mult"] * atr_now
            range_ok = rng >= CONFIG["disp_range_atr_mult"] * atr_now

            if direction == "long":
                dir_ok = float(row["close"]) > float(row["open"])
                structure_ok = float(row["close"]) >= asian_low if scfg["require_close_back_inside_range"] else float(row["close"]) > float(prev_row["high"])
            else:
                dir_ok = float(row["close"]) < float(row["open"])
                structure_ok = float(row["close"]) <= asian_high if scfg["require_close_back_inside_range"] else float(row["close"]) < float(prev_row["low"])

            if not (body_ok and range_ok and dir_ok and structure_ok):
                continue

            disp_idx = int(row["index"])
            disp_time = pd.Timestamp(row["time"])
            atr_pips = atr_pips_now
            break

        if disp_idx is None:
            continue

        disp_loc = today_df.index.get_loc(disp_idx)
        if disp_loc < 1 or disp_loc + 1 >= len(today_df):
            continue

        fvg_window = today_df.iloc[disp_loc - 1: disp_loc + 2].copy()
        fvg = fvg_from_displacement(fvg_window, direction)
        if fvg is None:
            continue
        fvg_low, fvg_high = fvg
        entry_ref = entry_price_from_fvg(fvg_low, fvg_high, scfg["entry_mode"])

        after_disp = today_df.iloc[disp_loc + 1: disp_loc + 1 + scfg["retest_wait_bars"]].copy()
        if after_disp.empty:
            continue

        entry_time = None
        raw_entry = None
        for _, row in after_disp.iterrows():
            high = float(row["high"])
            low = float(row["low"])
            if low <= entry_ref <= high:
                entry_time = pd.Timestamp(row["time"])
                raw_entry = entry_ref
                break

        if entry_time is None or raw_entry is None:
            continue

        signal_id = f"{strategy_name}|{entry_time.isoformat()}|{direction}"
        if signal_id in used_signal_ids:
            continue

        entry = raw_entry
        stop = sweep_extreme - stop_buf if direction == "long" else sweep_extreme + stop_buf
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            continue

        stop_pips = risk_per_unit / CONFIG["pip_size"]
        if stop_pips < CONFIG["min_stop_pips"] or stop_pips > CONFIG["max_stop_pips"]:
            continue

        target = entry + CONFIG["tp_r"] * risk_per_unit if direction == "long" else entry - CONFIG["tp_r"] * risk_per_unit

        signals.append(
            Signal(
                strategy=strategy_name,
                direction=direction,
                entry_time=entry_time,
                setup_time=sweep_time if disp_time is None else disp_time,
                entry=float(entry),
                stop=float(stop),
                target=float(target),
                asian_high=float(asian_high),
                asian_low=float(asian_low),
                asian_range_pips=float(asian_range_pips),
                atr_pips=float(atr_pips if atr_pips is not None else 0.0),
                reason="ICT-like sweep -> displacement -> FVG retest",
                signal_id=signal_id,
            )
        )

    signals.sort(key=lambda x: x.entry_time)
    remaining_slots = max(0, scfg["max_trades_per_day"] - current_trade_count)
    return signals[:remaining_slots]


def calc_order_size(symbol: str, entry: float, stop: float, equity: float) -> Tuple[float, float]:
    risk_amount = equity * CONFIG["risk_per_trade"]
    stop_pips = abs(entry - stop) / CONFIG["pip_size"]
    if stop_pips <= 0:
        return 0.0, 0.0
    lots = risk_amount / (stop_pips * CONFIG["pip_value_per_lot"])
    info = get_symbol_info(symbol)
    vol_min = float(info.volume_min)
    vol_step = float(info.volume_step)
    lots = max(vol_min, lots)
    steps = math.floor(lots / vol_step)
    lots = round(steps * vol_step, 2)
    return lots, risk_amount


def place_order(symbol: str, signal: Signal, volume: float) -> Tuple[Optional[int], bool, str]:
    tick = get_symbol_tick(symbol)
    order_type = mt5.ORDER_TYPE_BUY if signal.direction == "long" else mt5.ORDER_TYPE_SELL
    price = float(tick.ask) if signal.direction == "long" else float(tick.bid)
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_type,
        "price": price,
        "sl": signal.stop,
        "tp": signal.target,
        "deviation": CONFIG["deviation_points"],
        "magic": CONFIG["magic_number"],
        "comment": signal.strategy,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None:
        return None, False, "order_send returned None"
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return None, False, f"retcode={result.retcode}"
    return int(result.order), True, "filled"


def record_signal(signal: Signal):
    append_csv(SIGNALS_LOG, asdict(signal))


def record_trade(trade: PositionRecord):
    append_csv(TRADES_LOG, asdict(trade))


def sync_open_positions_with_mt5(symbol: str, state: Dict):
    if CONFIG["dry_run"]:
        return
    positions = mt5.positions_get(symbol=symbol)
    open_tickets = set()
    if positions:
        open_tickets = {int(p.ticket) for p in positions}
    for pos in state["open_positions"]:
        if pos["status"] != "open":
            continue
        ticket = pos.get("mt5_ticket")
        if ticket is None or ticket in open_tickets:
            continue
        pos["status"] = "closed"
        pos["exit_time"] = datetime.now(UTC).isoformat()
        pos["exit_reason"] = "closed_in_mt5"
        pos["exit_price"] = None
        pos["pnl_usd"] = None


def update_dry_run_positions(symbol: str, state: Dict):
    if not CONFIG["dry_run"]:
        return
    try:
        df = prepare_df(fetch_rates(symbol, CONFIG["timeframe"], bars=100))
    except Exception:
        return
    if df.empty:
        return
    last_bar = df.iloc[-1]
    high = float(last_bar["high"])
    low = float(last_bar["low"])
    close = float(last_bar["close"])
    now_iso = pd.Timestamp(last_bar["time"]).isoformat()

    for pos in state["open_positions"]:
        if pos["status"] != "open":
            continue
        direction = pos["direction"]
        stop = float(pos["stop_price"])
        target = float(pos["target_price"])
        entry = float(pos["entry_price"])
        vol = float(pos["volume"])
        exit_price = None
        exit_reason = None

        if direction == "long":
            if low <= stop:
                exit_price = stop
                exit_reason = "stop"
            elif high >= target:
                exit_price = target
                exit_reason = "target"
        else:
            if high >= stop:
                exit_price = stop
                exit_reason = "stop"
            elif low <= target:
                exit_price = target
                exit_reason = "target"

        if exit_price is None:
            entry_time = pd.Timestamp(pos["entry_time"])
            current_time = pd.Timestamp(last_bar["time"])
            bars_held = int((current_time - entry_time) / pd.Timedelta(minutes=15))
            if bars_held >= CONFIG["max_bars_in_trade"]:
                exit_price = close
                exit_reason = "time_exit"

        if exit_price is None:
            continue

        move_pips = (exit_price - entry) / CONFIG["pip_size"] if direction == "long" else (entry - exit_price) / CONFIG["pip_size"]
        pnl_usd = move_pips * CONFIG["pip_value_per_lot"] * vol - (7.0 * vol)

        pos["status"] = "closed"
        pos["exit_time"] = now_iso
        pos["exit_price"] = exit_price
        pos["exit_reason"] = exit_reason
        pos["pnl_usd"] = pnl_usd
        record_trade(PositionRecord(**pos))
        tg_send(f"[DRY-RUN CLOSED] {pos['strategy']} {direction} {exit_reason} pnl=${pnl_usd:.2f}")


def get_day_key(ts: datetime) -> str:
    return ts.strftime("%Y-%m-%d")


def get_week_key(ts: datetime) -> str:
    y, w, _ = ts.isocalendar()
    return f"{y}-W{w:02d}"


def initialize_equity_buckets(state: Dict, equity: float):
    now = datetime.now(UTC)
    day_key = get_day_key(now)
    week_key = get_week_key(now)
    if day_key not in state["daily_start_equity"]:
        state["daily_start_equity"][day_key] = equity
    if week_key not in state["weekly_start_equity"]:
        state["weekly_start_equity"][week_key] = equity


def risk_paused_today(state: Dict, equity: float) -> bool:
    now = datetime.now(UTC)
    day_key = get_day_key(now)
    week_key = get_week_key(now)
    initialize_equity_buckets(state, equity)
    day_start = float(state["daily_start_equity"][day_key])
    week_start = float(state["weekly_start_equity"][week_key])
    day_dd = (equity - day_start) / day_start
    week_dd = (equity - week_start) / week_start
    if day_dd <= -CONFIG["max_daily_loss_pct"]:
        if day_key not in state["paused_days"]:
            state["paused_days"].append(day_key)
            tg_send(f"[PAUSE] Daily drawdown limit hit: {day_dd:.2%}")
        return True
    if week_dd <= -CONFIG["max_weekly_loss_pct"]:
        if week_key not in state["paused_weeks"]:
            state["paused_weeks"].append(week_key)
            tg_send(f"[PAUSE] Weekly drawdown limit hit: {week_dd:.2%}")
        return True
    return day_key in state["paused_days"] or week_key in state["paused_weeks"]


def maybe_send_startup(symbol: str):
    mode = "DRY-RUN" if CONFIG["dry_run"] else "LIVE DEMO"
    text = (
        f"EURUSD Combined Bot started\n"
        f"Mode: {mode}\n"
        f"Symbol: {symbol}\n"
        f"Strategies: TOP2_v2 + TOP3\n"
        f"Risk/trade: {CONFIG['risk_per_trade']:.2%}\n"
        f"Portfolio risk cap: {CONFIG['max_total_portfolio_risk']:.2%}"
    )
    print(text)
    tg_send(text)


def main():
    initialize_mt5()
    symbol = resolve_symbol(CONFIG["symbol"])
    mt5.symbol_select(symbol, True)
    tg_test_message()
    maybe_send_startup(symbol)

    state = load_state()
    last_processed_bar_time = None

    while True:
        try:
            equity = get_account_equity()
            initialize_equity_buckets(state, equity)

            sync_open_positions_with_mt5(symbol, state)
            update_dry_run_positions(symbol, state)
            state["open_positions"] = [p for p in state["open_positions"] if p["status"] == "open"]

            df = prepare_df(fetch_rates(symbol, CONFIG["timeframe"], bars=500))
            latest_bar_time = pd.Timestamp(df["time"].iloc[-1])

            if last_processed_bar_time is not None and latest_bar_time <= last_processed_bar_time:
                save_state(state)
                time.sleep(CONFIG["poll_seconds"])
                continue

            last_processed_bar_time = latest_bar_time

            spread = current_spread_pips(symbol)
            if spread > CONFIG["max_spread_pips"]:
                print(f"[SKIP] Spread too high: {spread:.2f} pips")
                save_state(state)
                time.sleep(CONFIG["poll_seconds"])
                continue

            if risk_paused_today(state, equity):
                print("[PAUSE] Risk pause active. Monitoring only.")
                save_state(state)
                time.sleep(CONFIG["poll_seconds"])
                continue

            if len(state["open_positions"]) >= CONFIG["max_open_positions_total"]:
                print("[SKIP] Max open positions reached.")
                save_state(state)
                time.sleep(CONFIG["poll_seconds"])
                continue

            open_risk = portfolio_open_risk(state)
            if open_risk >= equity * CONFIG["max_total_portfolio_risk"]:
                print("[SKIP] Portfolio open risk cap reached.")
                save_state(state)
                time.sleep(CONFIG["poll_seconds"])
                continue

            all_new_signals: List[Signal] = []
            for strategy_name, scfg in STRATEGIES.items():
                signals = find_signals_for_strategy(df, strategy_name, scfg, state)
                all_new_signals.extend(signals)

            all_new_signals.sort(key=lambda x: x.entry_time)

            for signal in all_new_signals:
                if signal.signal_id in state["seen_signals"]:
                    continue

                equity = get_account_equity() if not CONFIG["dry_run"] else equity
                open_risk = portfolio_open_risk(state)

                if len(state["open_positions"]) >= CONFIG["max_open_positions_total"]:
                    break
                if open_risk >= equity * CONFIG["max_total_portfolio_risk"]:
                    break

                volume, risk_amount = calc_order_size(symbol, signal.entry, signal.stop, equity)
                if volume <= 0 or risk_amount <= 0:
                    continue

                projected_open_risk = open_risk + risk_amount
                if projected_open_risk > equity * CONFIG["max_total_portfolio_risk"]:
                    print(f"[SKIP] {signal.strategy} exceeds portfolio risk cap.")
                    continue

                record_signal(signal)

                if CONFIG["dry_run"]:
                    pos = PositionRecord(
                        strategy=signal.strategy,
                        signal_id=signal.signal_id,
                        direction=signal.direction,
                        entry_time=signal.entry_time.isoformat(),
                        entry_price=signal.entry,
                        stop_price=signal.stop,
                        target_price=signal.target,
                        volume=volume,
                        risk_amount=risk_amount,
                        mt5_ticket=None,
                        dry_run=True,
                        status="open",
                    )
                    state["open_positions"].append(asdict(pos))
                    state["seen_signals"].append(signal.signal_id)
                    day_key = f"{str(signal.entry_time.date())}|{signal.strategy}"
                    state["trade_count_by_day_strategy"][day_key] = int(state["trade_count_by_day_strategy"].get(day_key, 0)) + 1

                    msg = (
                        f"[DRY-RUN OPEN] {signal.strategy} {signal.direction}\n"
                        f"entry={signal.entry:.5f} stop={signal.stop:.5f} target={signal.target:.5f}\n"
                        f"vol={volume:.2f} risk=${risk_amount:.2f} spread={spread:.2f}p"
                    )
                    print(msg)
                    tg_send(msg)
                    continue

                ticket, ok, note = place_order(symbol, signal, volume)
                if not ok:
                    print(f"[ORDER FAIL] {signal.strategy} {signal.direction} {note}")
                    tg_send(f"[ORDER FAIL] {signal.strategy} {signal.direction} {note}")
                    continue

                pos = PositionRecord(
                    strategy=signal.strategy,
                    signal_id=signal.signal_id,
                    direction=signal.direction,
                    entry_time=signal.entry_time.isoformat(),
                    entry_price=signal.entry,
                    stop_price=signal.stop,
                    target_price=signal.target,
                    volume=volume,
                    risk_amount=risk_amount,
                    mt5_ticket=ticket,
                    dry_run=False,
                    status="open",
                )
                state["open_positions"].append(asdict(pos))
                state["seen_signals"].append(signal.signal_id)
                day_key = f"{str(signal.entry_time.date())}|{signal.strategy}"
                state["trade_count_by_day_strategy"][day_key] = int(state["trade_count_by_day_strategy"].get(day_key, 0)) + 1

                msg = (
                    f"[LIVE OPEN] {signal.strategy} {signal.direction}\n"
                    f"ticket={ticket} entry={signal.entry:.5f} stop={signal.stop:.5f} target={signal.target:.5f}\n"
                    f"vol={volume:.2f} risk=${risk_amount:.2f}"
                )
                print(msg)
                tg_send(msg)

            save_state(state)
            time.sleep(CONFIG["poll_seconds"])

        except KeyboardInterrupt:
            print("Stopped by user.")
            save_state(state)
            break
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            tg_send(f"[BOT ERROR] {type(e).__name__}: {e}")
            save_state(state)
            time.sleep(CONFIG["poll_seconds"])

    shutdown_mt5()


if __name__ == "__main__":
    main()
