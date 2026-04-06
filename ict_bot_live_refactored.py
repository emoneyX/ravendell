"""
ICT XAU/USD LIVE TRADING BOT v1.1 (refactored)
==============================================
Refactor goals
- safer config handling
- cleaner logging / structure
- fix Telegram enabled logic
- avoid hardcoded secrets in source
- timezone-aware UTC
- smaller helper functions
- safer MT5 request handling
- more defensive state handling

Notes
-----
1) Fill TELEGRAM_TOKEN / TELEGRAM_CHAT_ID from environment variables.
2) Keep dry_run=True until demo validation is clean.
3) This is still your strategy logic; cleanup is mostly structural/safety-oriented.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import requests


# =============================================================================
# CONFIG
# =============================================================================
@dataclass
class Config:
    # MT5
    symbol: str = "XAUUSD"
    magic_number: int = 360360
    deviation: int = 30

    # Telegram
    telegram_token: str = os.getenv("TELEGRAM_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")

    # Timeframes
    tf_htf: int = mt5.TIMEFRAME_H1
    tf_ltf: int = mt5.TIMEFRAME_M15
    htf_bars: int = 500
    ltf_bars: int = 500

    # Risk
    account_balance_override: float = 0.0
    risk_per_trade_pct: float = 1.0
    default_sl_pips: float = 150.0
    min_rr: float = 2.0
    max_rr: float = 6.0
    max_daily_trades: int = 2
    max_daily_loss_pct: float = 1.5
    max_dd_pct: float = 10.0

    # ICT params
    swing_strength: int = 3
    structure_lookback: int = 20
    ob_max_age_bars: int = 45
    fvg_min_size_pips: float = 28.0
    fvg_max_age_bars: int = 40
    liquidity_lookback: int = 50
    ote_fib_low: float = 0.618
    ote_fib_high: float = 0.786
    min_confluences: int = 2
    min_confluence_score: int = 3
    displacement_min_pips: float = 25.0
    displacement_lookback: int = 8
    sl_buffer_pips: float = 15.0

    # Kill zones UTC
    kill_zone_mode: str = "all"
    london_open: int = 7
    london_close: int = 10
    ny_am_open: int = 12
    ny_am_close: int = 15
    ny_pm_open: int = 15
    ny_pm_close: int = 17

    # Asian range UTC
    asian_open: int = 0
    asian_close: int = 6

    # Trade management
    be_trigger_r: float = 1.5
    be_lock_r: float = 0.3
    trail_trigger_r: float = 2.0
    trail_step_r: float = 0.5
    partial_tp_enabled: bool = True
    partial_tp_r: float = 1.5
    partial_tp_pct: float = 0.5

    # Limit entry
    limit_entry_enabled: bool = True
    limit_wait_bars: int = 4

    # Pip
    pip_size: float = 0.10
    contract_size: float = 1.0

    # Bot
    dry_run: bool = True
    scan_interval_sec: int = 5
    state_file: str = "bot_state.json"
    log_file: str = "bot_log.txt"


# =============================================================================
# LOGGING
# =============================================================================
def setup_logging(cfg: Config) -> logging.Logger:
    logger = logging.getLogger("ICTBot")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(cfg.log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


log: logging.Logger


# =============================================================================
# TELEGRAM
# =============================================================================
class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = token.strip()
        self.chat_id = chat_id.strip()
        self.enabled = bool(self.token and self.chat_id)
        self.base_url = f"https://api.telegram.org/bot{self.token}" if self.enabled else ""

    def send(self, text: str, parse_mode: str = "HTML") -> None:
        if not self.enabled:
            return
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "parse_mode": parse_mode},
                timeout=10,
            )
            if not resp.ok:
                log.warning("Telegram error: %s", resp.text)
        except Exception as exc:
            log.warning("Telegram send failed: %s", exc)

    def signal(self, direction: str, entry: float, sl: float, tp: float,
               confluences: List[str], kz: str, score: int, mode: str, rr: float) -> None:
        arrow = "🟢 LONG" if direction == "long" else "🔴 SHORT"
        confs = " | ".join(confluences)
        msg = (
            f"<b>{arrow} XAUUSD</b>\n"
            f"━━━━━━━━━━━━━━━\n"
            f"Entry  : <code>{entry:.2f}</code>\n"
            f"SL     : <code>{sl:.2f}</code>\n"
            f"TP     : <code>{tp:.2f}</code>\n"
            f"R:R    : <code>{rr:.1f}</code>\n"
            f"Mode   : {mode}\n"
            f"KZ     : {kz}\n"
            f"Score  : {score}\n"
            f"Confl  : {confs}\n"
            f"━━━━━━━━━━━━━━━"
        )
        self.send(msg)

    def trade_update(self, action: str, details: str = "") -> None:
        self.send(f"📋 <b>{action}</b>\n{details}")


# =============================================================================
# STATE
# =============================================================================
def default_state() -> Dict[str, Any]:
    return {
        "equity_peak": 0.0,
        "daily_trades": {},
        "daily_pnl": {},
        "last_trade_idx": 0,
        "open_trade": None,
        "pending_limit": None,
    }


def load_state(filepath: str) -> Dict[str, Any]:
    if not os.path.exists(filepath):
        return default_state()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            state = json.load(f)
        merged = default_state()
        merged.update(state)
        return merged
    except Exception:
        return default_state()


def save_state(filepath: str, state: Dict[str, Any]) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


# =============================================================================
# MT5
# =============================================================================
class MT5Manager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.symbol: Optional[str] = None

    def connect(self) -> bool:
        if not mt5.initialize():
            log.error("MT5 init failed: %s", mt5.last_error())
            return False

        candidates = [
            self.cfg.symbol, "XAUUSDm", "XAUUSD.raw", "XAUUSD.a",
            "XAUUSD.ecn", "XAUUSDc", "Gold", "GOLD", "GOLDm",
            "XAUUSD.pro", "XAUUSD.std", "XAUUSD#",
        ]

        for sym in candidates:
            if mt5.symbol_info(sym) is not None:
                self.symbol = sym
                mt5.symbol_select(sym, True)
                log.info("MT5 connected — symbol: %s", sym)
                return True

        all_syms = mt5.symbols_get()
        if all_syms:
            for s in all_syms:
                upper = s.name.upper()
                if "XAU" in upper and "USD" in upper:
                    self.symbol = s.name
                    mt5.symbol_select(s.name, True)
                    log.info("MT5 connected — symbol: %s", s.name)
                    return True

        log.error("No gold symbol found on broker")
        return False

    def get_bars(self, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        if not self.symbol:
            return None
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            log.warning("No data for %s TF=%s", self.symbol, timeframe)
            return None

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert(None)
        df.set_index("time", inplace=True)
        df.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "tick_volume": "Volume",
            },
            inplace=True,
        )
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def get_balance(self) -> float:
        if self.cfg.account_balance_override > 0:
            return self.cfg.account_balance_override
        info = mt5.account_info()
        return float(info.balance) if info else 10000.0

    def get_tick(self) -> Optional[float]:
        if not self.symbol:
            return None
        tick = mt5.symbol_info_tick(self.symbol)
        return float(tick.bid) if tick else None

    def get_my_positions(self) -> List[Any]:
        if not self.symbol:
            return []
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        return [p for p in positions if p.magic == self.cfg.magic_number]

    def get_my_pending(self) -> List[Any]:
        if not self.symbol:
            return []
        orders = mt5.orders_get(symbol=self.symbol)
        if orders is None:
            return []
        return [o for o in orders if o.magic == self.cfg.magic_number]

    def _send_request(self, request: Dict[str, Any]) -> Optional[Any]:
        if self.cfg.dry_run:
            log.info("[DRY RUN] %s", request)
            return type("DryResult", (), {"retcode": mt5.TRADE_RETCODE_DONE, "order": -1, "price": request.get("price", 0)})
        result = mt5.order_send(request)
        if result is None:
            log.error("order_send returned None: %s", mt5.last_error())
            return None
        return result

    def open_market(self, direction: str, lot: float, sl: float, tp: float,
                    comment: str = "ICT_v1") -> Optional[int]:
        if not self.symbol:
            return None
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            log.error("Cannot get tick for market order")
            return None

        is_long = direction == "long"
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": mt5.ORDER_TYPE_BUY if is_long else mt5.ORDER_TYPE_SELL,
            "price": tick.ask if is_long else tick.bid,
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "deviation": self.cfg.deviation,
            "magic": self.cfg.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = self._send_request(request)
        if result is None:
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("Order failed: %s", getattr(result, "comment", result.retcode))
            return None
        log.info("Market %s opened: ticket=%s price=%s lot=%.2f",
                 direction, result.order, getattr(result, "price", request["price"]), lot)
        return int(result.order)

    def open_limit(self, direction: str, lot: float, price: float, sl: float, tp: float,
                   expiry_minutes: int = 60, comment: str = "ICT_v1_LMT") -> Optional[int]:
        if not self.symbol:
            return None
        order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "long" else mt5.ORDER_TYPE_SELL_LIMIT
        expiry_time = datetime.now(UTC) + timedelta(minutes=expiry_minutes)
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": self.symbol,
            "volume": lot,
            "type": order_type,
            "price": round(price, 2),
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "deviation": self.cfg.deviation,
            "magic": self.cfg.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_SPECIFIED,
            "expiration": int(expiry_time.timestamp()),
        }
        result = self._send_request(request)
        if result is None:
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error("Limit order failed: %s", getattr(result, "comment", result.retcode))
            return None
        log.info("Limit %s placed: ticket=%s @ %.2f", direction, result.order, price)
        return int(result.order)

    def close_partial(self, position: Any, lot_to_close: float, comment: str = "Partial_TP") -> bool:
        if not self.symbol:
            return False
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False
        is_long = position.type == mt5.ORDER_TYPE_BUY
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": round(lot_to_close, 2),
            "type": mt5.ORDER_TYPE_SELL if is_long else mt5.ORDER_TYPE_BUY,
            "position": position.ticket,
            "price": tick.bid if is_long else tick.ask,
            "deviation": self.cfg.deviation,
            "magic": self.cfg.magic_number,
            "comment": comment,
        }
        result = self._send_request(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE

    def modify_sl(self, position: Any, new_sl: float) -> bool:
        if not self.symbol:
            return False
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": position.ticket,
            "sl": round(new_sl, 2),
            "tp": round(position.tp, 2),
        }
        result = self._send_request(request)
        return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


# =============================================================================
# ICT ENGINE
# =============================================================================
class ICTEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def find_swing_points(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        swings: List[Dict[str, Any]] = []
        n = self.cfg.swing_strength
        highs = df["High"].values
        lows = df["Low"].values

        for i in range(n, len(df) - n):
            if all(highs[i] >= highs[i-j] for j in range(1, n+1)) and all(highs[i] >= highs[i+j] for j in range(1, n+1)):
                swings.append({"index": i, "price": float(highs[i]), "timestamp": df.index[i], "type": "SH"})
            if all(lows[i] <= lows[i-j] for j in range(1, n+1)) and all(lows[i] <= lows[i+j] for j in range(1, n+1)):
                swings.append({"index": i, "price": float(lows[i]), "timestamp": df.index[i], "type": "SL"})
        return swings

    def analyze_structure(self, swings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        if len(swings) < 4:
            return events

        bias = "neutral"
        for i in range(1, len(swings)):
            curr = swings[i]
            prev_same = [s for s in swings[:i] if s["type"] == curr["type"]]
            if not prev_same:
                continue
            prev = prev_same[-1]

            if curr["type"] == "SH":
                if curr["price"] > prev["price"]:
                    etype = "BOS_BULL" if bias == "bullish" else "CHOCH_BULL"
                    events.append({"type": etype, "index": curr["index"], "price": curr["price"], "timestamp": curr["timestamp"], "bias": "bullish"})
                    bias = "bullish"
                elif bias == "bearish":
                    events.append({"type": "BOS_BEAR", "index": curr["index"], "price": curr["price"], "timestamp": curr["timestamp"], "bias": "bearish"})
            else:
                if curr["price"] < prev["price"]:
                    etype = "BOS_BEAR" if bias == "bearish" else "CHOCH_BEAR"
                    events.append({"type": etype, "index": curr["index"], "price": curr["price"], "timestamp": curr["timestamp"], "bias": "bearish"})
                    bias = "bearish"
                elif bias == "bullish":
                    events.append({"type": "BOS_BULL", "index": curr["index"], "price": curr["price"], "timestamp": curr["timestamp"], "bias": "bullish"})
        return events

    def find_order_blocks(self, df: pd.DataFrame, structure_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        obs: List[Dict[str, Any]] = []
        opens = df["Open"].values
        closes = df["Close"].values
        highs = df["High"].values
        lows = df["Low"].values

        for event in structure_events:
            idx = event["index"]
            if idx < 3:
                continue
            if "BULL" in event["type"]:
                for j in range(idx - 1, max(idx - 6, 0), -1):
                    if closes[j] < opens[j]:
                        obs.append({"index": j, "timestamp": df.index[j], "top": float(opens[j]), "bottom": float(lows[j]), "type": "bullish", "valid": True})
                        break
            elif "BEAR" in event["type"]:
                for j in range(idx - 1, max(idx - 6, 0), -1):
                    if closes[j] > opens[j]:
                        obs.append({"index": j, "timestamp": df.index[j], "top": float(highs[j]), "bottom": float(closes[j]), "type": "bearish", "valid": True})
                        break
        return obs

    def find_fvg(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        fvgs: List[Dict[str, Any]] = []
        highs = df["High"].values
        lows = df["Low"].values
        min_gap = self.cfg.fvg_min_size_pips * self.cfg.pip_size

        for i in range(2, len(df)):
            gap_bull = lows[i] - highs[i - 2]
            if gap_bull > min_gap:
                fvgs.append({"index": i, "timestamp": df.index[i], "top": float(lows[i]), "bottom": float(highs[i - 2]), "type": "bullish"})
            gap_bear = lows[i - 2] - highs[i]
            if gap_bear > min_gap:
                fvgs.append({"index": i, "timestamp": df.index[i], "top": float(lows[i - 2]), "bottom": float(highs[i]), "type": "bearish"})
        return fvgs

    def find_liquidity_levels(self, df: pd.DataFrame, current_idx: int) -> Dict[str, List[float]]:
        lookback = min(self.cfg.liquidity_lookback, current_idx)
        start = current_idx - lookback
        window = df.iloc[start:current_idx]
        prev_highs: List[float] = []
        prev_lows: List[float] = []
        highs = window["High"].values
        lows = window["Low"].values
        n = 2

        for i in range(n, len(window) - n):
            if all(highs[i] >= highs[i-j] for j in range(1, n+1)) and all(highs[i] >= highs[i+j] for j in range(1, n+1)):
                prev_highs.append(float(highs[i]))
            if all(lows[i] <= lows[i-j] for j in range(1, n+1)) and all(lows[i] <= lows[i+j] for j in range(1, n+1)):
                prev_lows.append(float(lows[i]))

        return {
            "buy_side_liquidity": sorted(prev_highs)[-3:] if prev_highs else [],
            "sell_side_liquidity": sorted(prev_lows)[:3] if prev_lows else [],
        }

    def is_kill_zone(self, timestamp: pd.Timestamp) -> Tuple[bool, str]:
        hour = timestamp.hour
        mode = self.cfg.kill_zone_mode
        london = self.cfg.london_open <= hour < self.cfg.london_close
        ny_am = self.cfg.ny_am_open <= hour < self.cfg.ny_am_close
        ny_pm = self.cfg.ny_pm_open <= hour < self.cfg.ny_pm_close

        if mode == "all":
            if london:
                return True, "London"
            if ny_am:
                return True, "NY_AM"
            if ny_pm:
                return True, "NY_PM"
        elif mode == "ny_am_only":
            if ny_am:
                return True, "NY_AM"
        elif mode == "london_ny":
            if london:
                return True, "London"
            if ny_am:
                return True, "NY_AM"
            if ny_pm:
                return True, "NY_PM"
        return False, "Off_Session"

    def get_asian_range(self, df: pd.DataFrame, current_date) -> Optional[Dict[str, float]]:
        asian_data = df[(df.index.date == current_date) & (df.index.hour >= self.cfg.asian_open) & (df.index.hour < self.cfg.asian_close)]
        if len(asian_data) < 4:
            return None
        high = float(asian_data["High"].max())
        low = float(asian_data["Low"].min())
        return {"high": high, "low": low, "mid": (high + low) / 2}

    def premium_discount(self, price: float, swing_high: float, swing_low: float) -> str:
        return "premium" if price > (swing_high + swing_low) / 2 else "discount"

    def ote_zone(self, swing_high: float, swing_low: float, direction: str) -> Tuple[float, float]:
        rng = swing_high - swing_low
        if direction == "long":
            return swing_high - rng * self.cfg.ote_fib_high, swing_high - rng * self.cfg.ote_fib_low
        return swing_low + rng * self.cfg.ote_fib_low, swing_low + rng * self.cfg.ote_fib_high

    def is_in_ote(self, price: float, swing_high: float, swing_low: float, direction: str) -> bool:
        ote_bot, ote_top = self.ote_zone(swing_high, swing_low, direction)
        return ote_bot <= price <= ote_top

    def check_displacement(self, df: pd.DataFrame, idx: int, direction: str) -> bool:
        min_size = self.cfg.displacement_min_pips * self.cfg.pip_size
        start = max(0, idx - self.cfg.displacement_lookback)

        for j in range(start, idx):
            body = abs(df["Close"].iloc[j] - df["Open"].iloc[j])
            candle_range = df["High"].iloc[j] - df["Low"].iloc[j]
            if candle_range < min_size:
                continue
            if candle_range > 0 and body / candle_range < 0.35:
                continue
            if direction == "long" and df["Close"].iloc[j] > df["Open"].iloc[j]:
                return True
            if direction == "short" and df["Close"].iloc[j] < df["Open"].iloc[j]:
                return True
        return False

    def get_previous_day_hl(self, df: pd.DataFrame, idx: int) -> Optional[Dict[str, float]]:
        current_date = df.index[idx].date()
        prev_day_data = df[df.index.date < current_date]
        if len(prev_day_data) == 0:
            return None
        last_date = prev_day_data.index[-1].date()
        day_data = prev_day_data[prev_day_data.index.date == last_date]
        if len(day_data) == 0:
            return None
        return {
            "high": float(day_data["High"].max()),
            "low": float(day_data["Low"].min()),
            "close": float(day_data["Close"].iloc[-1]),
            "open": float(day_data["Open"].iloc[0]),
        }

    def check_stop_hunt(self, df: pd.DataFrame, idx: int, prev_day: Dict[str, float], direction: str) -> bool:
        current_date = df.index[idx].date()
        today_data = df[(df.index.date == current_date) & (df.index <= df.index[idx])]
        if len(today_data) < 3:
            return False

        if direction == "long":
            return bool(today_data["Low"].min() < prev_day["low"] and df["Close"].iloc[idx] > prev_day["low"])
        return bool(today_data["High"].max() > prev_day["high"] and df["Close"].iloc[idx] < prev_day["high"])

    def check_liquidity_sweep(self, df: pd.DataFrame, idx: int, liq: Dict[str, List[float]], bias: str) -> bool:
        if idx < 2:
            return False
        prev_high = float(df["High"].iloc[idx - 1])
        prev_low = float(df["Low"].iloc[idx - 1])
        curr_close = float(df["Close"].iloc[idx])

        if bias == "bullish":
            return any(prev_low < level and curr_close > level for level in liq["sell_side_liquidity"])
        if bias == "bearish":
            return any(prev_high > level and curr_close < level for level in liq["buy_side_liquidity"])
        return False


# =============================================================================
# SCANNER
# =============================================================================
class LiveScanner:
    def __init__(self, cfg: Config, engine: ICTEngine):
        self.cfg = cfg
        self.engine = engine

    def _find_active_ob(self, obs: List[Dict[str, Any]], current_idx: int, price: float) -> Optional[Dict[str, Any]]:
        for ob in reversed(obs):
            if ob["index"] >= current_idx:
                continue
            if current_idx - ob["index"] > self.cfg.ob_max_age_bars:
                continue
            if ob["bottom"] <= price <= ob["top"]:
                return ob
        return None

    def _find_active_fvg(self, fvgs: List[Dict[str, Any]], current_idx: int, price: float) -> Optional[Dict[str, Any]]:
        for fvg in reversed(fvgs):
            if fvg["index"] >= current_idx:
                continue
            if current_idx - fvg["index"] > self.cfg.fvg_max_age_bars:
                continue
            if fvg["bottom"] <= price <= fvg["top"]:
                return fvg
        return None

    def get_current_bias(self, htf_structure: List[Dict[str, Any]], df_htf: pd.DataFrame,
                         df_ltf: pd.DataFrame) -> Tuple[str, str]:
        ema50 = df_htf["Close"].ewm(span=50, adjust=False).mean()
        ema200 = df_htf["Close"].ewm(span=200, adjust=False).mean()
        price = float(df_htf["Close"].iloc[-1])
        e50 = float(ema50.iloc[-1])
        e200 = float(ema200.iloc[-1])

        if price > e50 > e200:
            trend = "bullish"
        elif price < e50 < e200:
            trend = "bearish"
        elif price > e50:
            trend = "weak_bullish"
        elif price < e50:
            trend = "weak_bearish"
        else:
            trend = "neutral"

        struct_bias = "neutral"
        pending_bias = "neutral"
        pending_count = 0
        latest_ts = df_ltf.index[-1]

        for ev in htf_structure:
            if ev["timestamp"] <= latest_ts:
                new_bias = ev["bias"]
                if new_bias == pending_bias:
                    pending_count += 1
                else:
                    pending_bias = new_bias
                    pending_count = 1
                if "BOS" in ev["type"] or pending_count >= 2:
                    struct_bias = pending_bias

        if struct_bias == "bullish":
            if trend in ("bullish", "weak_bullish"):
                return "bullish", "strong"
            if trend == "weak_bearish":
                return "bullish", "conditional"
            return "neutral", "none"

        if struct_bias == "bearish":
            if trend in ("bearish", "weak_bearish"):
                return "bearish", "strong"
            if trend == "weak_bullish":
                return "bearish", "conditional"
            return "neutral", "none"

        return "neutral", "none"

    def scan(self, df_htf: pd.DataFrame, df_ltf: pd.DataFrame,
             htf_structure: List[Dict[str, Any]], htf_obs: List[Dict[str, Any]],
             ltf_swings: List[Dict[str, Any]], ltf_structure: List[Dict[str, Any]],
             ltf_obs: List[Dict[str, Any]], ltf_fvgs: List[Dict[str, Any]],
             daily_trades_count: int, daily_pnl: float, balance: float,
             equity_peak: float, last_trade_idx: int) -> Optional[Dict[str, Any]]:

        i = len(df_ltf) - 1
        ts = df_ltf.index[i]
        day = ts.date()
        high = float(df_ltf["High"].iloc[i])
        low = float(df_ltf["Low"].iloc[i])
        close = float(df_ltf["Close"].iloc[i])

        current_dd = ((equity_peak - balance) / equity_peak * 100) if equity_peak > 0 else 0.0
        if current_dd >= self.cfg.max_dd_pct:
            log.info("DD guard active: %.1f%% ≥ %.1f%%", current_dd, self.cfg.max_dd_pct)
            return None

        if daily_trades_count >= self.cfg.max_daily_trades:
            return None
        if daily_pnl < -(balance * self.cfg.max_daily_loss_pct / 100):
            return None

        in_kz, kz_name = self.engine.is_kill_zone(ts)
        if not in_kz:
            return None

        bias, confidence = self.get_current_bias(htf_structure, df_htf, df_ltf)
        if bias == "neutral":
            return None

        asian = self.engine.get_asian_range(df_ltf, day)
        asian_bias_aligned = False
        if asian:
            if bias == "bullish" and close > asian["mid"]:
                asian_bias_aligned = True
            elif bias == "bearish" and close < asian["mid"]:
                asian_bias_aligned = True

        if daily_trades_count >= 1 and last_trade_idx > 0:
            if not any(last_trade_idx < ev["index"] <= i for ev in ltf_structure):
                return None

        confluences: List[str] = []
        score = 0
        has_core_trigger = False

        active_ob = self._find_active_ob(ltf_obs, i, close)
        if active_ob and ((bias == "bullish" and active_ob["type"] == "bullish") or (bias == "bearish" and active_ob["type"] == "bearish")):
            confluences.append(f"OB_{active_ob['type']}")
            score += 1 if active_ob["type"] == "bullish" else 2

        active_fvg = self._find_active_fvg(ltf_fvgs, i, close)
        if active_fvg and not (active_fvg["type"] == "bearish" and bias == "bullish"):
            confluences.append(f"FVG_{active_fvg['type']}")
            score += 2

        liq = self.engine.find_liquidity_levels(df_ltf, i)
        if self.engine.check_liquidity_sweep(df_ltf, i, liq, bias):
            confluences.append("LIQ_SWEEP")
            score += 3
            has_core_trigger = True

        has_displacement = self.engine.check_displacement(df_ltf, i, bias)
        if has_displacement:
            confluences.append("DISPLACEMENT")
            score += 3
            has_core_trigger = True

        prev_day = self.engine.get_previous_day_hl(df_ltf, i)
        if prev_day and self.engine.check_stop_hunt(df_ltf, i, prev_day, bias):
            confluences.append("STOP_HUNT")
            score += 3
            has_core_trigger = True

        if asian:
            recent_lows = [float(df_ltf["Low"].iloc[j]) for j in range(max(0, i - 3), i + 1)]
            recent_highs = [float(df_ltf["High"].iloc[j]) for j in range(max(0, i - 3), i + 1)]
            if bias == "bullish" and min(recent_lows) < asian["low"] and close > asian["mid"]:
                confluences.append("ASIAN_SWEEP")
                score += 2
                has_core_trigger = True
            elif bias == "bearish" and max(recent_highs) > asian["high"] and close < asian["mid"]:
                confluences.append("ASIAN_SWEEP")
                score += 2
                has_core_trigger = True

        if asian_bias_aligned:
            score += 1

        if not has_core_trigger:
            return None
        if confidence == "conditional" and not any(c in confluences for c in ("STOP_HUNT", "DISPLACEMENT", "ASIAN_SWEEP")):
            return None
        if len(confluences) < self.cfg.min_confluences:
            return None
        if score < self.cfg.min_confluence_score:
            return None

        candle_range = high - low
        if candle_range > 0:
            if bias == "bullish" and (close - low) / candle_range < 0.35:
                return None
            if bias == "bearish" and (high - close) / candle_range < 0.35:
                return None

        sl_buffer = self.cfg.sl_buffer_pips * self.cfg.pip_size
        recent_sh = [s for s in ltf_swings if s["type"] == "SH" and s["index"] < i and s["index"] > i - 40]
        recent_sl = [s for s in ltf_swings if s["type"] == "SL" and s["index"] < i and s["index"] > i - 40]

        entry_mode = "confirmation"
        limit_price: Optional[float] = None

        if bias == "bullish" and recent_sl:
            swing_low = min(s["price"] for s in recent_sl[-3:])
            sl_price = swing_low - sl_buffer

            if recent_sh:
                swing_high = max(s["price"] for s in recent_sh[-3:])
                if self.engine.premium_discount(close, swing_high, swing_low) == "premium":
                    return None
                confluences.append("DISCOUNT")

                if self.cfg.limit_entry_enabled and "LIQ_SWEEP" in confluences:
                    ote_bot, ote_top = self.engine.ote_zone(swing_high, swing_low, "long")
                    if ote_bot < close:
                        ote_entry = (ote_bot + ote_top) / 2
                        if sl_price < ote_entry < close:
                            limit_price = ote_entry
                            entry_mode = "limit"
                            confluences.append("LIMIT_OTE")

                if self.engine.is_in_ote(close, swing_high, swing_low, "long"):
                    confluences.append("OTE")

            entry_price = limit_price if entry_mode == "limit" else close
            risk = entry_price - sl_price
            if risk <= 0 or risk < 2.0:
                return None
            tp_price = entry_price + risk * self.cfg.min_rr
            direction = "long"

        elif bias == "bearish" and recent_sh:
            swing_high = max(s["price"] for s in recent_sh[-3:])
            sl_price = swing_high + sl_buffer

            if recent_sl:
                swing_low = min(s["price"] for s in recent_sl[-3:])
                if self.engine.premium_discount(close, swing_high, swing_low) == "discount":
                    return None
                confluences.append("PREMIUM")

                if self.cfg.limit_entry_enabled and "LIQ_SWEEP" in confluences:
                    ote_bot, ote_top = self.engine.ote_zone(swing_high, swing_low, "short")
                    if ote_top > close:
                        ote_entry = (ote_bot + ote_top) / 2
                        if close < ote_entry < sl_price:
                            limit_price = ote_entry
                            entry_mode = "limit"
                            confluences.append("LIMIT_OTE")

                if self.engine.is_in_ote(close, swing_high, swing_low, "short"):
                    confluences.append("OTE")

            entry_price = limit_price if entry_mode == "limit" else close
            risk = sl_price - entry_price
            if risk <= 0 or risk < 2.0:
                return None
            tp_price = entry_price - risk * self.cfg.min_rr
            direction = "short"
        else:
            return None

        rr = abs(tp_price - entry_price) / abs(entry_price - sl_price)
        if rr < self.cfg.min_rr or rr > self.cfg.max_rr:
            return None

        risk_amount = balance * (self.cfg.risk_per_trade_pct / 100)
        risk_pips = abs(entry_price - sl_price) / self.cfg.pip_size
        lot_size = round(risk_amount / (risk_pips * 10), 2)
        lot_size = max(0.01, min(lot_size, 5.0))

        return {
            "direction": direction,
            "entry_price": round(entry_price, 2),
            "sl_price": round(sl_price, 2),
            "tp_price": round(tp_price, 2),
            "lot_size": lot_size,
            "confluences": confluences,
            "kill_zone": kz_name,
            "score": score,
            "entry_mode": entry_mode,
            "has_displacement": has_displacement,
            "rr": round(rr, 1),
            "bias": bias,
            "confidence": confidence,
        }


# =============================================================================
# TRADE MANAGER
# =============================================================================
class TradeManager:
    def __init__(self, cfg: Config, mt5mgr: MT5Manager, tg: Telegram):
        self.cfg = cfg
        self.mt5 = mt5mgr
        self.tg = tg

    def manage(self, position: Any, state: Dict[str, Any]) -> Dict[str, Any]:
        entry_price = float(state["entry_price"])
        original_sl = float(state["original_sl"])
        risk = abs(entry_price - original_sl)
        if risk <= 0:
            return state

        tick = self.mt5.get_tick()
        if tick is None:
            return state

        is_long = position.type == mt5.ORDER_TYPE_BUY
        current_r = ((tick - entry_price) / risk) if is_long else ((entry_price - tick) / risk)

        if self.cfg.partial_tp_enabled and not state.get("partial_closed", False) and current_r >= self.cfg.partial_tp_r:
            partial_lot = round(float(state["original_lot"]) * self.cfg.partial_tp_pct, 2)
            partial_lot = max(0.01, partial_lot)

            if self.mt5.close_partial(position, partial_lot):
                new_sl = entry_price + risk * self.cfg.be_lock_r if is_long else entry_price - risk * self.cfg.be_lock_r
                self.mt5.modify_sl(position, new_sl)
                state["partial_closed"] = True
                state["current_sl"] = new_sl
                self.tg.trade_update("Partial TP Taken (50%)", f"R: {current_r:.1f}R | SL → BE ({new_sl:.2f})")
                log.info("Partial TP: closed %.2f lot @ %.1fR, SL → %.2f", partial_lot, current_r, new_sl)

        if state.get("has_displacement", False) and current_r >= self.cfg.trail_trigger_r:
            if is_long:
                trail_sl = entry_price + risk * (current_r - self.cfg.trail_step_r)
                if trail_sl > state.get("current_sl", position.sl):
                    self.mt5.modify_sl(position, trail_sl)
                    state["current_sl"] = trail_sl
                    log.info("Trail SL → %.2f (R=%.1f)", trail_sl, current_r)
            else:
                trail_sl = entry_price - risk * (current_r - self.cfg.trail_step_r)
                if trail_sl < state.get("current_sl", position.sl):
                    self.mt5.modify_sl(position, trail_sl)
                    state["current_sl"] = trail_sl
                    log.info("Trail SL → %.2f (R=%.1f)", trail_sl, current_r)

        elif not state.get("partial_closed", False) and current_r >= self.cfg.be_trigger_r:
            if is_long and position.sl < entry_price:
                new_sl = entry_price + risk * self.cfg.be_lock_r
                self.mt5.modify_sl(position, new_sl)
                state["current_sl"] = new_sl
                log.info("BE triggered → SL=%.2f", new_sl)
            elif not is_long and position.sl > entry_price:
                new_sl = entry_price - risk * self.cfg.be_lock_r
                self.mt5.modify_sl(position, new_sl)
                state["current_sl"] = new_sl
                log.info("BE triggered → SL=%.2f", new_sl)

        return state


# =============================================================================
# BOT
# =============================================================================
class ICTBot:
    def __init__(self):
        self.cfg = Config()
        global log
        log = setup_logging(self.cfg)

        self.tg = Telegram(self.cfg.telegram_token, self.cfg.telegram_chat_id)
        self.mt5 = MT5Manager(self.cfg)
        self.engine = ICTEngine(self.cfg)
        self.scanner = LiveScanner(self.cfg, self.engine)
        self.trade_mgr = TradeManager(self.cfg, self.mt5, self.tg)
        self.state = load_state(self.cfg.state_file)
        self.last_bar_time = None

    def _print_banner(self) -> None:
        print("╔══════════════════════════════════════════════════════════╗")
        print("║   ICT XAU/USD LIVE BOT v1.1                             ║")
        print("║   Refactored · Safer Config · Cleaner State             ║")
        print("║   Based on Backtest v3.6 logic                          ║")
        print("╚══════════════════════════════════════════════════════════╝")

    def _build_context(self) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]]:
        df_htf = self.mt5.get_bars(self.cfg.tf_htf, self.cfg.htf_bars)
        df_ltf = self.mt5.get_bars(self.cfg.tf_ltf, self.cfg.ltf_bars)
        if df_htf is None or df_ltf is None:
            return None

        htf_swings = self.engine.find_swing_points(df_htf)
        htf_structure = self.engine.analyze_structure(htf_swings)
        htf_obs = self.engine.find_order_blocks(df_htf, htf_structure)

        ltf_swings = self.engine.find_swing_points(df_ltf)
        ltf_structure = self.engine.analyze_structure(ltf_swings)
        ltf_obs = self.engine.find_order_blocks(df_ltf, ltf_structure)
        ltf_fvgs = self.engine.find_fvg(df_ltf)

        return df_htf, df_ltf, htf_structure, htf_obs, ltf_swings, ltf_structure, ltf_obs, ltf_fvgs

    def _handle_existing_trade(self) -> None:
        positions = self.mt5.get_my_positions()
        if positions:
            pos = positions[0]
            if self.state.get("open_trade"):
                self.state["open_trade"] = self.trade_mgr.manage(pos, self.state["open_trade"])
                save_state(self.cfg.state_file, self.state)
        elif self.state.get("open_trade"):
            log.info("Position closed (SL/TP/manual by MT5)")
            self.tg.trade_update("Position Closed", "SL / TP / manual close.")
            self.state["open_trade"] = None
            save_state(self.cfg.state_file, self.state)

    def _execute_signal(self, signal: Dict[str, Any], df_ltf: pd.DataFrame) -> None:
        log.info(
            "🎯 SIGNAL: %s | Entry=%s | SL=%s | TP=%s | R:R=%s | Mode=%s | Score=%s | %s",
            signal["direction"].upper(), signal["entry_price"], signal["sl_price"],
            signal["tp_price"], signal["rr"], signal["entry_mode"], signal["score"],
            signal["confluences"],
        )

        self.tg.signal(
            signal["direction"], signal["entry_price"], signal["sl_price"], signal["tp_price"],
            signal["confluences"], signal["kill_zone"], signal["score"], signal["entry_mode"], signal["rr"]
        )

        if signal["entry_mode"] == "limit":
            expiry_min = self.cfg.limit_wait_bars * 15
            ticket = self.mt5.open_limit(
                signal["direction"], signal["lot_size"], signal["entry_price"],
                signal["sl_price"], signal["tp_price"], expiry_min
            )
            if ticket is not None:
                self.state["pending_limit"] = {
                    "entry_price": signal["entry_price"],
                    "original_sl": signal["sl_price"],
                    "original_lot": signal["lot_size"],
                    "has_displacement": signal["has_displacement"],
                    "partial_closed": False,
                    "current_sl": signal["sl_price"],
                }
        else:
            ticket = self.mt5.open_market(
                signal["direction"], signal["lot_size"], signal["sl_price"], signal["tp_price"]
            )
            if ticket is not None:
                self.state["open_trade"] = {
                    "entry_price": signal["entry_price"],
                    "original_sl": signal["sl_price"],
                    "original_lot": signal["lot_size"],
                    "has_displacement": signal["has_displacement"],
                    "partial_closed": False,
                    "current_sl": signal["sl_price"],
                }

        if ticket is not None:
            today = str(datetime.now(UTC).date())
            self.state["daily_trades"][today] = int(self.state["daily_trades"].get(today, 0)) + 1
            self.state["last_trade_idx"] = len(df_ltf) - 1
            save_state(self.cfg.state_file, self.state)
            self.tg.trade_update(
                f"{'Limit' if signal['entry_mode'] == 'limit' else 'Market'} Order Placed",
                f"Ticket: #{ticket} | Lot: {signal['lot_size']}"
            )

    def _process_bar(self) -> None:
        context = self._build_context()
        if context is None:
            log.warning("Data pull failed, skipping this bar.")
            return

        (df_htf, df_ltf, htf_structure, htf_obs, ltf_swings, ltf_structure, ltf_obs, ltf_fvgs) = context

        self._handle_existing_trade()

        pending = self.mt5.get_my_pending()
        positions = self.mt5.get_my_positions()

        if not positions and pending:
            return

        if positions:
            return

        today = str(datetime.now(UTC).date())
        daily_count = int(self.state["daily_trades"].get(today, 0))
        daily_pnl = float(self.state["daily_pnl"].get(today, 0.0))
        balance = self.mt5.get_balance()
        self.state["equity_peak"] = max(float(self.state["equity_peak"]), balance)

        signal = self.scanner.scan(
            df_htf, df_ltf, htf_structure, htf_obs, ltf_swings, ltf_structure,
            ltf_obs, ltf_fvgs, daily_count, daily_pnl, balance,
            float(self.state["equity_peak"]), int(self.state["last_trade_idx"])
        )
        if signal is None:
            return

        self._execute_signal(signal, df_ltf)

    def _tick(self) -> None:
        df = self.mt5.get_bars(self.cfg.tf_ltf, 2)
        if df is None or len(df) < 2:
            return

        latest_closed = df.index[-2]
        if self.last_bar_time == latest_closed:
            return

        self.last_bar_time = latest_closed
        log.info("── New M15 bar: %s ──", latest_closed)
        self._process_bar()

    def run(self) -> None:
        self._print_banner()

        if not self.mt5.connect():
            log.error("Cannot connect to MT5. Exiting.")
            return

        balance = self.mt5.get_balance()
        if float(self.state["equity_peak"]) == 0:
            self.state["equity_peak"] = balance

        mode = "DRY RUN" if self.cfg.dry_run else "LIVE"
        log.info("Bot started — %s | Balance: $%.2f | KZ: %s", mode, balance, self.cfg.kill_zone_mode)
        self.tg.send(
            f"🤖 <b>ICT Bot Started</b>\n"
            f"Mode: {mode}\n"
            f"Balance: ${balance:,.2f}\n"
            f"Risk: {self.cfg.risk_per_trade_pct}% per trade\n"
            f"Kill Zones: {self.cfg.kill_zone_mode}"
        )

        try:
            while True:
                self._tick()
                time.sleep(self.cfg.scan_interval_sec)
        except KeyboardInterrupt:
            log.info("Bot stopped by user.")
            self.tg.send("🛑 <b>Bot stopped</b> (manual)")
        finally:
            save_state(self.cfg.state_file, self.state)
            mt5.shutdown()


if __name__ == "__main__":
    bot = ICTBot()
    bot.run()
