"""
=============================================================================
ICT XAU/USD LIVE TRADING BOT v1.0
=============================================================================
Full auto execution on MT5 with Telegram notifications.
Ported from backtest engine v3.6 (PF 2.1 | WR 59% | 70.4% return over 4yr).

Logic: ICT Smart Money Concepts
  - Market Structure (BOS/CHOCH) on H1
  - Order Blocks, FVG, Liquidity Sweeps on M15
  - Kill Zones (London, NY AM, NY PM)
  - Dual entry mode: Confirmation + Limit (OTE)
  - Trade management: Partial TP → BE → Trail

Author : Claude × Riyan
Date   : 2026-04
=============================================================================
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import requests
import json
import time
import os
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# =============================================================================
# CONFIGURATION — EDIT THIS SECTION
# =============================================================================
@dataclass
class Config:
    # ── MT5 ──
    symbol: str = "XAUUSD"
    magic_number: int = 360360       # unique ID for this bot's orders
    deviation: int = 30              # max slippage in points

    # ── Telegram ──
    telegram_token: str = "7268820797:AAEuC9k2C7fuVEJihEzHybXI5cTt95KofiU"
    telegram_chat_id: str = "789297530"

    # ── Timeframes ──
    tf_htf: int = mt5.TIMEFRAME_H1
    tf_ltf: int = mt5.TIMEFRAME_M15
    htf_bars: int = 500              # rolling window for H1 (enough for EMA200)
    ltf_bars: int = 500              # rolling window for M15

    # ── Risk Management ──
    account_balance_override: float = 0  # 0 = use MT5 account balance
    risk_per_trade_pct: float = 1.0
    default_sl_pips: float = 150.0
    min_rr: float = 2.0
    max_rr: float = 6.0
    max_daily_trades: int = 2
    max_daily_loss_pct: float = 1.5
    max_dd_pct: float = 10.0

    # ── ICT Parameters (matched to backtest v3.6) ──
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

    # ── Kill Zones (UTC hours) ──
    kill_zone_mode: str = "all"      # "all" = London + NY_AM + NY_PM
    london_open: int = 7
    london_close: int = 10
    ny_am_open: int = 12
    ny_am_close: int = 15
    ny_pm_open: int = 15
    ny_pm_close: int = 17

    # ── Asian Range (UTC) ──
    asian_open: int = 0
    asian_close: int = 6

    # ── Trade Management ──
    be_trigger_r: float = 1.5
    be_lock_r: float = 0.3
    trail_trigger_r: float = 2.0
    trail_step_r: float = 0.5
    partial_tp_enabled: bool = True
    partial_tp_r: float = 1.5
    partial_tp_pct: float = 0.5      # close 50%

    # ── Limit Entry ──
    limit_entry_enabled: bool = True
    limit_wait_bars: int = 4         # expiry = 4 × 15min = 60min

    # ── Pip ──
    pip_size: float = 0.10
    contract_size: float = 1.0

    # ── Bot ──
    dry_run: bool = True             # True = log only, no real orders
    scan_interval_sec: int = 5       # check interval within wait loop
    state_file: str = "bot_state.json"
    log_file: str = "bot_log.txt"


# =============================================================================
# LOGGING
# =============================================================================
def setup_logging(cfg: Config):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(cfg.log_file, encoding="utf-8"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("ICTBot")

log = None  # initialized in main()


# =============================================================================
# TELEGRAM NOTIFIER
# =============================================================================
class Telegram:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
        self.enabled = token != "YOUR_BOT_TOKEN_HERE"

    def send(self, text: str, parse_mode: str = "HTML"):
        if not self.enabled:
            return
        try:
            resp = requests.post(
                f"{self.base_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode
                },
                timeout=10
            )
            if not resp.ok:
                log.warning(f"Telegram error: {resp.text}")
        except Exception as e:
            log.warning(f"Telegram send failed: {e}")

    def signal(self, direction, entry, sl, tp, confluences, kz, score, mode, rr):
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

    def trade_update(self, action, details=""):
        self.send(f"📋 <b>{action}</b>\n{details}")

    def daily_summary(self, trades_today, pnl_today, balance):
        msg = (
            f"📊 <b>Daily Summary</b>\n"
            f"Trades : {trades_today}\n"
            f"P&L    : ${pnl_today:+.2f}\n"
            f"Balance: ${balance:,.2f}"
        )
        self.send(msg)


# =============================================================================
# MT5 MANAGER
# =============================================================================
class MT5Manager:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.symbol = None  # resolved symbol name

    def connect(self) -> bool:
        if not mt5.initialize():
            log.error(f"MT5 init failed: {mt5.last_error()}")
            return False

        # Auto-detect gold symbol
        candidates = [
            self.cfg.symbol, "XAUUSDm", "XAUUSD.raw", "XAUUSD.a",
            "XAUUSD.ecn", "XAUUSDc", "Gold", "GOLD", "GOLDm",
            "XAUUSD.pro", "XAUUSD.std", "XAUUSD#",
        ]
        for sym in candidates:
            info = mt5.symbol_info(sym)
            if info is not None:
                self.symbol = sym
                mt5.symbol_select(sym, True)
                log.info(f"MT5 connected — symbol: {sym}")
                return True

        # Fallback: search
        all_syms = mt5.symbols_get()
        if all_syms:
            for s in all_syms:
                if "XAU" in s.name.upper() and "USD" in s.name.upper():
                    self.symbol = s.name
                    mt5.symbol_select(s.name, True)
                    log.info(f"MT5 connected — symbol: {s.name}")
                    return True

        log.error("No gold symbol found on broker")
        return False

    def get_bars(self, timeframe: int, count: int) -> Optional[pd.DataFrame]:
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, count)
        if rates is None or len(rates) == 0:
            log.warning(f"No data for {self.symbol} TF={timeframe}")
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df.rename(columns={
            "open": "Open", "high": "High",
            "low": "Low", "close": "Close",
            "tick_volume": "Volume"
        }, inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]]

    def get_balance(self) -> float:
        if self.cfg.account_balance_override > 0:
            return self.cfg.account_balance_override
        info = mt5.account_info()
        return info.balance if info else 10000.0

    def get_equity(self) -> float:
        info = mt5.account_info()
        return info.equity if info else self.get_balance()

    def get_tick(self) -> Optional[float]:
        tick = mt5.symbol_info_tick(self.symbol)
        return tick.bid if tick else None

    def get_my_positions(self) -> list:
        """Get positions opened by this bot (by magic number)."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return []
        return [p for p in positions if p.magic == self.cfg.magic_number]

    def get_my_pending(self) -> list:
        """Get pending orders by this bot."""
        orders = mt5.orders_get(symbol=self.symbol)
        if orders is None:
            return []
        return [o for o in orders if o.magic == self.cfg.magic_number]

    def open_market(self, direction: str, lot: float, sl: float, tp: float,
                    comment: str = "ICT_v1") -> Optional[int]:
        """Open market order. Returns ticket or None."""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            log.error("Cannot get tick for market order")
            return None

        if direction == "long":
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": round(sl, 2),
            "tp": round(tp, 2),
            "deviation": self.cfg.deviation,
            "magic": self.cfg.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
        }

        if self.cfg.dry_run:
            log.info(f"[DRY RUN] Market {direction} {lot} lot @ {price:.2f} "
                     f"SL={sl:.2f} TP={tp:.2f}")
            return -1  # fake ticket

        result = mt5.order_send(request)
        if result is None:
            log.error(f"order_send returned None: {mt5.last_error()}")
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"Order failed: {result.retcode} — {result.comment}")
            return None

        log.info(f"Market {direction} opened: ticket={result.order} "
                 f"price={result.price} lot={lot}")
        return result.order

    def open_limit(self, direction: str, lot: float, price: float,
                   sl: float, tp: float, expiry_minutes: int = 60,
                   comment: str = "ICT_v1_LMT") -> Optional[int]:
        """Place limit order with expiry."""
        if direction == "long":
            order_type = mt5.ORDER_TYPE_BUY_LIMIT
        else:
            order_type = mt5.ORDER_TYPE_SELL_LIMIT

        expiry_time = datetime.utcnow() + timedelta(minutes=expiry_minutes)

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

        if self.cfg.dry_run:
            log.info(f"[DRY RUN] Limit {direction} {lot} lot @ {price:.2f} "
                     f"SL={sl:.2f} TP={tp:.2f} exp={expiry_minutes}min")
            return -1

        result = mt5.order_send(request)
        if result is None:
            log.error(f"Limit order_send None: {mt5.last_error()}")
            return None
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            log.error(f"Limit order failed: {result.retcode} — {result.comment}")
            return None

        log.info(f"Limit {direction} placed: ticket={result.order} @ {price:.2f}")
        return result.order

    def close_partial(self, position, lot_to_close: float,
                      comment: str = "Partial_TP") -> bool:
        """Close partial lot of an open position."""
        if position.type == mt5.ORDER_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            tick = mt5.symbol_info_tick(self.symbol)
            price = tick.bid if tick else 0
        else:
            close_type = mt5.ORDER_TYPE_BUY
            tick = mt5.symbol_info_tick(self.symbol)
            price = tick.ask if tick else 0

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": round(lot_to_close, 2),
            "type": close_type,
            "position": position.ticket,
            "price": price,
            "deviation": self.cfg.deviation,
            "magic": self.cfg.magic_number,
            "comment": comment,
        }

        if self.cfg.dry_run:
            log.info(f"[DRY RUN] Partial close {lot_to_close} lot "
                     f"of ticket {position.ticket}")
            return True

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"Partial close OK: {lot_to_close} lot of #{position.ticket}")
            return True
        log.error(f"Partial close failed: {result}")
        return False

    def modify_sl(self, position, new_sl: float) -> bool:
        """Modify SL of an open position."""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": self.symbol,
            "position": position.ticket,
            "sl": round(new_sl, 2),
            "tp": round(position.tp, 2),
        }

        if self.cfg.dry_run:
            log.info(f"[DRY RUN] Modify SL → {new_sl:.2f} for #{position.ticket}")
            return True

        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info(f"SL modified → {new_sl:.2f} for #{position.ticket}")
            return True
        log.error(f"SL modify failed: {result}")
        return False


# =============================================================================
# ICT ENGINE (exact port from backtest v3.6)
# =============================================================================
class ICTEngine:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def find_swing_points(self, df: pd.DataFrame) -> list:
        swings = []
        n = self.cfg.swing_strength
        highs = df["High"].values
        lows = df["Low"].values
        for i in range(n, len(df) - n):
            if all(highs[i] >= highs[i-j] for j in range(1, n+1)) and \
               all(highs[i] >= highs[i+j] for j in range(1, n+1)):
                swings.append({"index": i, "price": highs[i],
                               "timestamp": df.index[i], "type": "SH"})
            if all(lows[i] <= lows[i-j] for j in range(1, n+1)) and \
               all(lows[i] <= lows[i+j] for j in range(1, n+1)):
                swings.append({"index": i, "price": lows[i],
                               "timestamp": df.index[i], "type": "SL"})
        return swings

    def analyze_structure(self, swings: list) -> list:
        events = []
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
                    events.append({"type": etype, "index": curr["index"],
                                   "price": curr["price"],
                                   "timestamp": curr["timestamp"],
                                   "bias": "bullish"})
                    bias = "bullish"
                else:
                    if bias == "bearish":
                        events.append({"type": "BOS_BEAR", "index": curr["index"],
                                       "price": curr["price"],
                                       "timestamp": curr["timestamp"],
                                       "bias": "bearish"})
            elif curr["type"] == "SL":
                if curr["price"] < prev["price"]:
                    etype = "BOS_BEAR" if bias == "bearish" else "CHOCH_BEAR"
                    events.append({"type": etype, "index": curr["index"],
                                   "price": curr["price"],
                                   "timestamp": curr["timestamp"],
                                   "bias": "bearish"})
                    bias = "bearish"
                else:
                    if bias == "bullish":
                        events.append({"type": "BOS_BULL", "index": curr["index"],
                                       "price": curr["price"],
                                       "timestamp": curr["timestamp"],
                                       "bias": "bullish"})
        return events

    def find_order_blocks(self, df, structure_events) -> list:
        obs = []
        opens = df["Open"].values
        closes = df["Close"].values
        highs = df["High"].values
        lows = df["Low"].values
        for event in structure_events:
            idx = event["index"]
            if idx < 3:
                continue
            if "BULL" in event["type"]:
                for j in range(idx-1, max(idx-6, 0), -1):
                    if closes[j] < opens[j]:
                        obs.append({"index": j, "timestamp": df.index[j],
                                    "top": opens[j], "bottom": lows[j],
                                    "type": "bullish", "valid": True})
                        break
            elif "BEAR" in event["type"]:
                for j in range(idx-1, max(idx-6, 0), -1):
                    if closes[j] > opens[j]:
                        obs.append({"index": j, "timestamp": df.index[j],
                                    "top": highs[j], "bottom": closes[j],
                                    "type": "bearish", "valid": True})
                        break
        return obs

    def find_fvg(self, df) -> list:
        fvgs = []
        highs = df["High"].values
        lows = df["Low"].values
        min_gap = self.cfg.fvg_min_size_pips * self.cfg.pip_size
        for i in range(2, len(df)):
            gap_bull = lows[i] - highs[i-2]
            if gap_bull > min_gap:
                fvgs.append({"index": i, "timestamp": df.index[i],
                             "top": lows[i], "bottom": highs[i-2],
                             "type": "bullish"})
            gap_bear = lows[i-2] - highs[i]
            if gap_bear > min_gap:
                fvgs.append({"index": i, "timestamp": df.index[i],
                             "top": lows[i-2], "bottom": highs[i],
                             "type": "bearish"})
        return fvgs

    def find_liquidity_levels(self, df, current_idx) -> dict:
        lookback = min(self.cfg.liquidity_lookback, current_idx)
        start = current_idx - lookback
        window = df.iloc[start:current_idx]
        prev_highs, prev_lows = [], []
        highs = window["High"].values
        lows = window["Low"].values
        n = 2
        for i in range(n, len(window) - n):
            if all(highs[i] >= highs[i-j] for j in range(1, n+1)) and \
               all(highs[i] >= highs[i+j] for j in range(1, n+1)):
                prev_highs.append(highs[i])
            if all(lows[i] <= lows[i-j] for j in range(1, n+1)) and \
               all(lows[i] <= lows[i+j] for j in range(1, n+1)):
                prev_lows.append(lows[i])
        return {
            "buy_side_liquidity": sorted(prev_highs)[-3:] if prev_highs else [],
            "sell_side_liquidity": sorted(prev_lows)[:3] if prev_lows else [],
        }

    def is_kill_zone(self, timestamp) -> Tuple[bool, str]:
        hour = timestamp.hour
        mode = self.cfg.kill_zone_mode
        london = self.cfg.london_open <= hour < self.cfg.london_close
        ny_am = self.cfg.ny_am_open <= hour < self.cfg.ny_am_close
        ny_pm = self.cfg.ny_pm_open <= hour < self.cfg.ny_pm_close

        if mode == "all":
            if london: return True, "London"
            if ny_am: return True, "NY_AM"
            if ny_pm: return True, "NY_PM"
        elif mode == "ny_am_only":
            if ny_am: return True, "NY_AM"
        elif mode == "london_ny":
            if london: return True, "London"
            if ny_am: return True, "NY_AM"
            if ny_pm: return True, "NY_PM"
        return False, "Off_Session"

    def get_asian_range(self, df, current_date) -> Optional[dict]:
        asian_data = df[
            (df.index.date == current_date) &
            (df.index.hour >= self.cfg.asian_open) &
            (df.index.hour < self.cfg.asian_close)
        ]
        if len(asian_data) < 4:
            return None
        return {
            "high": asian_data["High"].max(),
            "low": asian_data["Low"].min(),
            "mid": (asian_data["High"].max() + asian_data["Low"].min()) / 2,
        }

    def premium_discount(self, price, swing_high, swing_low) -> str:
        eq = (swing_high + swing_low) / 2
        return "premium" if price > eq else "discount"

    def ote_zone(self, swing_high, swing_low, direction) -> Tuple[float, float]:
        rng = swing_high - swing_low
        if direction == "long":
            return (swing_high - rng * self.cfg.ote_fib_high,
                    swing_high - rng * self.cfg.ote_fib_low)
        else:
            return (swing_low + rng * self.cfg.ote_fib_low,
                    swing_low + rng * self.cfg.ote_fib_high)

    def is_in_ote(self, price, swing_high, swing_low, direction) -> bool:
        ote_bot, ote_top = self.ote_zone(swing_high, swing_low, direction)
        return ote_bot <= price <= ote_top

    def check_displacement(self, df, idx, direction) -> bool:
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

    def get_previous_day_hl(self, df, idx) -> Optional[dict]:
        current_date = df.index[idx].date()
        prev_day_data = df[df.index.date < current_date]
        if len(prev_day_data) == 0:
            return None
        last_date = prev_day_data.index[-1].date()
        day_data = prev_day_data[prev_day_data.index.date == last_date]
        if len(day_data) == 0:
            return None
        return {
            "high": day_data["High"].max(),
            "low": day_data["Low"].min(),
            "close": day_data["Close"].iloc[-1],
            "open": day_data["Open"].iloc[0],
        }

    def check_stop_hunt(self, df, idx, prev_day, direction) -> bool:
        current_date = df.index[idx].date()
        today_data = df[(df.index.date == current_date) &
                        (df.index <= df.index[idx])]
        if len(today_data) < 3:
            return False
        if direction == "long":
            if today_data["Low"].min() < prev_day["low"]:
                if df["Close"].iloc[idx] > prev_day["low"]:
                    return True
        elif direction == "short":
            if today_data["High"].max() > prev_day["high"]:
                if df["Close"].iloc[idx] < prev_day["high"]:
                    return True
        return False

    def check_liquidity_sweep(self, df, idx, liq, bias) -> bool:
        if idx < 2:
            return False
        prev_high = df["High"].iloc[idx - 1]
        prev_low = df["Low"].iloc[idx - 1]
        curr_close = df["Close"].iloc[idx]
        if bias == "bullish":
            for level in liq["sell_side_liquidity"]:
                if prev_low < level and curr_close > level:
                    return True
        if bias == "bearish":
            for level in liq["buy_side_liquidity"]:
                if prev_high > level and curr_close < level:
                    return True
        return False


# =============================================================================
# LIVE SCANNER — signal generation (port from backtest scan loop)
# =============================================================================
class LiveScanner:
    def __init__(self, cfg: Config, engine: ICTEngine):
        self.cfg = cfg
        self.engine = engine

    def get_current_bias(self, htf_structure, df_htf, df_ltf):
        """Compute bias + confidence for the latest LTF bar."""
        # HTF EMA trend
        ema50 = df_htf["Close"].ewm(span=50, adjust=False).mean()
        ema200 = df_htf["Close"].ewm(span=200, adjust=False).mean()
        price = df_htf["Close"].iloc[-1]
        e50 = ema50.iloc[-1]
        e200 = ema200.iloc[-1]

        if price > e50 and e50 > e200:
            trend = "bullish"
        elif price < e50 and e50 < e200:
            trend = "bearish"
        elif price > e50:
            trend = "weak_bullish"
        elif price < e50:
            trend = "weak_bearish"
        else:
            trend = "neutral"

        # Structure bias (with confirmation — 2 events for CHOCH)
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

        # Combine: structure + EMA → bias + confidence
        if struct_bias == "bullish":
            if trend in ("bullish", "weak_bullish"):
                return "bullish", "strong"
            elif trend == "weak_bearish":
                return "bullish", "conditional"
            else:
                return "neutral", "none"
        elif struct_bias == "bearish":
            if trend in ("bearish", "weak_bearish"):
                return "bearish", "strong"
            elif trend == "weak_bullish":
                return "bearish", "conditional"
            else:
                return "neutral", "none"
        return "neutral", "none"

    def scan(self, df_htf, df_ltf, htf_structure, htf_obs,
             ltf_swings, ltf_structure, ltf_obs, ltf_fvgs,
             daily_trades_count, daily_pnl, balance, equity_peak,
             last_trade_idx) -> Optional[dict]:
        """
        Scan the latest M15 bar for a valid setup.
        Returns trade dict or None.
        """
        i = len(df_ltf) - 1  # latest bar
        ts = df_ltf.index[i]
        day = ts.date()
        high = df_ltf["High"].iloc[i]
        low = df_ltf["Low"].iloc[i]
        close = df_ltf["Close"].iloc[i]
        opn = df_ltf["Open"].iloc[i]

        # ── DD circuit breaker ──
        current_dd = (equity_peak - balance) / equity_peak * 100 if equity_peak > 0 else 0
        if current_dd >= self.cfg.max_dd_pct:
            log.info(f"DD guard active: {current_dd:.1f}% ≥ {self.cfg.max_dd_pct}%")
            return None

        # ── Daily limits ──
        if daily_trades_count >= self.cfg.max_daily_trades:
            return None
        if daily_pnl < -(balance * self.cfg.max_daily_loss_pct / 100):
            return None

        # ── Kill zone ──
        in_kz, kz_name = self.engine.is_kill_zone(ts)
        if not in_kz:
            return None

        # ── Bias ──
        bias, confidence = self.get_current_bias(htf_structure, df_htf, df_ltf)
        if bias == "neutral":
            return None

        # ── Asian range ──
        asian = self.engine.get_asian_range(df_ltf, day)
        asian_bias_aligned = False
        if asian:
            if bias == "bullish" and close > asian["mid"]:
                asian_bias_aligned = True
            elif bias == "bearish" and close < asian["mid"]:
                asian_bias_aligned = True

        # ── 2nd trade: require new structure since last trade ──
        if daily_trades_count >= 1 and last_trade_idx > 0:
            new_structure = any(
                last_trade_idx < ev["index"] <= i for ev in ltf_structure
            )
            if not new_structure:
                return None

        # ── Build confluences ──
        confluences = []
        score = 0
        has_core_trigger = False

        # OB
        active_ob = self._find_active_ob(ltf_obs, i, close)
        if active_ob:
            if (bias == "bullish" and active_ob["type"] == "bullish") or \
               (bias == "bearish" and active_ob["type"] == "bearish"):
                confluences.append(f"OB_{active_ob['type']}")
                score += 1 if active_ob["type"] == "bullish" else 2

        # FVG
        active_fvg = self._find_active_fvg(ltf_fvgs, i, close)
        if active_fvg:
            if not (active_fvg["type"] == "bearish" and bias == "bullish"):
                confluences.append(f"FVG_{active_fvg['type']}")
                score += 2

        # Liquidity sweep — CORE
        liq = self.engine.find_liquidity_levels(df_ltf, i)
        if self.engine.check_liquidity_sweep(df_ltf, i, liq, bias):
            confluences.append("LIQ_SWEEP")
            score += 3
            has_core_trigger = True

        # Displacement — CORE
        has_displacement = self.engine.check_displacement(df_ltf, i, bias)
        if has_displacement:
            confluences.append("DISPLACEMENT")
            score += 3
            has_core_trigger = True

        # Stop hunt — CORE
        prev_day = self.engine.get_previous_day_hl(df_ltf, i)
        if prev_day:
            if self.engine.check_stop_hunt(df_ltf, i, prev_day, bias):
                confluences.append("STOP_HUNT")
                score += 3
                has_core_trigger = True

        # Asian sweep — CORE
        if asian:
            recent_lows = [df_ltf["Low"].iloc[j] for j in range(max(0, i-3), i+1)]
            recent_highs = [df_ltf["High"].iloc[j] for j in range(max(0, i-3), i+1)]
            if bias == "bullish" and min(recent_lows) < asian["low"] \
               and close > asian["mid"]:
                confluences.append("ASIAN_SWEEP")
                score += 2
                has_core_trigger = True
            elif bias == "bearish" and max(recent_highs) > asian["high"] \
                 and close < asian["mid"]:
                confluences.append("ASIAN_SWEEP")
                score += 2
                has_core_trigger = True

        if asian_bias_aligned:
            score += 1

        # ── Gates ──
        if not has_core_trigger:
            return None
        if confidence == "conditional":
            qualifying = any(c in confluences
                             for c in ("STOP_HUNT", "DISPLACEMENT", "ASIAN_SWEEP"))
            if not qualifying:
                return None
        if len(confluences) < self.cfg.min_confluences:
            return None
        if score < self.cfg.min_confluence_score:
            return None

        # ── Candle confirmation ──
        candle_range = high - low
        if candle_range > 0:
            if bias == "bullish" and (close - low) / candle_range < 0.35:
                return None
            if bias == "bearish" and (high - close) / candle_range < 0.35:
                return None

        # ── Entry calculation ──
        sl_buffer = self.cfg.sl_buffer_pips * self.cfg.pip_size
        recent_sh = [s for s in ltf_swings
                     if s["type"] == "SH" and s["index"] < i and s["index"] > i - 40]
        recent_sl = [s for s in ltf_swings
                     if s["type"] == "SL" and s["index"] < i and s["index"] > i - 40]

        entry_mode = "confirmation"
        limit_price = None

        if bias == "bullish" and recent_sl:
            swing_low = min(s["price"] for s in recent_sl[-3:])
            sl_price = swing_low - sl_buffer

            if recent_sh:
                swing_high = max(s["price"] for s in recent_sh[-3:])
                zone = self.engine.premium_discount(close, swing_high, swing_low)
                if zone == "premium":
                    return None
                confluences.append("DISCOUNT")

                # Limit entry: OTE + LIQ_SWEEP
                if self.cfg.limit_entry_enabled and "LIQ_SWEEP" in confluences:
                    ote_bot, ote_top = self.engine.ote_zone(
                        swing_high, swing_low, "long")
                    if ote_bot < close:
                        ote_entry = (ote_bot + ote_top) / 2
                        if ote_entry < close and ote_entry > sl_price:
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
                zone = self.engine.premium_discount(close, swing_high, swing_low)
                if zone == "discount":
                    return None
                confluences.append("PREMIUM")

                if self.cfg.limit_entry_enabled and "LIQ_SWEEP" in confluences:
                    ote_bot, ote_top = self.engine.ote_zone(
                        swing_high, swing_low, "short")
                    if ote_top > close:
                        ote_entry = (ote_bot + ote_top) / 2
                        if ote_entry > close and ote_entry < sl_price:
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

        # ── R:R check ──
        rr = abs(tp_price - entry_price) / abs(entry_price - sl_price)
        if rr < self.cfg.min_rr:
            return None

        # ── Position sizing ──
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

    def _find_active_ob(self, obs, current_idx, price):
        for ob in reversed(obs):
            if ob["index"] >= current_idx:
                continue
            if current_idx - ob["index"] > self.cfg.ob_max_age_bars:
                continue
            if ob["bottom"] <= price <= ob["top"]:
                return ob
        return None

    def _find_active_fvg(self, fvgs, current_idx, price):
        for fvg in reversed(fvgs):
            if fvg["index"] >= current_idx:
                continue
            if current_idx - fvg["index"] > self.cfg.fvg_max_age_bars:
                continue
            if fvg["bottom"] <= price <= fvg["top"]:
                return fvg
        return None


# =============================================================================
# TRADE MANAGER — partial TP, trailing, BE
# =============================================================================
class TradeManager:
    def __init__(self, cfg: Config, mt5mgr: MT5Manager, tg: Telegram):
        self.cfg = cfg
        self.mt5 = mt5mgr
        self.tg = tg

    def manage(self, position, state: dict) -> dict:
        """
        Check and apply: partial TP → BE → trail.
        state tracks per-trade info (entry, original SL, partial status).
        Returns updated state.
        """
        entry_price = state["entry_price"]
        original_sl = state["original_sl"]
        risk = abs(entry_price - original_sl)
        if risk <= 0:
            return state

        is_long = position.type == mt5.ORDER_TYPE_BUY
        tick = self.mt5.get_tick()
        if tick is None:
            return state

        current_price = tick
        if is_long:
            current_r = (current_price - entry_price) / risk
        else:
            current_r = (entry_price - current_price) / risk

        # ── Partial TP ──
        if (self.cfg.partial_tp_enabled and
                not state.get("partial_closed", False) and
                current_r >= self.cfg.partial_tp_r):

            partial_lot = round(state["original_lot"] * self.cfg.partial_tp_pct, 2)
            partial_lot = max(0.01, partial_lot)

            if self.mt5.close_partial(position, partial_lot):
                # Move SL to BE
                if is_long:
                    new_sl = entry_price + risk * self.cfg.be_lock_r
                else:
                    new_sl = entry_price - risk * self.cfg.be_lock_r

                self.mt5.modify_sl(position, new_sl)
                state["partial_closed"] = True
                state["current_sl"] = new_sl

                self.tg.trade_update(
                    "Partial TP Taken (50%)",
                    f"R: {current_r:.1f}R | SL → BE ({new_sl:.2f})"
                )
                log.info(f"Partial TP: closed {partial_lot} lot @ {current_r:.1f}R, "
                         f"SL → {new_sl:.2f}")

        # ── Trailing (only with displacement) ──
        if (state.get("has_displacement", False) and
                current_r >= self.cfg.trail_trigger_r):

            if is_long:
                trail_sl = entry_price + risk * (current_r - self.cfg.trail_step_r)
                if trail_sl > state.get("current_sl", position.sl):
                    self.mt5.modify_sl(position, trail_sl)
                    state["current_sl"] = trail_sl
                    log.info(f"Trail SL → {trail_sl:.2f} (R={current_r:.1f})")
            else:
                trail_sl = entry_price - risk * (current_r - self.cfg.trail_step_r)
                if trail_sl < state.get("current_sl", position.sl):
                    self.mt5.modify_sl(position, trail_sl)
                    state["current_sl"] = trail_sl
                    log.info(f"Trail SL → {trail_sl:.2f} (R={current_r:.1f})")

        # ── BE (if partial not yet triggered) ──
        elif (not state.get("partial_closed", False) and
              current_r >= self.cfg.be_trigger_r):
            if is_long and position.sl < entry_price:
                new_sl = entry_price + risk * self.cfg.be_lock_r
                self.mt5.modify_sl(position, new_sl)
                state["current_sl"] = new_sl
                log.info(f"BE triggered → SL={new_sl:.2f}")
            elif not is_long and position.sl > entry_price:
                new_sl = entry_price - risk * self.cfg.be_lock_r
                self.mt5.modify_sl(position, new_sl)
                state["current_sl"] = new_sl
                log.info(f"BE triggered → SL={new_sl:.2f}")

        return state


# =============================================================================
# STATE PERSISTENCE
# =============================================================================
def load_state(filepath: str) -> dict:
    default = {
        "equity_peak": 0,
        "daily_trades": {},   # "YYYY-MM-DD": count
        "daily_pnl": {},      # "YYYY-MM-DD": float
        "last_trade_idx": 0,
        "open_trade": None,   # trade state dict
    }
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except:
            pass
    return default

def save_state(filepath: str, state: dict):
    with open(filepath, "w") as f:
        json.dump(state, f, indent=2, default=str)


# =============================================================================
# MAIN BOT
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

    def run(self):
        """Main entry — connect and start loop."""
        self._print_banner()

        if not self.mt5.connect():
            log.error("Cannot connect to MT5. Exiting.")
            return

        balance = self.mt5.get_balance()
        if self.state["equity_peak"] == 0:
            self.state["equity_peak"] = balance

        mode = "DRY RUN" if self.cfg.dry_run else "LIVE"
        log.info(f"Bot started — {mode} | Balance: ${balance:,.2f} | "
                 f"KZ: {self.cfg.kill_zone_mode}")
        self.tg.send(
            f"🤖 <b>ICT Bot Started</b>\n"
            f"Mode: {mode}\n"
            f"Balance: ${balance:,.2f}\n"
            f"Risk: {self.cfg.risk_per_trade_pct}% per trade\n"
            f"Kill Zones: {self.cfg.kill_zone_mode}"
        )

        # Main loop
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

    def _tick(self):
        """Called every scan_interval_sec. Checks for new bar close."""
        # Get latest M15 bar
        df = self.mt5.get_bars(self.cfg.tf_ltf, 2)
        if df is None or len(df) < 2:
            return

        # The COMPLETED bar is the second-to-last one
        latest_closed = df.index[-2]

        # Only process once per bar
        if self.last_bar_time == latest_closed:
            return
        self.last_bar_time = latest_closed

        log.info(f"── New M15 bar: {latest_closed} ──")
        self._process_bar()

    def _process_bar(self):
        """Full analysis + execution cycle on each M15 bar close."""
        # ── Pull data ──
        df_htf = self.mt5.get_bars(self.cfg.tf_htf, self.cfg.htf_bars)
        df_ltf = self.mt5.get_bars(self.cfg.tf_ltf, self.cfg.ltf_bars)

        if df_htf is None or df_ltf is None:
            log.warning("Data pull failed, skipping this bar.")
            return

        # ── HTF Analysis ──
        htf_swings = self.engine.find_swing_points(df_htf)
        htf_structure = self.engine.analyze_structure(htf_swings)
        htf_obs = self.engine.find_order_blocks(df_htf, htf_structure)

        # ── LTF Analysis ──
        ltf_swings = self.engine.find_swing_points(df_ltf)
        ltf_structure = self.engine.analyze_structure(ltf_swings)
        ltf_obs = self.engine.find_order_blocks(df_ltf, ltf_structure)
        ltf_fvgs = self.engine.find_fvg(df_ltf)

        # ── Manage existing position ──
        positions = self.mt5.get_my_positions()
        if positions:
            pos = positions[0]
            if self.state.get("open_trade"):
                self.state["open_trade"] = self.trade_mgr.manage(
                    pos, self.state["open_trade"]
                )
                save_state(self.cfg.state_file, self.state)

            # Also check if position was closed (SL/TP hit by MT5)
            # If we had a trade state but no position → it closed
        elif self.state.get("open_trade"):
            # Position gone → was closed by SL/TP
            log.info("Position closed (SL/TP hit by MT5)")
            self.tg.trade_update("Position Closed", "SL or TP hit.")
            self.state["open_trade"] = None
            save_state(self.cfg.state_file, self.state)

        # ── Check for pending limit orders that may have filled ──
        pending = self.mt5.get_my_pending()
        new_positions = self.mt5.get_my_positions()

        # If we had pending and now have position, limit was filled
        if not positions and new_positions and self.state.get("pending_limit"):
            self.state["open_trade"] = self.state.pop("pending_limit")
            self.tg.trade_update("Limit Order Filled!",
                                 f"Entry: {self.state['open_trade']['entry_price']:.2f}")
            log.info("Limit order filled, now managing position.")

        # ── Skip scan if already in trade or have pending ──
        if new_positions or pending:
            return

        # ── Daily state ──
        today = str(datetime.utcnow().date())
        daily_count = self.state["daily_trades"].get(today, 0)
        daily_pnl = self.state["daily_pnl"].get(today, 0.0)
        balance = self.mt5.get_balance()

        # Update equity peak
        if balance > self.state["equity_peak"]:
            self.state["equity_peak"] = balance

        # ── Scan for setup ──
        signal = self.scanner.scan(
            df_htf, df_ltf,
            htf_structure, htf_obs,
            ltf_swings, ltf_structure, ltf_obs, ltf_fvgs,
            daily_count, daily_pnl, balance,
            self.state["equity_peak"],
            self.state["last_trade_idx"]
        )

        if signal is None:
            return

        # ── SIGNAL FOUND — Execute ──
        log.info(f"🎯 SIGNAL: {signal['direction'].upper()} | "
                 f"Entry={signal['entry_price']} | SL={signal['sl_price']} | "
                 f"TP={signal['tp_price']} | R:R={signal['rr']} | "
                 f"Mode={signal['entry_mode']} | "
                 f"Score={signal['score']} | {signal['confluences']}")

        self.tg.signal(
            signal["direction"], signal["entry_price"],
            signal["sl_price"], signal["tp_price"],
            signal["confluences"], signal["kill_zone"],
            signal["score"], signal["entry_mode"], signal["rr"]
        )

        # Execute
        ticket = None
        if signal["entry_mode"] == "limit":
            expiry_min = self.cfg.limit_wait_bars * 15
            ticket = self.mt5.open_limit(
                signal["direction"], signal["lot_size"],
                signal["entry_price"], signal["sl_price"],
                signal["tp_price"], expiry_min
            )
            if ticket:
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
                signal["direction"], signal["lot_size"],
                signal["sl_price"], signal["tp_price"]
            )
            if ticket:
                self.state["open_trade"] = {
                    "entry_price": signal["entry_price"],
                    "original_sl": signal["sl_price"],
                    "original_lot": signal["lot_size"],
                    "has_displacement": signal["has_displacement"],
                    "partial_closed": False,
                    "current_sl": signal["sl_price"],
                }

        if ticket:
            self.state["daily_trades"][today] = daily_count + 1
            self.state["last_trade_idx"] = len(df_ltf) - 1
            save_state(self.cfg.state_file, self.state)
            self.tg.trade_update(
                f"{'Limit' if signal['entry_mode'] == 'limit' else 'Market'} "
                f"Order Placed",
                f"Ticket: #{ticket} | Lot: {signal['lot_size']}"
            )

    def _print_banner(self):
        print("╔══════════════════════════════════════════════════════════╗")
        print("║   ICT XAU/USD LIVE BOT v1.0                             ║")
        print("║   Full Auto · Telegram · Partial TP · Trail              ║")
        print("║   Based on Backtest v3.6 (PF 2.1 | WR 59%)              ║")
        print("╚══════════════════════════════════════════════════════════╝")


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    bot = ICTBot()
    bot.run()
