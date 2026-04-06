#!/usr/bin/env python3
"""
Analyze results for a stat-arbitrage research/live trading bot.

What it does:
- Scans a directory for CSV logs/research outputs
- Tries to identify signal logs, trade logs, equity logs, universe/research files
- Produces a plain-English report, summary JSON, CSV summaries, and charts
- Works even when your schema is slightly inconsistent

Typical usage on VPS:
    python3 analyze_stat_arb_results.py --logs-dir ~/apps/ravendell
    python3 analyze_stat_arb_results.py --logs-dir ~/apps/ravendell/logs --recursive
    python3 analyze_stat_arb_results.py --logs-dir . --out-dir analysis_output --starting-equity 10000

Recommended packages:
    pip install pandas numpy matplotlib
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


TIME_CANDIDATES = [
    "timestamp", "time", "datetime", "date", "ts", "bar_time", "scan_time",
    "created_at", "updated_at", "event_time", "signal_time", "open_time",
    "entry_time", "close_time", "exit_time", "closed_at", "opened_at",
]

PAIR_CANDIDATES = [
    "pair", "pair_name", "pair_symbol", "symbol_pair", "asset_pair",
    "spread_pair", "combo", "instrument_pair",
]

SIGNAL_CANDIDATES = ["signal", "side", "direction", "action", "bias"]
PNL_CANDIDATES = ["pnl", "realized_pnl", "net_pnl", "profit", "pnl_usd", "usd_pnl", "realized"]
EQUITY_CANDIDATES = ["equity", "balance", "account_equity", "demo_equity", "cum_equity"]
SCORE_CANDIDATES = ["live_score", "score", "composite_score", "quality_score"]
ZSCORE_CANDIDATES = ["z_score", "zscore", "z", "spread_z", "spread_zscore", "live_z"]
CORR_CANDIDATES = ["corr", "correlation", "pearson_corr"]
ROLLING_CORR_CANDIDATES = ["rolling_corr", "roll_corr", "rolling_correlation"]
ADF_CANDIDATES = ["adf_p", "adf_pvalue", "adf_p_value", "p_value", "cointegration_p"]
RECENT_ADF_CANDIDATES = ["recent_adf_p", "recent_adf_pvalue", "recent_cointegration_p"]
HALF_LIFE_CANDIDATES = ["half_life", "halflife", "hl"]
HEDGE_RATIO_CANDIDATES = ["hedge_ratio", "beta", "hr"]
SECTOR_CANDIDATES = ["sector", "cluster", "theme", "bucket"]
PAIR_TYPE_CANDIDATES = ["pair_type", "type", "relationship_type"]
EXIT_REASON_CANDIDATES = ["exit_reason", "reason", "close_reason"]
TRADE_ID_CANDIDATES = ["trade_id", "position_id", "id", "ticket"]
LEG_A_CANDIDATES = ["leg_a", "asset_a", "symbol_a", "base_symbol", "long_symbol", "left_symbol"]
LEG_B_CANDIDATES = ["leg_b", "asset_b", "symbol_b", "quote_symbol", "short_symbol", "right_symbol"]

ROLE_PATTERNS = {
    "signals": ["signal", "alert", "entry_candidates", "trade_signal"],
    "trades": ["trade", "fill", "execution", "position", "closed_trade", "orders"],
    "equity": ["equity", "balance", "curve", "pnl_curve"],
    "universe": ["universe", "research", "candidate", "screener", "rank", "pair_stats", "pair_metrics"],
}


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [slugify(str(c)) for c in df.columns]
    return df


def choose_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def parse_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if col in TIME_CANDIDATES or any(tok in col for tok in ["time", "date", "timestamp"]):
            try:
                parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
                if parsed.notna().sum() >= max(2, int(0.5 * len(df))):
                    df[col] = parsed
            except Exception:
                pass
    return df


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        if df[col].dtype == object:
            s = df[col].astype(str).str.replace(",", "", regex=False).str.replace("$", "", regex=False)
            num = pd.to_numeric(s, errors="coerce")
            if num.notna().sum() >= max(2, int(0.6 * len(df))):
                df[col] = num
    return df


def infer_role(path: Path, df: pd.DataFrame) -> str:
    name = path.name.lower()

    for role, patterns in ROLE_PATTERNS.items():
        if any(p in name for p in patterns):
            return role

    score = {"signals": 0, "trades": 0, "equity": 0, "universe": 0}
    cols = set(df.columns)

    if cols.intersection(SIGNAL_CANDIDATES):
        score["signals"] += 3
    if cols.intersection(ZSCORE_CANDIDATES):
        score["signals"] += 2
        score["universe"] += 2
    if cols.intersection(SCORE_CANDIDATES):
        score["signals"] += 2
        score["universe"] += 2
    if cols.intersection(PNL_CANDIDATES):
        score["trades"] += 4
    if cols.intersection(EQUITY_CANDIDATES):
        score["equity"] += 4
    if cols.intersection(ADF_CANDIDATES) or cols.intersection(CORR_CANDIDATES):
        score["universe"] += 2
        score["signals"] += 1
    if cols.intersection(EXIT_REASON_CANDIDATES):
        score["trades"] += 1
    if cols.intersection(TRADE_ID_CANDIDATES):
        score["trades"] += 1

    best_role = max(score, key=score.get)
    return best_role if score[best_role] > 0 else "other"


@dataclass
class LoadedFile:
    path: Path
    role: str
    rows: int
    cols: List[str]
    earliest_time: Optional[pd.Timestamp]
    latest_time: Optional[pd.Timestamp]
    df: pd.DataFrame


def find_csv_files(root: Path, recursive: bool = True) -> List[Path]:
    patterns = ["*.csv", "*.CSV"]
    files: List[Path] = []
    if recursive:
        for pattern in patterns:
            files.extend(root.rglob(pattern))
    else:
        for pattern in patterns:
            files.extend(root.glob(pattern))
    return sorted({p for p in files if p.is_file() and "analysis_output" not in str(p)})


def load_file(path: Path) -> Optional[LoadedFile]:
    for encoding in ["utf-8", "utf-8-sig", "latin1"]:
        try:
            df = pd.read_csv(path, low_memory=False, encoding=encoding)
            break
        except Exception:
            df = None
    if df is None:
        return None

    df = normalize_columns(df)
    df = parse_datetimes(df)
    df = coerce_numeric(df)

    ts_col = choose_timestamp_col(df)
    earliest = latest = None
    if ts_col is not None:
        series = df[ts_col].dropna()
        if not series.empty:
            earliest = series.min()
            latest = series.max()

    role = infer_role(path, df)
    return LoadedFile(
        path=path,
        role=role,
        rows=len(df),
        cols=list(df.columns),
        earliest_time=earliest,
        latest_time=latest,
        df=df,
    )


def choose_timestamp_col(df: pd.DataFrame) -> Optional[str]:
    candidates = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            candidates.append(col)
    if not candidates:
        return None
    preferred = [c for c in TIME_CANDIDATES if c in candidates]
    return preferred[0] if preferred else candidates[0]


def first_present(df: pd.DataFrame, candidate_groups: Sequence[Sequence[str]]) -> Optional[str]:
    for group in candidate_groups:
        col = choose_col(df, group)
        if col:
            return col
    return None


def ensure_pair_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[str]]:
    df = df.copy()
    pair_col = choose_col(df, PAIR_CANDIDATES)
    if pair_col:
        df[pair_col] = df[pair_col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        return df, pair_col

    leg_a = choose_col(df, LEG_A_CANDIDATES)
    leg_b = choose_col(df, LEG_B_CANDIDATES)
    if leg_a and leg_b:
        df["pair"] = df[leg_a].astype(str).str.strip() + " / " + df[leg_b].astype(str).str.strip()
        return df, "pair"

    return df, None


def half_life_quality(x: float) -> float:
    if pd.isna(x) or x <= 0:
        return 0.0
    if 2 <= x <= 24:
        return 1.0
    if x < 2:
        return max(0.0, x / 2)
    if x <= 72:
        return max(0.0, 1 - (x - 24) / 48)
    return 0.0


def quality_score_from_row(row: pd.Series) -> float:
    corr = float(row.get("corr", np.nan)) if not pd.isna(row.get("corr", np.nan)) else np.nan
    roll = float(row.get("rolling_corr", np.nan)) if not pd.isna(row.get("rolling_corr", np.nan)) else np.nan
    z = float(row.get("z_score", np.nan)) if not pd.isna(row.get("z_score", np.nan)) else np.nan
    adf = float(row.get("adf_p", np.nan)) if not pd.isna(row.get("adf_p", np.nan)) else np.nan
    hl = float(row.get("half_life", np.nan)) if not pd.isna(row.get("half_life", np.nan)) else np.nan

    score = 0.0
    if not np.isnan(corr):
        score += 30.0 * min(max(corr, 0.0) / 0.95, 1.0)
    if not np.isnan(roll):
        score += 20.0 * min(max(roll, 0.0) / 0.95, 1.0)
    if not np.isnan(z):
        score += 20.0 * min(max(abs(z) - 1.0, 0.0) / 2.0, 1.0)
    if not np.isnan(adf):
        score += 20.0 * (1.0 - min(max(adf, 0.0) / 0.10, 1.0))
    if not np.isnan(hl):
        score += 10.0 * half_life_quality(hl)
    return round(score, 2)


def classify_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "quality_score" not in df.columns:
        df["quality_score"] = df.apply(quality_score_from_row, axis=1)
    df["quality_bucket"] = pd.cut(
        df["quality_score"],
        bins=[-np.inf, 40, 60, 75, np.inf],
        labels=["weak", "borderline", "good", "elite"],
    )
    return df


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    return obj


def pct(x: float) -> float:
    return round(100.0 * x, 2)


def fmt_num(x: Any, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    if isinstance(x, (np.integer, int)):
        return f"{int(x):,}"
    try:
        return f"{float(x):,.{digits}f}"
    except Exception:
        return str(x)


def safe_value_counts(s: pd.Series, n: int = 10) -> pd.Series:
    return s.astype(str).replace("nan", np.nan).dropna().value_counts().head(n)


def analyze_signals(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    df, pair_col = ensure_pair_column(df)
    ts_col = choose_timestamp_col(df)
    sig_col = choose_col(df, SIGNAL_CANDIDATES)
    score_col = choose_col(df, SCORE_CANDIDATES)
    z_col = choose_col(df, ZSCORE_CANDIDATES)
    corr_col = choose_col(df, CORR_CANDIDATES)
    roll_col = choose_col(df, ROLLING_CORR_CANDIDATES)
    adf_col = choose_col(df, ADF_CANDIDATES)
    recent_adf_col = choose_col(df, RECENT_ADF_CANDIDATES)
    hl_col = choose_col(df, HALF_LIFE_CANDIDATES)
    hr_col = choose_col(df, HEDGE_RATIO_CANDIDATES)
    sector_col = choose_col(df, SECTOR_CANDIDATES)
    pair_type_col = choose_col(df, PAIR_TYPE_CANDIDATES)

    rename_map = {}
    for src, dst in [
        (score_col, "quality_score" if score_col else None),
        (z_col, "z_score" if z_col else None),
        (corr_col, "corr" if corr_col else None),
        (roll_col, "rolling_corr" if roll_col else None),
        (adf_col, "adf_p" if adf_col else None),
        (recent_adf_col, "recent_adf_p" if recent_adf_col else None),
        (hl_col, "half_life" if hl_col else None),
        (hr_col, "hedge_ratio" if hr_col else None),
    ]:
        if src and dst and src != dst:
            rename_map[src] = dst
    df = df.rename(columns=rename_map)
    df = classify_quality(df)

    if ts_col:
        df = df.sort_values(ts_col)

    out: Dict[str, Any] = {
        "row_count": len(df),
        "timestamp_col": ts_col,
        "pair_col": pair_col,
        "signal_col": sig_col,
        "earliest": str(df[ts_col].min()) if ts_col else None,
        "latest": str(df[ts_col].max()) if ts_col else None,
        "avg_quality_score": round(float(df["quality_score"].dropna().mean()), 2) if "quality_score" in df.columns else None,
        "quality_bucket_counts": df["quality_bucket"].astype(str).value_counts(dropna=False).to_dict() if "quality_bucket" in df.columns else {},
    }

    if pair_col:
        out["unique_pairs"] = int(df[pair_col].nunique(dropna=True))
        pair_counts = safe_value_counts(df[pair_col], 15)
        out["top_pairs_by_signal_count"] = pair_counts.to_dict()

    if sig_col:
        out["signal_distribution"] = safe_value_counts(df[sig_col], 20).to_dict()

    for col in ["quality_score", "z_score", "corr", "rolling_corr", "adf_p", "recent_adf_p", "half_life", "hedge_ratio"]:
        if col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").dropna()
            if not series.empty:
                out[f"{col}_mean"] = round(float(series.mean()), 4)
                out[f"{col}_median"] = round(float(series.median()), 4)
                out[f"{col}_p10"] = round(float(series.quantile(0.10)), 4)
                out[f"{col}_p90"] = round(float(series.quantile(0.90)), 4)

    if pair_col:
        pair_summary = df.groupby(pair_col).agg(
            signals=(pair_col, "size"),
            avg_quality_score=("quality_score", "mean") if "quality_score" in df.columns else (pair_col, "size"),
            avg_abs_z=("z_score", lambda s: np.nanmean(np.abs(pd.to_numeric(s, errors="coerce")))) if "z_score" in df.columns else (pair_col, "size"),
            avg_corr=("corr", "mean") if "corr" in df.columns else (pair_col, "size"),
            avg_adf_p=("adf_p", "mean") if "adf_p" in df.columns else (pair_col, "size"),
            median_half_life=("half_life", "median") if "half_life" in df.columns else (pair_col, "size"),
        ).reset_index()

        # Clean dummy columns if source columns were missing
        for col in ["avg_quality_score", "avg_abs_z", "avg_corr", "avg_adf_p", "median_half_life"]:
            if col in pair_summary.columns:
                if pd.api.types.is_numeric_dtype(pair_summary[col]):
                    continue
                pair_summary = pair_summary.drop(columns=[col])
        if "avg_quality_score" not in pair_summary.columns:
            pair_summary["avg_quality_score"] = np.nan
        pair_summary = pair_summary.sort_values(["avg_quality_score", "signals"], ascending=[False, False])
        out["pair_summary"] = pair_summary

    if sector_col:
        out["sector_distribution"] = safe_value_counts(df[sector_col], 20).to_dict()
    if pair_type_col:
        out["pair_type_distribution"] = safe_value_counts(df[pair_type_col], 20).to_dict()

    if ts_col:
        t = df[ts_col].dropna().sort_values()
        if len(t) >= 3:
            diffs = t.diff().dropna().dt.total_seconds() / 60.0
            out["median_scan_gap_minutes"] = round(float(diffs.median()), 2)
            out["p90_scan_gap_minutes"] = round(float(diffs.quantile(0.90)), 2)
            out["max_scan_gap_minutes"] = round(float(diffs.max()), 2)

    if ts_col and "quality_score" in df.columns:
        out["recent_signals"] = df.sort_values(ts_col, ascending=False).head(15).copy()
    else:
        out["recent_signals"] = df.head(15).copy()

    return out


def build_equity_curve_from_trades(trades: pd.DataFrame, pnl_col: str, exit_ts_col: Optional[str], starting_equity: float) -> pd.DataFrame:
    t = trades.copy()
    if exit_ts_col and exit_ts_col in t.columns:
        t = t.sort_values(exit_ts_col)
        x = exit_ts_col
    else:
        t = t.reset_index().rename(columns={"index": "trade_idx"})
        x = "trade_idx"
    t["cum_pnl"] = pd.to_numeric(t[pnl_col], errors="coerce").fillna(0.0).cumsum()
    t["equity_curve"] = starting_equity + t["cum_pnl"]
    return t[[x, pnl_col, "cum_pnl", "equity_curve"]]


def analyze_trades(df: pd.DataFrame, starting_equity: float = 10000.0) -> Dict[str, Any]:
    df = df.copy()
    df, pair_col = ensure_pair_column(df)

    pnl_col = choose_col(df, PNL_CANDIDATES)
    sig_col = choose_col(df, SIGNAL_CANDIDATES)
    exit_reason_col = choose_col(df, EXIT_REASON_CANDIDATES)
    trade_id_col = choose_col(df, TRADE_ID_CANDIDATES)

    entry_ts_col = first_present(df, [["entry_time"], ["open_time"], ["opened_at"], ["signal_time"], ["timestamp"], ["time"]])
    exit_ts_col = first_present(df, [["exit_time"], ["close_time"], ["closed_at"], ["timestamp"], ["time"]])

    if not pnl_col:
        # Try to infer from balance/equity deltas if no explicit pnl exists.
        equity_col = choose_col(df, EQUITY_CANDIDATES)
        if equity_col:
            tcol = choose_timestamp_col(df)
            temp = df.sort_values(tcol) if tcol else df.copy()
            temp["derived_pnl"] = pd.to_numeric(temp[equity_col], errors="coerce").diff()
            if temp["derived_pnl"].notna().sum() >= 2:
                df = temp
                pnl_col = "derived_pnl"

    if not pnl_col:
        return {
            "row_count": len(df),
            "error": "No PnL column found in trade-like file.",
            "trades_df": df,
            "pair_summary": pd.DataFrame(),
            "daily_summary": pd.DataFrame(),
            "equity_curve": pd.DataFrame(),
        }

    df[pnl_col] = pd.to_numeric(df[pnl_col], errors="coerce")
    trades = df[df[pnl_col].notna()].copy()
    if exit_ts_col and exit_ts_col in trades.columns:
        trades = trades.sort_values(exit_ts_col)

    if entry_ts_col and exit_ts_col and entry_ts_col in trades.columns and exit_ts_col in trades.columns:
        duration = (trades[exit_ts_col] - trades[entry_ts_col]).dt.total_seconds() / 3600.0
        trades["holding_hours"] = duration

    wins = trades[pnl_col] > 0
    losses = trades[pnl_col] < 0
    total_pnl = float(trades[pnl_col].sum()) if len(trades) else 0.0
    gross_profit = float(trades.loc[wins, pnl_col].sum()) if wins.any() else 0.0
    gross_loss = float(trades.loc[losses, pnl_col].sum()) if losses.any() else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else np.nan
    avg_pnl = float(trades[pnl_col].mean()) if len(trades) else np.nan
    median_pnl = float(trades[pnl_col].median()) if len(trades) else np.nan
    win_rate = float(wins.mean()) if len(trades) else np.nan

    equity_curve = build_equity_curve_from_trades(trades, pnl_col, exit_ts_col, starting_equity)
    if not equity_curve.empty:
        ec = equity_curve["equity_curve"]
        rolling_peak = ec.cummax()
        drawdown = (ec - rolling_peak) / rolling_peak.replace(0, np.nan)
        max_dd = float(drawdown.min()) if len(drawdown) else np.nan
    else:
        max_dd = np.nan

    pair_summary = pd.DataFrame()
    if pair_col and pair_col in trades.columns:
        pair_summary = trades.groupby(pair_col).agg(
            trades=(pair_col, "size"),
            total_pnl=(pnl_col, "sum"),
            avg_pnl=(pnl_col, "mean"),
            median_pnl=(pnl_col, "median"),
            win_rate=(pnl_col, lambda s: (s > 0).mean()),
            gross_profit=(pnl_col, lambda s: s[s > 0].sum()),
            gross_loss=(pnl_col, lambda s: s[s < 0].sum()),
        ).reset_index()
        pair_summary["profit_factor"] = pair_summary.apply(
            lambda r: (r["gross_profit"] / abs(r["gross_loss"])) if r["gross_loss"] < 0 else np.nan,
            axis=1,
        )
        pair_summary = pair_summary.sort_values(["total_pnl", "win_rate", "trades"], ascending=[False, False, False])

    daily_summary = pd.DataFrame()
    if exit_ts_col and exit_ts_col in trades.columns:
        trades["trade_date"] = trades[exit_ts_col].dt.date
        daily_summary = trades.groupby("trade_date").agg(
            trades=(pnl_col, "size"),
            total_pnl=(pnl_col, "sum"),
            win_rate=(pnl_col, lambda s: (s > 0).mean()),
            avg_pnl=(pnl_col, "mean"),
        ).reset_index().sort_values("trade_date")

    signal_summary = pd.DataFrame()
    if sig_col and sig_col in trades.columns:
        signal_summary = trades.groupby(sig_col).agg(
            trades=(pnl_col, "size"),
            total_pnl=(pnl_col, "sum"),
            win_rate=(pnl_col, lambda s: (s > 0).mean()),
            avg_pnl=(pnl_col, "mean"),
        ).reset_index().sort_values("total_pnl", ascending=False)

    out = {
        "row_count": len(df),
        "realized_trades": int(len(trades)),
        "pnl_col": pnl_col,
        "entry_timestamp_col": entry_ts_col,
        "exit_timestamp_col": exit_ts_col,
        "trade_id_col": trade_id_col,
        "total_pnl": round(total_pnl, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": round(float(profit_factor), 4) if not np.isnan(profit_factor) else None,
        "avg_pnl": round(avg_pnl, 4) if not np.isnan(avg_pnl) else None,
        "median_pnl": round(median_pnl, 4) if not np.isnan(median_pnl) else None,
        "win_rate": round(win_rate, 4) if not np.isnan(win_rate) else None,
        "max_drawdown": round(max_dd, 4) if not np.isnan(max_dd) else None,
        "best_trade": round(float(trades[pnl_col].max()), 2) if len(trades) else None,
        "worst_trade": round(float(trades[pnl_col].min()), 2) if len(trades) else None,
        "avg_holding_hours": round(float(trades["holding_hours"].mean()), 2) if "holding_hours" in trades.columns and trades["holding_hours"].notna().any() else None,
        "pair_summary": pair_summary,
        "daily_summary": daily_summary,
        "signal_summary": signal_summary,
        "equity_curve": equity_curve,
        "trades_df": trades,
    }

    if exit_reason_col and exit_reason_col in trades.columns:
        out["exit_reason_distribution"] = safe_value_counts(trades[exit_reason_col], 20).to_dict()

    return out


def analyze_universe(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    df, pair_col = ensure_pair_column(df)
    rename_map = {}
    for src, dst_candidates in [
        (choose_col(df, SCORE_CANDIDATES), "quality_score"),
        (choose_col(df, ZSCORE_CANDIDATES), "z_score"),
        (choose_col(df, CORR_CANDIDATES), "corr"),
        (choose_col(df, ROLLING_CORR_CANDIDATES), "rolling_corr"),
        (choose_col(df, ADF_CANDIDATES), "adf_p"),
        (choose_col(df, RECENT_ADF_CANDIDATES), "recent_adf_p"),
        (choose_col(df, HALF_LIFE_CANDIDATES), "half_life"),
        (choose_col(df, HEDGE_RATIO_CANDIDATES), "hedge_ratio"),
    ]:
        if src and src != dst_candidates:
            rename_map[src] = dst_candidates
    df = df.rename(columns=rename_map)
    df = classify_quality(df)

    out: Dict[str, Any] = {
        "row_count": len(df),
        "pair_col": pair_col,
        "avg_quality_score": round(float(df["quality_score"].mean()), 2) if "quality_score" in df.columns and df["quality_score"].notna().any() else None,
        "quality_bucket_counts": df["quality_bucket"].astype(str).value_counts(dropna=False).to_dict() if "quality_bucket" in df.columns else {},
    }

    for col in ["quality_score", "z_score", "corr", "rolling_corr", "adf_p", "recent_adf_p", "half_life", "hedge_ratio"]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                out[f"{col}_mean"] = round(float(s.mean()), 4)
                out[f"{col}_median"] = round(float(s.median()), 4)
                out[f"{col}_p10"] = round(float(s.quantile(0.10)), 4)
                out[f"{col}_p90"] = round(float(s.quantile(0.90)), 4)

    if pair_col:
        top_cols = [c for c in [pair_col, "quality_score", "z_score", "corr", "rolling_corr", "adf_p", "half_life", "hedge_ratio"] if c in df.columns]
        out["top_candidates"] = df.sort_values(["quality_score"], ascending=False)[top_cols].head(25).copy() if "quality_score" in df.columns else df[top_cols].head(25).copy()

    elite_mask = pd.Series([True] * len(df), index=df.index)
    if "corr" in df.columns:
        elite_mask &= pd.to_numeric(df["corr"], errors="coerce") >= 0.80
    if "rolling_corr" in df.columns:
        elite_mask &= pd.to_numeric(df["rolling_corr"], errors="coerce") >= 0.75
    if "adf_p" in df.columns:
        elite_mask &= pd.to_numeric(df["adf_p"], errors="coerce") <= 0.05
    if "z_score" in df.columns:
        elite_mask &= pd.to_numeric(df["z_score"], errors="coerce").abs() >= 1.8
    if "half_life" in df.columns:
        hl = pd.to_numeric(df["half_life"], errors="coerce")
        elite_mask &= hl.between(2, 48, inclusive="both")
    out["elite_candidates_count"] = int(elite_mask.sum())
    out["elite_ratio"] = round(float(elite_mask.mean()), 4) if len(df) else None

    return out


def analyze_equity(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.copy()
    tcol = choose_timestamp_col(df)
    ecol = choose_col(df, EQUITY_CANDIDATES)
    if not ecol:
        return {"row_count": len(df), "error": "No equity column found.", "equity_curve": pd.DataFrame()}
    df[ecol] = pd.to_numeric(df[ecol], errors="coerce")
    curve = df[[c for c in [tcol, ecol] if c]].dropna().copy()
    if tcol:
        curve = curve.sort_values(tcol)
    curve = curve.rename(columns={ecol: "equity_curve"})
    if not curve.empty:
        ec = curve["equity_curve"]
        rolling_peak = ec.cummax()
        drawdown = (ec - rolling_peak) / rolling_peak.replace(0, np.nan)
        out = {
            "row_count": len(df),
            "start_equity": round(float(ec.iloc[0]), 2),
            "end_equity": round(float(ec.iloc[-1]), 2),
            "net_change": round(float(ec.iloc[-1] - ec.iloc[0]), 2),
            "return_pct": round(float((ec.iloc[-1] / ec.iloc[0] - 1) * 100), 2) if ec.iloc[0] != 0 else None,
            "max_drawdown": round(float(drawdown.min()), 4),
            "equity_curve": curve,
        }
    else:
        out = {"row_count": len(df), "equity_curve": curve}
    return out


def combine_by_role(loaded_files: List[LoadedFile], role: str) -> Optional[pd.DataFrame]:
    dfs = [lf.df.assign(__source_file=str(lf.path)) for lf in loaded_files if lf.role == role]
    if not dfs:
        return None
    try:
        return pd.concat(dfs, ignore_index=True, sort=False)
    except Exception:
        return dfs[0]


def save_df(df: pd.DataFrame, path: Path) -> None:
    if df is not None and not df.empty:
        df.to_csv(path, index=False)


def make_plots(out_dir: Path, signals: Optional[Dict[str, Any]], trades: Optional[Dict[str, Any]], universe: Optional[Dict[str, Any]], equity: Optional[Dict[str, Any]]) -> List[str]:
    if not HAS_MPL:
        return []

    created: List[str] = []

    # Pair signal counts
    if signals and isinstance(signals.get("pair_summary"), pd.DataFrame) and not signals["pair_summary"].empty:
        df = signals["pair_summary"].head(15).copy()
        if "signals" in df.columns:
            fig, ax = plt.subplots(figsize=(11, 6))
            ax.barh(df.iloc[::-1, 0].astype(str), df.iloc[::-1]["signals"])
            ax.set_title("Top Pairs by Signal Count")
            ax.set_xlabel("Signals")
            fig.tight_layout()
            p = out_dir / "signals_top_pairs.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            created.append(p.name)

    # Universe scatter
    if universe and isinstance(universe.get("top_candidates"), pd.DataFrame):
        source = universe["top_candidates"]
        if not source.empty and {"corr", "adf_p"}.issubset(source.columns):
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.scatter(source["corr"], source["adf_p"])
            ax.set_title("Universe Quality Scatter")
            ax.set_xlabel("Correlation")
            ax.set_ylabel("ADF p-value")
            fig.tight_layout()
            p = out_dir / "universe_quality_scatter.png"
            fig.savefig(p, dpi=160)
            plt.close(fig)
            created.append(p.name)

    # Equity curve
    eq_curve = None
    if trades and isinstance(trades.get("equity_curve"), pd.DataFrame) and not trades["equity_curve"].empty:
        eq_curve = trades["equity_curve"]
    elif equity and isinstance(equity.get("equity_curve"), pd.DataFrame) and not equity["equity_curve"].empty:
        eq_curve = equity["equity_curve"]
    if eq_curve is not None:
        xcol = eq_curve.columns[0]
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.plot(eq_curve[xcol], eq_curve["equity_curve"])
        ax.set_title("Equity Curve")
        ax.set_xlabel(xcol)
        ax.set_ylabel("Equity")
        fig.tight_layout()
        p = out_dir / "equity_curve.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        created.append(p.name)

    # Daily pnl
    if trades and isinstance(trades.get("daily_summary"), pd.DataFrame) and not trades["daily_summary"].empty:
        df = trades["daily_summary"]
        fig, ax = plt.subplots(figsize=(11, 6))
        ax.bar(df["trade_date"].astype(str), df["total_pnl"])
        ax.set_title("Daily PnL")
        ax.set_xlabel("Date")
        ax.set_ylabel("PnL")
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()
        p = out_dir / "daily_pnl.png"
        fig.savefig(p, dpi=160)
        plt.close(fig)
        created.append(p.name)

    return created


def build_file_inventory(loaded_files: List[LoadedFile]) -> pd.DataFrame:
    rows = []
    for lf in loaded_files:
        rows.append({
            "path": str(lf.path),
            "role": lf.role,
            "rows": lf.rows,
            "earliest_time": lf.earliest_time,
            "latest_time": lf.latest_time,
            "columns": ", ".join(lf.cols[:25]),
        })
    return pd.DataFrame(rows)


def build_report(
    file_inventory: pd.DataFrame,
    signals: Optional[Dict[str, Any]],
    trades: Optional[Dict[str, Any]],
    universe: Optional[Dict[str, Any]],
    equity: Optional[Dict[str, Any]],
    charts: List[str],
) -> str:
    lines: List[str] = []
    lines.append("STAT-ARB BOT ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")

    lines.append("1) FILE INVENTORY")
    lines.append("-" * 80)
    if file_inventory.empty:
        lines.append("No CSV files found.")
    else:
        role_counts = file_inventory["role"].value_counts().to_dict()
        lines.append(f"Files found: {len(file_inventory)}")
        lines.append("Role breakdown: " + ", ".join(f"{k}={v}" for k, v in role_counts.items()))
        latest_any = pd.to_datetime(file_inventory["latest_time"], errors="coerce").dropna()
        if not latest_any.empty:
            lines.append(f"Latest timestamp across all files: {latest_any.max()}")
    lines.append("")

    if signals:
        lines.append("2) SIGNAL LOG ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Signal rows: {signals.get('row_count', 0)}")
        if signals.get("unique_pairs") is not None:
            lines.append(f"Unique pairs: {signals['unique_pairs']}")
        if signals.get("avg_quality_score") is not None:
            lines.append(f"Average quality score: {fmt_num(signals['avg_quality_score'])}")
        if signals.get("latest"):
            lines.append(f"Latest signal timestamp: {signals['latest']}")
        if signals.get("median_scan_gap_minutes") is not None:
            lines.append(f"Median scan gap: {fmt_num(signals['median_scan_gap_minutes'])} minutes")
        if signals.get("signal_distribution"):
            lines.append("Signal distribution: " + ", ".join(f"{k}={v}" for k, v in signals["signal_distribution"].items()))
        if signals.get("top_pairs_by_signal_count"):
            preview = list(signals["top_pairs_by_signal_count"].items())[:8]
            lines.append("Top pairs by frequency: " + ", ".join(f"{k}={v}" for k, v in preview))
        lines.append("")

    if universe:
        lines.append("3) RESEARCH / UNIVERSE ANALYSIS")
        lines.append("-" * 80)
        lines.append(f"Universe rows: {universe.get('row_count', 0)}")
        if universe.get("avg_quality_score") is not None:
            lines.append(f"Average candidate quality score: {fmt_num(universe['avg_quality_score'])}")
        if universe.get("elite_candidates_count") is not None:
            elite_ratio = universe.get("elite_ratio")
            elite_text = f" ({pct(elite_ratio)}%)" if elite_ratio is not None else ""
            lines.append(f"Elite candidates: {universe['elite_candidates_count']}{elite_text}")
        qb = universe.get("quality_bucket_counts") or {}
        if qb:
            lines.append("Quality buckets: " + ", ".join(f"{k}={v}" for k, v in qb.items()))
        lines.append("")

    if trades:
        lines.append("4) TRADE PERFORMANCE")
        lines.append("-" * 80)
        if trades.get("error"):
            lines.append(trades["error"])
        else:
            lines.append(f"Realized trades: {trades.get('realized_trades', 0)}")
            lines.append(f"Total PnL: {fmt_num(trades.get('total_pnl'))}")
            if trades.get("profit_factor") is not None:
                lines.append(f"Profit factor: {fmt_num(trades.get('profit_factor'), 3)}")
            if trades.get("win_rate") is not None:
                lines.append(f"Win rate: {pct(trades.get('win_rate'))}%")
            if trades.get("avg_pnl") is not None:
                lines.append(f"Average trade PnL: {fmt_num(trades.get('avg_pnl'))}")
            if trades.get("max_drawdown") is not None:
                lines.append(f"Max drawdown: {pct(abs(trades.get('max_drawdown')))}%")
            if trades.get("avg_holding_hours") is not None:
                lines.append(f"Average holding time: {fmt_num(trades.get('avg_holding_hours'))} hours")
            if trades.get("exit_reason_distribution"):
                lines.append("Exit reasons: " + ", ".join(f"{k}={v}" for k, v in trades["exit_reason_distribution"].items()))
        lines.append("")

    if equity:
        lines.append("5) EQUITY CURVE")
        lines.append("-" * 80)
        if equity.get("error"):
            lines.append(equity["error"])
        else:
            if equity.get("start_equity") is not None:
                lines.append(f"Start equity: {fmt_num(equity.get('start_equity'))}")
            if equity.get("end_equity") is not None:
                lines.append(f"End equity: {fmt_num(equity.get('end_equity'))}")
            if equity.get("net_change") is not None:
                lines.append(f"Net change: {fmt_num(equity.get('net_change'))}")
            if equity.get("return_pct") is not None:
                lines.append(f"Return: {fmt_num(equity.get('return_pct'))}%")
            if equity.get("max_drawdown") is not None:
                lines.append(f"Max drawdown: {pct(abs(equity.get('max_drawdown')))}%")
        lines.append("")

    lines.append("6) BRUTAL TAKE")
    lines.append("-" * 80)
    brutal_points: List[str] = []
    if signals:
        if signals.get("avg_quality_score") is not None and signals["avg_quality_score"] < 60:
            brutal_points.append("Your filter stack is still too permissive. Average candidate quality is mediocre, so the bot is spending time on junk.")
        if signals.get("median_scan_gap_minutes") is not None and signals["median_scan_gap_minutes"] > 75:
            brutal_points.append("Your operational cadence is weak. Scan gaps are too wide for a 1h evaluation engine unless this is deliberate and tested.")
    if universe:
        if universe.get("elite_ratio") is not None and universe["elite_ratio"] < 0.10:
            brutal_points.append("Your universe is bloated. Less than 10% of candidates qualify as elite, which means your screener still needs harder selection pressure.")
    if trades and not trades.get("error"):
        wr = trades.get("win_rate")
        pf = trades.get("profit_factor")
        if wr is not None and wr < 0.50:
            brutal_points.append("Your edge is not proven. A sub-50% win rate is survivable only with strong payoff asymmetry, and you need data proving that.")
        if pf is not None and pf < 1.10:
            brutal_points.append("You do not have a deployable system yet. Profit factor under 1.1 is noise after fees, slippage, and bad execution days.")
        if trades.get("realized_trades", 0) < 30:
            brutal_points.append("Your sample size is too small. Do not fool yourself with a handful of trades.")
    if not brutal_points:
        brutal_points.append("Nothing catastrophic jumped out, but that does not mean you are done. It means your next bottleneck is discipline, data quality, or execution leakage.")
    lines.extend(f"- {p}" for p in brutal_points)
    lines.append("")

    lines.append("7) OUTPUT FILES")
    lines.append("-" * 80)
    lines.append("Generated charts: " + (", ".join(charts) if charts else "none"))
    lines.append("Generated tables: file_inventory.csv, plus any *_summary.csv files that had enough data")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze stat-arb research/live bot CSV outputs.")
    parser.add_argument("--logs-dir", default=".", help="Directory containing CSV logs/research outputs")
    parser.add_argument("--out-dir", default="analysis_output", help="Directory to write results")
    parser.add_argument("--starting-equity", type=float, default=10000.0, help="Starting equity for synthetic equity curve if only trade PnL exists")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan subfolders")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation")
    args = parser.parse_args()

    root = Path(args.logs_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_csv_files(root, recursive=args.recursive)
    loaded: List[LoadedFile] = []
    failed: List[str] = []
    for path in files:
        lf = load_file(path)
        if lf is None:
            failed.append(str(path))
        else:
            loaded.append(lf)

    file_inventory = build_file_inventory(loaded)
    save_df(file_inventory, out_dir / "file_inventory.csv")

    signals_df = combine_by_role(loaded, "signals")
    trades_df = combine_by_role(loaded, "trades")
    universe_df = combine_by_role(loaded, "universe")
    equity_df = combine_by_role(loaded, "equity")

    signals = analyze_signals(signals_df) if signals_df is not None and not signals_df.empty else None
    trades = analyze_trades(trades_df, starting_equity=args.starting_equity) if trades_df is not None and not trades_df.empty else None
    universe = analyze_universe(universe_df) if universe_df is not None and not universe_df.empty else None
    equity = analyze_equity(equity_df) if equity_df is not None and not equity_df.empty else None

    if signals and isinstance(signals.get("pair_summary"), pd.DataFrame):
        save_df(signals["pair_summary"], out_dir / "signals_pair_summary.csv")
    if signals and isinstance(signals.get("recent_signals"), pd.DataFrame):
        save_df(signals["recent_signals"], out_dir / "recent_signals.csv")
    if trades and isinstance(trades.get("pair_summary"), pd.DataFrame):
        save_df(trades["pair_summary"], out_dir / "trade_pair_summary.csv")
    if trades and isinstance(trades.get("daily_summary"), pd.DataFrame):
        save_df(trades["daily_summary"], out_dir / "trade_daily_summary.csv")
    if trades and isinstance(trades.get("signal_summary"), pd.DataFrame):
        save_df(trades["signal_summary"], out_dir / "trade_signal_summary.csv")
    if trades and isinstance(trades.get("equity_curve"), pd.DataFrame):
        save_df(trades["equity_curve"], out_dir / "trade_equity_curve.csv")
    if universe and isinstance(universe.get("top_candidates"), pd.DataFrame):
        save_df(universe["top_candidates"], out_dir / "universe_top_candidates.csv")
    if equity and isinstance(equity.get("equity_curve"), pd.DataFrame):
        save_df(equity["equity_curve"], out_dir / "equity_curve.csv")

    charts = []
    if not args.no_charts:
        charts = make_plots(out_dir, signals, trades, universe, equity)

    report = build_report(file_inventory, signals, trades, universe, equity, charts)
    (out_dir / "analysis_report.txt").write_text(report, encoding="utf-8")

    summary = {
        "root": root,
        "out_dir": out_dir,
        "files_scanned": len(files),
        "files_loaded": len(loaded),
        "files_failed": failed,
        "signals": {k: v for k, v in (signals or {}).items() if not isinstance(v, pd.DataFrame)},
        "trades": {k: v for k, v in (trades or {}).items() if not isinstance(v, pd.DataFrame)},
        "universe": {k: v for k, v in (universe or {}).items() if not isinstance(v, pd.DataFrame)},
        "equity": {k: v for k, v in (equity or {}).items() if not isinstance(v, pd.DataFrame)},
        "charts": charts,
    }
    (out_dir / "summary.json").write_text(json.dumps(to_serializable(summary), indent=2), encoding="utf-8")

    print(report)
    print("\nSaved outputs to:", out_dir)


if __name__ == "__main__":
    main()
