"""
S&P500 momentum-rotation engine with correlation-aware replacement.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import utils


@dataclass
class RotationConfig:
    years: int = 20
    holdings_count: int = 30
    momentum_lookback_days: int = 252
    momentum_skip_days: int = 21
    correlation_lookback_days: int = 126
    significant_momentum_drop: float = 0.35
    min_hold_days: int = 30
    refresh_interval_hours: int = 4


def build_sp500_data(
    output_csv: str,
    years: int = 20,
    min_columns: int = 300,
    force_refresh: bool = False,
) -> dict:
    """
    Build local S&P500-style price data for the most recent `years`.

    Falls back to deterministic synthetic data if network access is unavailable.
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    source = "utils.download_data"
    fallback_reason = None
    end_date = pd.Timestamp.utcnow().normalize()
    start_date = end_date - pd.DateOffset(years=years)
    prices = None

    if not force_refresh and os.path.exists(output_csv):
        prices = pd.read_csv(output_csv, index_col=0, parse_dates=True).sort_index()
    else:
        try:
            utils.download_data(output_csv)
            prices = pd.read_csv(output_csv, index_col=0, parse_dates=True).sort_index()
        except Exception as exc:
            fallback_reason = str(exc)

    if prices is None or prices.empty:
        source = "synthetic_factor_model"
        n_assets = max(min_columns, 500)
        business_days = pd.bdate_range(start=start_date, end=end_date)
        n_days = len(business_days)
        rng = np.random.default_rng(1234)

        market = rng.normal(0.0002, 0.01, size=n_days)
        sectors = rng.normal(0.0, 0.006, size=(n_days, 8))
        idio = rng.normal(0.0, 0.01, size=(n_days, n_assets))
        beta_m = rng.uniform(0.6, 1.4, size=n_assets)
        beta_s = rng.uniform(-0.3, 0.7, size=(8, n_assets))

        sector_component = sectors @ beta_s
        rets = market[:, None] * beta_m[None, :] + sector_component + idio
        prices_np = 100.0 * np.exp(np.cumsum(rets, axis=0))
        symbols = [f"SYN_{i:03d}" for i in range(n_assets)]
        prices = pd.DataFrame(prices_np, index=business_days, columns=symbols)

    prices = prices.loc[start_date:end_date].dropna(axis=1)
    if prices.shape[1] < min_columns:
        raise ValueError(
            f"Only {prices.shape[1]} symbols available, below minimum {min_columns}."
        )

    prices.to_csv(output_csv)
    return {
        "output_csv": output_csv,
        "source": source,
        "rows": int(prices.shape[0]),
        "columns": int(prices.shape[1]),
        "start": prices.index.min().strftime("%Y-%m-%d"),
        "end": prices.index.max().strftime("%Y-%m-%d"),
        "fallback_reason": fallback_reason,
    }


def _compute_momentum(
    prices: pd.DataFrame,
    as_of: pd.Timestamp,
    lookback_days: int,
    skip_days: int,
) -> pd.Series:
    log_returns = np.log(prices).diff().dropna(how="all")
    idx_tz = getattr(log_returns.index, "tz", None)
    if idx_tz is not None and getattr(as_of, "tzinfo", None) is None:
        as_of = as_of.tz_localize(idx_tz)
    elif idx_tz is None and getattr(as_of, "tzinfo", None) is not None:
        as_of = as_of.tz_localize(None)
    hist = log_returns.loc[:as_of]
    if len(hist) < lookback_days + skip_days + 5:
        return pd.Series(dtype=float)
    win = hist.iloc[-(lookback_days + skip_days) : -skip_days]
    return win.sum(axis=0).replace([np.inf, -np.inf], np.nan).dropna()


def _to_naive_timestamp(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t.tz_localize(None) if t.tzinfo is not None else t


def _normalize_timestamp_for_index(ts, index) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    idx_tz = getattr(index, "tz", None)
    if idx_tz is not None:
        if t.tzinfo is None:
            return t.tz_localize(idx_tz)
        return t.tz_convert(idx_tz)
    if t.tzinfo is not None:
        return t.tz_localize(None)
    return t


def _least_correlated_candidate(
    candidate_scores: pd.Series,
    survivors: list[str],
    corr_matrix: pd.DataFrame,
) -> str:
    best_symbol = None
    best_tuple = None
    for symbol, mom in candidate_scores.items():
        if symbol in survivors:
            continue
        if not survivors:
            avg_abs_corr = 0.0
        else:
            avg_abs_corr = float(corr_matrix.loc[symbol, survivors].abs().mean())
        key = (avg_abs_corr, -float(mom))
        if best_tuple is None or key < best_tuple:
            best_tuple = key
            best_symbol = symbol
    return best_symbol


def run_rotation_backtest(
    data_csv: str,
    report_json: str,
    config: RotationConfig,
) -> dict:
    """
    Backtest momentum-drop replacement logic over a full 20-year sample.
    """
    prices = pd.read_csv(data_csv, index_col=0, parse_dates=True).sort_index()
    prices = prices.dropna(axis=1)
    if prices.empty:
        raise ValueError("Input data has no usable symbols.")

    rebalance_dates = prices.index[::5]  # weekly-ish checks on daily bars
    if len(rebalance_dates) < 60:
        raise ValueError("Insufficient history for backtest.")

    first_date = rebalance_dates[0]
    first_mom = pd.Series(dtype=float)
    first_idx = 0
    for i, dt in enumerate(rebalance_dates):
        candidate = _compute_momentum(
            prices,
            dt,
            config.momentum_lookback_days,
            config.momentum_skip_days,
        )
        if not candidate.empty:
            first_mom = candidate.sort_values(ascending=False)
            first_date = dt
            first_idx = i
            break
    if first_mom.empty:
        raise ValueError("Not enough history to initialize holdings.")
    holdings = first_mom.head(config.holdings_count).index.tolist()

    entry_state = {
        sym: {
            "entry_date": first_date,
            "entry_momentum": float(first_mom[sym]),
            "peak_momentum": float(first_mom[sym]),
        }
        for sym in holdings
    }
    actions = []
    daily_portfolio_returns = []
    prev_date = first_date

    for as_of in rebalance_dates[first_idx + 1 :]:
        # Portfolio return since previous check (equal-weight over held names)
        ret_slice = prices.loc[prev_date:as_of, holdings].pct_change().dropna(how="all")
        if not ret_slice.empty:
            daily_portfolio_returns.extend(ret_slice.mean(axis=1).fillna(0.0).to_list())

        momentum = _compute_momentum(
            prices, as_of, config.momentum_lookback_days, config.momentum_skip_days
        )
        if momentum.empty:
            prev_date = as_of
            continue
        momentum = momentum.sort_values(ascending=False)
        corr_returns = np.log(
            prices[prices.columns.intersection(momentum.index)]
        ).diff()
        corr_win = corr_returns.loc[:as_of].iloc[-config.correlation_lookback_days :]
        corr_matrix = corr_win.corr().fillna(0.0)

        survivors = []
        dropped = []
        for sym in holdings:
            if sym not in momentum.index:
                dropped.append(sym)
                continue
            cur_mom = float(momentum[sym])
            state = entry_state[sym]
            state["peak_momentum"] = max(state["peak_momentum"], cur_mom)
            hold_days = int((as_of - state["entry_date"]).days)
            threshold = state["peak_momentum"] * (1 - config.significant_momentum_drop)
            if hold_days < config.min_hold_days or cur_mom >= threshold:
                survivors.append(sym)
            else:
                dropped.append(sym)

        candidates = momentum.drop(index=[s for s in survivors if s in momentum.index])
        additions = []
        while (
            len(survivors) + len(additions) < config.holdings_count
            and not candidates.empty
        ):
            pick = _least_correlated_candidate(
                candidates, survivors + additions, corr_matrix
            )
            if pick is None:
                break
            additions.append(pick)
            candidates = candidates.drop(index=[pick], errors="ignore")

        if dropped or additions:
            actions.append(
                {
                    "date": as_of.strftime("%Y-%m-%d"),
                    "drop": dropped,
                    "add": additions,
                }
            )

        holdings = (survivors + additions)[: config.holdings_count]
        new_state = {}
        for sym in holdings:
            if sym in entry_state and sym not in additions:
                new_state[sym] = entry_state[sym]
            else:
                cur = float(momentum[sym]) if sym in momentum.index else 0.0
                new_state[sym] = {
                    "entry_date": as_of,
                    "entry_momentum": cur,
                    "peak_momentum": cur,
                }
        entry_state = new_state
        prev_date = as_of

    returns = pd.Series(daily_portfolio_returns, dtype=float)
    if returns.empty:
        raise ValueError("Backtest generated no returns.")
    equity = (1 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1)
    annualized_return = float((equity.iloc[-1]) ** (252 / len(returns)) - 1)
    annualized_vol = float(returns.std(ddof=0) * np.sqrt(252))
    sharpe = float(annualized_return / annualized_vol) if annualized_vol > 0 else np.nan
    drawdown = equity / equity.cummax() - 1
    max_drawdown = float(drawdown.min())

    report = {
        "config": {
            "years": config.years,
            "holdings_count": config.holdings_count,
            "momentum_lookback_days": config.momentum_lookback_days,
            "momentum_skip_days": config.momentum_skip_days,
            "correlation_lookback_days": config.correlation_lookback_days,
            "significant_momentum_drop": config.significant_momentum_drop,
            "min_hold_days": config.min_hold_days,
            "refresh_interval_hours": config.refresh_interval_hours,
        },
        "data": {
            "csv": data_csv,
            "rows": int(prices.shape[0]),
            "columns": int(prices.shape[1]),
            "start": prices.index.min().strftime("%Y-%m-%d"),
            "end": prices.index.max().strftime("%Y-%m-%d"),
        },
        "performance": {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "rebalance_events": len(actions),
        },
        "final_holdings": holdings,
        "recent_actions": actions[-20:],
    }

    os.makedirs(os.path.dirname(report_json), exist_ok=True)
    with open(report_json, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    report["report_json"] = report_json
    return report


def generate_live_recommendation(
    data_csv: str,
    holdings_json: str,
    recommendation_json: str,
    config: RotationConfig,
) -> dict:
    """
    Generate one live recommendation snapshot.
    """
    prices = (
        pd.read_csv(data_csv, index_col=0, parse_dates=True).sort_index().dropna(axis=1)
    )
    as_of = prices.index.max()
    momentum = _compute_momentum(
        prices, as_of, config.momentum_lookback_days, config.momentum_skip_days
    ).sort_values(ascending=False)
    corr_returns = np.log(prices[prices.columns.intersection(momentum.index)]).diff()
    corr_win = corr_returns.iloc[-config.correlation_lookback_days :]
    corr_matrix = corr_win.corr().fillna(0.0)

    if os.path.exists(holdings_json):
        with open(holdings_json, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        holdings = [s for s in state.get("holdings", []) if s in momentum.index]
        entry_state = state.get("entry_state", {})
    else:
        holdings = momentum.head(config.holdings_count).index.tolist()
        entry_state = {}

    for sym in holdings:
        if sym not in entry_state:
            val = float(momentum[sym]) if sym in momentum.index else 0.0
            entry_state[sym] = {
                "entry_timestamp": as_of.strftime("%Y-%m-%d"),
                "entry_momentum": val,
                "peak_momentum": val,
            }

    survivors, dropped = [], []
    for sym in holdings:
        if sym not in momentum.index:
            dropped.append(sym)
            continue
        cur = float(momentum[sym])
        st = entry_state[sym]
        st["peak_momentum"] = max(float(st.get("peak_momentum", cur)), cur)
        entry_ts = _normalize_timestamp_for_index(st["entry_timestamp"], prices.index)
        hold_days = int((as_of - entry_ts).days)
        threshold = float(st["peak_momentum"]) * (1 - config.significant_momentum_drop)
        if hold_days < config.min_hold_days or cur >= threshold:
            survivors.append(sym)
        else:
            dropped.append(sym)

    candidates = momentum.drop(index=[s for s in survivors if s in momentum.index])
    additions = []
    while (
        len(survivors) + len(additions) < config.holdings_count and not candidates.empty
    ):
        pick = _least_correlated_candidate(
            candidates, survivors + additions, corr_matrix
        )
        if pick is None:
            break
        additions.append(pick)
        candidates = candidates.drop(index=[pick], errors="ignore")

    updated_holdings = (survivors + additions)[: config.holdings_count]
    new_entry_state = {}
    for sym in updated_holdings:
        if sym in entry_state and sym not in additions:
            new_entry_state[sym] = entry_state[sym]
        else:
            cur = float(momentum[sym]) if sym in momentum.index else 0.0
            new_entry_state[sym] = {
                "entry_timestamp": as_of.strftime("%Y-%m-%d"),
                "entry_momentum": cur,
                "peak_momentum": cur,
            }

    state_out = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "holdings": updated_holdings,
        "entry_state": new_entry_state,
    }
    os.makedirs(os.path.dirname(holdings_json), exist_ok=True)
    with open(holdings_json, "w", encoding="utf-8") as fh:
        json.dump(state_out, fh, indent=2)

    recommendation = {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "drop": dropped,
        "add": additions,
        "hold": survivors,
        "holdings_after_update": updated_holdings,
    }
    os.makedirs(os.path.dirname(recommendation_json), exist_ok=True)
    with open(recommendation_json, "w", encoding="utf-8") as fh:
        json.dump(recommendation, fh, indent=2)

    return recommendation


def run_live_loop(
    data_csv: str,
    holdings_json: str,
    recommendation_json: str,
    config: RotationConfig,
    once: bool = False,
):
    """
    Run recommendation generation every configured interval.
    """
    while True:
        generate_live_recommendation(
            data_csv=data_csv,
            holdings_json=holdings_json,
            recommendation_json=recommendation_json,
            config=config,
        )
        if once:
            break
        time.sleep(config.refresh_interval_hours * 3600)
