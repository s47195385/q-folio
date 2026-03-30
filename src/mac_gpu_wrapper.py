"""
Mac-compatible wrapper for portfolio optimization workflows.

This module provides:
1) Backend selection with graceful fallback when cuOpt/CUDA is unavailable.
2) A demo runner for a 4-year S&P 500-style momentum + low-correlation strategy.
"""

import time
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd
import os

from . import backtest, cvar_utils, utils
from .cvar_optimizer import CVaR
from .cvar_parameters import CvarParameters
from .portfolio import Portfolio


@dataclass
class ComputeBackend:
    name: str
    optimizer_api: str
    kde_device: str
    solver_settings: dict


def select_compute_backend(prefer_gpu: bool = True) -> ComputeBackend:
    """
    Select available compute backend and return a compatible configuration.

    On Mac M-series systems, this typically returns CPU/CVXPY while still allowing
    GPU acceleration for scenario generation if compatible libraries are present.
    """
    if prefer_gpu:
        try:
            __import__("cuopt.linear_programming.problem")
            return ComputeBackend(
                name="nvidia-gpu-cuopt",
                optimizer_api="cuopt_python",
                kde_device="GPU",
                solver_settings={"time_limit": 60},
            )
        except Exception:
            pass

    return ComputeBackend(
        name="cpu-cvxpy",
        optimizer_api="cvxpy",
        kde_device="CPU",
        solver_settings={"solver": cp.CLARABEL, "verbose": False},
    )


def _select_momentum_uncorrelated_universe(
    price_df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    momentum_lookback_days: int = 252,
    momentum_skip_days: int = 21,
    corr_lookback_days: int = 126,
    preselect_top_n: int = 100,
    target_n: int = 25,
) -> list[str]:
    """
    Build a momentum-ranked, low-correlation universe via greedy diversification.
    """
    full_returns = np.log(price_df).diff().dropna(how="all")
    cutoff = as_of_date
    hist = full_returns.loc[:cutoff]

    if len(hist) < momentum_lookback_days + momentum_skip_days + 5:
        raise ValueError("Not enough history for momentum selection.")

    momentum_window = hist.iloc[
        -(momentum_lookback_days + momentum_skip_days) : -momentum_skip_days
    ]
    momentum_scores = momentum_window.sum(axis=0).dropna()
    ranked = momentum_scores.sort_values(ascending=False).index.tolist()
    ranked = ranked[: min(preselect_top_n, len(ranked))]

    corr_window = hist.iloc[-corr_lookback_days:]
    corr_matrix = corr_window[ranked].corr().fillna(0.0)

    selected: list[str] = []
    for ticker in ranked:
        if len(selected) >= target_n:
            break
        if not selected:
            selected.append(ticker)
            continue
        avg_abs_corr = corr_matrix.loc[ticker, selected].abs().mean()
        if avg_abs_corr <= 0.35:
            selected.append(ticker)

    if len(selected) < target_n:
        for ticker in ranked:
            if ticker not in selected:
                selected.append(ticker)
            if len(selected) >= target_n:
                break

    return selected[:target_n]


def run_mac_compatible_demo(
    dataset_path: str,
    output_dir: str,
    years: int = 4,
    max_assets: int = 25,
    num_scen: int = 3000,
    prefer_gpu: bool = True,
) -> dict:
    """
    Run a 4-year compatible demo and return metrics, timing, and any warnings/bugs.
    """
    t0 = time.perf_counter()
    backend = select_compute_backend(prefer_gpu=prefer_gpu)
    warnings = []

    price_df = pd.read_csv(dataset_path, index_col=0, parse_dates=True).sort_index()
    price_df = price_df.dropna(axis=1)
    if price_df.empty:
        raise ValueError("Input dataset has no usable columns after dropping NaNs.")

    end_date = price_df.index.max()
    start_date = end_date - pd.DateOffset(years=years)
    sample_prices = price_df.loc[start_date:end_date].copy()
    sample_prices = sample_prices.dropna(axis=1)
    if sample_prices.shape[1] < max_assets:
        warnings.append(
            f"Available tickers in sample ({sample_prices.shape[1]}) < max_assets ({max_assets})."
        )

    selected = _select_momentum_uncorrelated_universe(
        sample_prices,
        as_of_date=sample_prices.index.max(),
        target_n=min(max_assets, sample_prices.shape[1]),
    )
    sample_prices = sample_prices[selected].dropna()

    regime = {
        "name": "mac-demo-4y",
        "range": (
            sample_prices.index.min().strftime("%Y-%m-%d"),
            sample_prices.index.max().strftime("%Y-%m-%d"),
        ),
    }
    returns_compute_settings = {"return_type": "LOG", "freq": 1}
    scenario_generation_settings = {
        "fit_type": "kde",
        "num_scen": num_scen,
        "verbose": False,
        "kde_settings": {
            "device": backend.kde_device,
            "bandwidth": 0.05,
            "kernel": "gaussian",
        },
    }

    returns_dict = utils.calculate_returns(sample_prices, regime, returns_compute_settings)
    try:
        returns_dict = cvar_utils.generate_cvar_data(returns_dict, scenario_generation_settings)
    except Exception as exc:
        warnings.append(
            f"KDE generation on {backend.kde_device} failed: {exc}. "
            "Falling back to gaussian."
        )
        scenario_generation_settings["fit_type"] = "gaussian"
        returns_dict = cvar_utils.generate_cvar_data(returns_dict, scenario_generation_settings)

    cvar_params = CvarParameters(
        w_min=0.0,
        w_max=0.15,
        c_min=0.0,
        c_max=0.05,
        L_tar=1.0,
        confidence=0.95,
        risk_aversion=1.0,
        cardinality=min(max_assets, len(selected)),
    )

    api_settings = {
        "api": backend.optimizer_api,
        "weight_constraints_type": "bounds",
        "cash_constraints_type": "bounds",
    }
    optimizer = CVaR(returns_dict=returns_dict, cvar_params=cvar_params, api_settings=api_settings)

    solve_start = time.perf_counter()
    result_row, optimal_portfolio = optimizer.solve_optimization_problem(
        backend.solver_settings, print_results=False
    )
    solve_seconds = time.perf_counter() - solve_start

    bt = backtest.portfolio_backtester(
        test_portfolio=optimal_portfolio,
        returns_dict=returns_dict,
        risk_free_rate=0.02,
        test_method="historical",
    )
    bt_result = bt.backtest_single_portfolio(optimal_portfolio).iloc[0]
    cumulative = bt_result["cumulative returns"]
    total_return = float(cumulative[-1] / cumulative[0] - 1)

    active_positions = int(np.sum(np.abs(optimal_portfolio.weights) > 1e-6))
    if active_positions > max_assets:
        warnings.append(
            f"Active positions ({active_positions}) exceeded requested max_assets ({max_assets})."
        )

    report = {
        "backend": backend.name,
        "optimizer_api": backend.optimizer_api,
        "kde_device": backend.kde_device,
        "period_start": regime["range"][0],
        "period_end": regime["range"][1],
        "tickers_selected": selected,
        "tickers_count": len(selected),
        "active_positions": active_positions,
        "requested_max_assets": max_assets,
        "metrics": {
            "objective": float(result_row["obj"]),
            "expected_return": float(result_row["return"]),
            "cvar": float(result_row["CVaR"]),
            "sharpe": float(bt_result["sharpe"]),
            "sortino": float(bt_result["sortino"]),
            "max_drawdown": float(bt_result["max drawdown"]),
            "total_return": total_return,
        },
        "timing_seconds": {
            "solve_seconds": solve_seconds,
            "total_demo_seconds": time.perf_counter() - t0,
        },
        "warnings_or_bugs": warnings,
    }

    output_path = os.path.join(output_dir, "mac_gpu_demo_report.json")
    os.makedirs(output_dir, exist_ok=True)
    pd.Series(report).to_json(output_path, indent=2)
    report["report_path"] = output_path
    return report
