#!/usr/bin/env python3
"""
Operational entrypoint for S&P500 20-year momentum/correlation rotation.
"""

from __future__ import annotations

import argparse
import json
import os

from src.snp500_rotation import (
    RotationConfig,
    build_sp500_data,
    generate_live_recommendation,
    run_live_loop,
    run_rotation_backtest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run S&P500 momentum-rotation backtest and live recommendations."
    )
    parser.add_argument(
        "--data-csv",
        default="/home/runner/work/q-folio/q-folio/data/stock_data/sp500_20y.csv",
        help="Absolute path to local S&P500 data CSV.",
    )
    parser.add_argument(
        "--report-json",
        default="/home/runner/work/q-folio/q-folio/data/stock_data/snp500_rotation_report.json",
        help="Absolute path to backtest report JSON.",
    )
    parser.add_argument(
        "--holdings-state-json",
        default="/home/runner/work/q-folio/q-folio/data/stock_data/snp500_live_holdings.json",
        help="Absolute path to rolling holdings state JSON.",
    )
    parser.add_argument(
        "--recommendation-json",
        default="/home/runner/work/q-folio/q-folio/data/stock_data/snp500_live_recommendation.json",
        help="Absolute path to latest recommendation JSON.",
    )
    parser.add_argument("--years", type=int, default=20)
    parser.add_argument("--holdings-count", type=int, default=30)
    parser.add_argument("--momentum-drop", type=float, default=0.35)
    parser.add_argument("--min-hold-days", type=int, default=30)
    parser.add_argument("--refresh-hours", type=int, default=4)
    parser.add_argument(
        "--mode",
        choices=["backtest", "recommend-once", "recommend-loop", "all"],
        default="all",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.data_csv), exist_ok=True)

    cfg = RotationConfig(
        years=args.years,
        holdings_count=args.holdings_count,
        significant_momentum_drop=args.momentum_drop,
        min_hold_days=args.min_hold_days,
        refresh_interval_hours=args.refresh_hours,
    )

    data_info = build_sp500_data(output_csv=args.data_csv, years=cfg.years)
    print(json.dumps({"data_info": data_info}, indent=2))

    if args.mode in {"backtest", "all"}:
        report = run_rotation_backtest(
            data_csv=args.data_csv, report_json=args.report_json, config=cfg
        )
        print(json.dumps({"backtest_report": report}, indent=2))

    if args.mode in {"recommend-once", "all"}:
        rec = generate_live_recommendation(
            data_csv=args.data_csv,
            holdings_json=args.holdings_state_json,
            recommendation_json=args.recommendation_json,
            config=cfg,
        )
        print(json.dumps({"recommendation": rec}, indent=2))

    if args.mode == "recommend-loop":
        run_live_loop(
            data_csv=args.data_csv,
            holdings_json=args.holdings_state_json,
            recommendation_json=args.recommendation_json,
            config=cfg,
            once=False,
        )


if __name__ == "__main__":
    main()
