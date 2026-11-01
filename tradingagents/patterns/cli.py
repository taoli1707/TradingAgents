"""Command-line entry point.

Examples
--------
Mine, validate, walk-forward, fit the model and backtest on a universe::

    python -m tradingagents.patterns.cli mine \
        --symbols SPY QQQ AAPL MSFT NVDA ... --start 2005-01-01 \
        --walk-forward 4 --ml --backtest --out results/patterns

Check which validated patterns fire today::

    python -m tradingagents.patterns.cli scan --patterns results/patterns/patterns.json \
        --symbols SPY QQQ AAPL
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

import pandas as pd

from .backtest import BacktestConfig, backtest_patterns, buy_and_hold
from .data import LoadReport, load_prices
from .features import build_panel, compute_features
from .miner import MinerConfig, mine_patterns, panel_arrays, rows_in, simulate_trades, window_stats
from .report import render_markdown, save_outputs
from .rules import Pattern

DEFAULT_UNIVERSE = [
    "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "GLD", "TLT",
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "JPM", "BAC", "WFC", "GS", "XOM", "CVX", "JNJ",
    "PFE", "MRK", "UNH", "PG", "KO", "PEP", "WMT", "HD", "MCD", "NKE", "DIS", "INTC", "CSCO", "ORCL", "IBM",
    "CAT", "BA", "GE", "MMM", "HON", "UPS", "T", "VZ", "CMCSA", "AMD", "QCOM", "TXN", "COST", "LOW", "SBUX",
]


def _parse_int_list(s: str) -> tuple:
    return tuple(int(x) for x in s.split(",") if x.strip())


def cmd_mine(args: argparse.Namespace) -> int:
    symbols = args.symbols or DEFAULT_UNIVERSE
    rep = LoadReport([], {})
    prices = load_prices(symbols, start=args.start, end=args.end, source=args.source, csv_dir=args.csv_dir,
                         cache_dir=args.cache_dir, refresh=args.refresh, report=rep)
    if rep.failed:
        print(f"[data] skipped {len(rep.failed)} symbols: " + "; ".join(f"{k}: {v}" for k, v in list(rep.failed.items())[:10]), file=sys.stderr)
    if len(prices) < 2:
        print("need at least two symbols with data", file=sys.stderr)
        return 2
    market = args.market if args.market in prices else None
    horizons = _parse_int_list(args.horizons)
    panel = build_panel(prices, horizons=horizons, market_symbol=market)
    cfg = MinerConfig(
        horizons=horizons,
        max_conditions=args.max_conditions,
        beam_width=args.beam,
        min_trades_per_year=args.min_trades_per_year,
        min_signals=args.min_signals,
        cost_bps=args.cost_bps,
        fdr_q=args.fdr_q,
        min_t_screen=args.min_t_screen,
        train_frac=args.train_frac,
        valid_frac=args.valid_frac,
        allow_short=not args.long_only,
    )
    report = mine_patterns(panel, cfg)
    extras = {}
    wf_dict = ml_dict = bt_dict = bench_dict = None

    if args.walk_forward:
        from .walkforward import walk_forward

        wf = walk_forward(panel, cfg, n_folds=args.walk_forward, min_train_years=args.min_train_years)
        wf_dict = wf.to_dict()
        extras["walk_forward"] = wf_dict

    if args.ml:
        from .ml import MLConfig, train_walk_forward

        ml = train_walk_forward(panel, MLConfig(horizon=args.ml_horizon, n_folds=max(args.walk_forward, 3), min_train_years=args.min_train_years, cost_bps=args.cost_bps))
        ml_dict = ml.to_dict()
        # evaluate surrogate rules with the exact simulator on the test window
        if ml.surrogate_patterns and "test" in report.splits:
            arrs = panel_arrays(panel, horizons, base_rows=rows_in(panel, tuple(pd.Timestamp(x) for x in report.splits["train"])))
            test_rows = rows_in(panel, tuple(pd.Timestamp(x) for x in report.splits["test"]))
            years = (pd.Timestamp(report.splits["test"][1]) - pd.Timestamp(report.splits["test"][0])).days / 365.25
            rows = []
            for p in ml.surrogate_patterns:
                if args.ml_horizon in horizons:
                    tr = simulate_trades(panel, arrs, p, args.ml_horizon, test_rows, cfg.cost_bps)
                    s = window_stats(tr, years)
                    rows.append({"pattern": p.describe(), **{k: s[k] for k in ("n", "trades_per_year", "win_rate", "mean_excess", "t_stat")}})
            ml_dict["surrogate_test"] = rows
        extras["ml"] = ml_dict

    if args.backtest and report.selected and "test" in report.splits:
        t0, t1 = report.splits["test"]
        bt = backtest_patterns(prices, [(r.pattern, r.horizon) for r in report.selected],
                               BacktestConfig(max_positions=args.max_positions, cost_bps=args.cost_bps, allow_short=not args.long_only),
                               start=t0, end=t1, market_symbol=market)
        bt_dict = bt.to_dict()
        bh = buy_and_hold(prices, start=t0, end=t1)
        from .backtest import _metrics

        bench_dict = _metrics(bh, pd.DataFrame(columns=["ret", "pnl"]), pd.Series(1.0, index=bh.index))
        extras["backtest"] = bt_dict
        extras["benchmark"] = bench_dict
        os.makedirs(args.out, exist_ok=True)
        bt.equity.to_csv(os.path.join(args.out, "equity_test.csv"))
        bt.trades.to_csv(os.path.join(args.out, "trades_test.csv"), index=False)

    note = f"Data source: {args.source}; {args.start} to {args.end or 'today'}."
    md = render_markdown(report, wf_dict, ml_dict, bt_dict, bench_dict, data_note=note)
    paths = save_outputs(args.out, report, md, extras)
    print(md)
    print(f"[done] wrote {paths}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    with open(args.patterns) as fh:
        items = json.load(fh)
    if not items:
        print("no selected patterns in file")
        return 0
    patterns = [(Pattern.from_dict(d["pattern"]), d["horizon"], d.get("description")) for d in items]
    symbols = args.symbols or DEFAULT_UNIVERSE
    start = (pd.Timestamp(args.date or pd.Timestamp.today()) - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    prices = load_prices(symbols, start=start, end=args.date, source=args.source, csv_dir=args.csv_dir, cache_dir=args.cache_dir, refresh=True, min_rows=260)
    market = prices.get(args.market)
    hits = []
    for sym, df in prices.items():
        f = compute_features(df, market=market)
        last = f.iloc[[-1]]
        for pat, h, desc in patterns:
            if pat.mask(last)[0]:
                hits.append({"symbol": sym, "date": str(last.index[0].date()), "side": pat.side, "hold_days": h, "pattern": desc or pat.describe()})
    if not hits:
        print("no active signals")
    else:
        print(pd.DataFrame(hits).to_string(index=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tradingagents.patterns", description="Equity pattern mining and backtesting engine")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_data_args(sp):
        sp.add_argument("--symbols", nargs="*", help="tickers (default: a 60-name large-cap/ETF universe)")
        sp.add_argument("--source", default="yfinance", choices=["yfinance", "lean", "csv"])
        sp.add_argument("--csv-dir", default=None)
        sp.add_argument("--cache-dir", default=None)
        sp.add_argument("--market", default="SPY", help="symbol used for market-regime features")

    m = sub.add_parser("mine", help="discover, validate and backtest patterns")
    add_data_args(m)
    m.add_argument("--start", default="2005-01-01")
    m.add_argument("--end", default=None)
    m.add_argument("--refresh", action="store_true", help="ignore the price cache")
    m.add_argument("--horizons", default="1,3,5,10,20", help="holding periods in trading days")
    m.add_argument("--max-conditions", type=int, default=3)
    m.add_argument("--beam", type=int, default=40)
    m.add_argument("--min-trades-per-year", type=float, default=6.0)
    m.add_argument("--min-signals", type=int, default=60)
    m.add_argument("--cost-bps", type=float, default=5.0, help="per-side cost in basis points")
    m.add_argument("--fdr-q", type=float, default=0.10)
    m.add_argument("--min-t-screen", type=float, default=2.5)
    m.add_argument("--train-frac", type=float, default=0.6)
    m.add_argument("--valid-frac", type=float, default=0.2)
    m.add_argument("--long-only", action="store_true")
    m.add_argument("--walk-forward", type=int, default=0, help="number of walk-forward folds (0 = skip)")
    m.add_argument("--min-train-years", type=float, default=5.0)
    m.add_argument("--ml", action="store_true", help="also train the walk-forward gradient-boosting model")
    m.add_argument("--ml-horizon", type=int, default=5)
    m.add_argument("--backtest", action="store_true", help="portfolio backtest of selected patterns on the test window")
    m.add_argument("--max-positions", type=int, default=10)
    m.add_argument("--out", default="results/patterns")
    m.set_defaults(func=cmd_mine)

    s = sub.add_parser("scan", help="list validated patterns that fire on the latest bar")
    add_data_args(s)
    s.add_argument("--patterns", required=True, help="patterns.json written by `mine`")
    s.add_argument("--date", default=None)
    s.set_defaults(func=cmd_scan)
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
