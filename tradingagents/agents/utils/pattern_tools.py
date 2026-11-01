"""Agent tool: report which backtest-validated patterns fire for a symbol.

The tool reads ``patterns.json`` written by ``python -m tradingagents.patterns.cli mine``
and evaluates each validated pattern on the bar of ``curr_date``.  It gives the
market analyst a quantitative, out-of-sample-tested signal to weigh against
its qualitative reading of the indicators.
"""

from __future__ import annotations

import json
import os
from typing import Annotated, Optional

from langchain_core.tools import tool

from tradingagents.dataflows.config import get_config

_PATTERNS_FILE_ENV = "TRADINGAGENTS_PATTERNS_FILE"


def _patterns_path() -> Optional[str]:
    path = os.getenv(_PATTERNS_FILE_ENV)
    if path and os.path.exists(path):
        return path
    config = get_config()
    candidate = os.path.join(config.get("results_dir", "./results"), "patterns", "patterns.json")
    return candidate if os.path.exists(candidate) else None


def describe_active_patterns(symbol: str, curr_date: str, market_symbol: str = "SPY") -> str:
    import pandas as pd

    from tradingagents.patterns.data import load_prices
    from tradingagents.patterns.features import compute_features
    from tradingagents.patterns.rules import Pattern

    path = _patterns_path()
    if not path:
        return (
            "No validated pattern file found. Run `python -m tradingagents.patterns.cli mine` first, "
            f"or point {_PATTERNS_FILE_ENV} at a patterns.json."
        )
    with open(path) as fh:
        items = json.load(fh)
    if not items:
        return f"The pattern file {path} contains no validated patterns (none passed the statistical gates)."

    end = pd.Timestamp(curr_date)
    start = (end - pd.DateOffset(years=2)).strftime("%Y-%m-%d")
    prices = load_prices([symbol, market_symbol], start=start, end=end.strftime("%Y-%m-%d"), min_rows=260)
    if symbol.upper() not in prices:
        return f"Could not load price history for {symbol}."
    feats = compute_features(prices[symbol.upper()], market=prices.get(market_symbol.upper()))
    feats = feats.loc[:end]
    if feats.empty:
        return f"No bars for {symbol} on or before {curr_date}."
    last = feats.iloc[[-1]]
    bar_date = last.index[0].date()

    lines = [f"Validated patterns for {symbol.upper()} evaluated on the {bar_date} close ({len(items)} patterns checked):"]
    hits = 0
    for d in items:
        pat = Pattern.from_dict(d["pattern"])
        if pat.mask(last)[0]:
            hits += 1
            lines.append(f"- ACTIVE: {d.get('description', pat.describe())}; suggested hold {d['horizon']} trading days, entry at next open.")
    if hits == 0:
        lines.append("- No validated pattern is active on this bar.")
    lines.append(
        "Note: these patterns passed walk-forward validation on historical data; "
        "treat them as one quantitative input, not a guarantee."
    )
    return "\n".join(lines)


@tool
def get_validated_patterns(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
) -> str:
    """
    List the backtest-validated technical patterns (from the pattern mining engine)
    that are active for a ticker on the given date, with the suggested holding period.
    Returns a message if no pattern file has been generated yet.
    """
    return describe_active_patterns(symbol, curr_date)
