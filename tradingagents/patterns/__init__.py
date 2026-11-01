"""Equity pattern recognition and backtesting engine.

The engine finds conjunctions of technical-indicator conditions ("patterns")
whose forward returns beat the unconditional base rate, and it validates them
with walk-forward testing, multiple-hypothesis correction and cost-aware
backtests.  See ``tradingagents/patterns/README.md`` for the methodology.
"""

from .data import load_prices
from .features import compute_features, forward_returns
from .rules import Condition, Pattern
from .miner import MinerConfig, mine_patterns
from .backtest import BacktestConfig, backtest_pattern

__all__ = [
    "load_prices",
    "compute_features",
    "forward_returns",
    "Condition",
    "Pattern",
    "MinerConfig",
    "mine_patterns",
    "BacktestConfig",
    "backtest_pattern",
]
