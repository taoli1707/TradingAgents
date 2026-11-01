"""Portfolio-level backtest of one or more validated patterns.

The simulator walks the calendar day by day.  A signal observed at the close
of day ``t`` opens a position at the open of day ``t+1`` if the symbol has no
open position and the book has capacity.  The position is closed at the
close ``horizon`` bars later.  Costs are charged per side.  Cash earns zero.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import compute_features
from .rules import Pattern


@dataclass
class BacktestConfig:
    max_positions: int = 10
    cost_bps: float = 5.0           # per side
    initial_capital: float = 100_000.0
    allow_short: bool = True
    position_frac: Optional[float] = None  # fraction of equity per trade; default 1/max_positions


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    metrics: Dict[str, float]
    exposure: pd.Series

    def to_dict(self) -> Dict:
        return {"metrics": self.metrics, "n_trades": int(len(self.trades))}


def _metrics(equity: pd.Series, trades: pd.DataFrame, exposure: pd.Series) -> Dict[str, float]:
    ret = equity.pct_change().dropna()
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1e-9)
    total = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if total > -1 else -1.0
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() > 0 else 0.0
    downside = ret[ret < 0].std()
    sortino = float(ret.mean() / downside * np.sqrt(252)) if downside and downside > 0 else 0.0
    dd = equity / equity.cummax() - 1.0
    wins = trades["ret"] > 0 if len(trades) else pd.Series(dtype=bool)
    gains = trades.loc[trades["ret"] > 0, "pnl"].sum() if len(trades) else 0.0
    losses = -trades.loc[trades["ret"] < 0, "pnl"].sum() if len(trades) else 0.0
    return {
        "total_return": float(total),
        "cagr": float(cagr),
        "annual_vol": float(ret.std() * np.sqrt(252)),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(dd.min()),
        "calmar": float(cagr / abs(dd.min())) if dd.min() < 0 else 0.0,
        "n_trades": int(len(trades)),
        "trades_per_year": float(len(trades) / years),
        "win_rate": float(wins.mean()) if len(trades) else 0.0,
        "avg_trade_ret": float(trades["ret"].mean()) if len(trades) else 0.0,
        "profit_factor": float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0,
        "avg_exposure": float(exposure.mean()),
        "years": float(years),
    }


def backtest_patterns(
    prices: Dict[str, pd.DataFrame],
    patterns: Sequence[Tuple[Pattern, int]],
    cfg: Optional[BacktestConfig] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    market_symbol: Optional[str] = None,
    features: Optional[Dict[str, pd.DataFrame]] = None,
) -> BacktestResult:
    """Trade a set of ``(pattern, horizon)`` pairs on a universe.

    When several patterns fire on the same symbol and day, the longest
    horizon wins (fewer round trips, lower cost).  Opposing sides on the same
    day cancel out and no trade is taken.
    """
    cfg = cfg or BacktestConfig()
    market = prices.get(market_symbol) if market_symbol else None
    feats = features or {s: compute_features(df, market=market) for s, df in prices.items()}

    # signal table: for each symbol/day -> (side, horizon)
    signals: Dict[str, pd.DataFrame] = {}
    for sym, f in feats.items():
        side = np.zeros(len(f))
        hor = np.zeros(len(f), dtype=int)
        for pat, h in patterns:
            if pat.side == "short" and not cfg.allow_short:
                continue
            m = pat.mask(f)
            s = 1.0 if pat.side == "long" else -1.0
            side[m] += s
            hor[m] = np.maximum(hor[m], h)
        signals[sym] = pd.DataFrame({"side": np.sign(side), "horizon": hor}, index=f.index)

    all_dates = sorted(set().union(*[set(df.index) for df in prices.values()]))
    all_dates = pd.DatetimeIndex(all_dates)
    if start:
        all_dates = all_dates[all_dates >= pd.Timestamp(start)]
    if end:
        all_dates = all_dates[all_dates <= pd.Timestamp(end)]

    # fast lookups
    loc = {s: {d: i for i, d in enumerate(df.index)} for s, df in prices.items()}
    opens = {s: df["Open"].to_numpy() for s, df in prices.items()}
    closes = {s: df["Close"].to_numpy() for s, df in prices.items()}
    sig_side = {s: signals[s]["side"].to_numpy() for s in prices}
    sig_hor = {s: signals[s]["horizon"].to_numpy() for s in prices}

    cost = cfg.cost_bps / 1e4
    frac = cfg.position_frac or 1.0 / cfg.max_positions
    cash = cfg.initial_capital
    open_pos: Dict[str, Dict] = {}
    pending: List[Tuple[str, float, int, pd.Timestamp]] = []  # orders to fill at next open
    equity_curve, exposure_curve, trades = [], [], []

    for d in all_dates:
        # 1. fill pending entries at today's open
        for sym, side, h, sig_date in pending:
            if sym in open_pos or len(open_pos) >= cfg.max_positions:
                continue
            i = loc[sym].get(d)
            if i is None:
                continue
            px = opens[sym][i]
            last_equity = equity_curve[-1] if equity_curve else cash
            notional = frac * last_equity
            shares = side * notional / px
            cash -= side * notional + abs(notional) * cost
            open_pos[sym] = {"shares": shares, "entry_px": px, "entry_date": d, "exit_idx": i + h, "notional": notional, "side": side, "signal_date": sig_date}
        pending = []

        # 2. mark to market / exit at today's close
        mtm = 0.0
        for sym in list(open_pos):
            p = open_pos[sym]
            i = loc[sym].get(d)
            if i is None:
                mtm += p["notional"]
                continue
            px = closes[sym][i]
            value = p["shares"] * px
            if i >= p["exit_idx"] or i == len(closes[sym]) - 1:
                cash += value - abs(value) * cost
                ret = p["side"] * (px / p["entry_px"] - 1.0) - 2 * cost
                trades.append({"symbol": sym, "entry_date": p["entry_date"], "exit_date": d, "side": p["side"], "ret": ret, "pnl": ret * p["notional"]})
                del open_pos[sym]
            else:
                mtm += value
        gross = sum(abs(p["shares"] * closes[s][loc[s][d]]) for s, p in open_pos.items() if d in loc[s])
        equity = cash + mtm
        equity_curve.append(equity)
        exposure_curve.append(gross / equity if equity > 0 else 0.0)

        # 3. read today's signals, queue entries for tomorrow's open
        for sym in prices:
            i = loc[sym].get(d)
            if i is None or i + 1 >= len(opens[sym]):
                continue
            s = sig_side[sym][i]
            if s != 0 and sym not in open_pos:
                pending.append((sym, float(s), int(sig_hor[sym][i]), d))

    eq = pd.Series(equity_curve, index=all_dates, name="equity")
    ex = pd.Series(exposure_curve, index=all_dates, name="exposure")
    tr = pd.DataFrame(trades)
    return BacktestResult(eq, tr, _metrics(eq, tr, ex), ex)


def backtest_pattern(prices, pattern: Pattern, horizon: int, cfg: Optional[BacktestConfig] = None, **kw) -> BacktestResult:
    return backtest_patterns(prices, [(pattern, horizon)], cfg, **kw)


def buy_and_hold(prices: Dict[str, pd.DataFrame], start=None, end=None, initial_capital: float = 100_000.0) -> pd.Series:
    """Equal-weight buy-and-hold benchmark on the same universe (daily rebalanced)."""
    closes = pd.concat({s: df["Close"] for s, df in prices.items()}, axis=1).sort_index()
    if start:
        closes = closes.loc[start:]
    if end:
        closes = closes.loc[:end]
    rets = closes.pct_change().mean(axis=1).fillna(0.0)
    return initial_capital * (1 + rets).cumprod()
