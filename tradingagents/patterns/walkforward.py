"""Rolling walk-forward evaluation of the whole discovery process.

A single train/valid/test split tells you how the patterns found *once* did
afterwards.  Walk-forward repeats the entire mine-and-select procedure at
several points in time and trades the selections only on the data that came
after each selection.  The concatenated out-of-sample trades estimate what a
trader who re-ran this engine every ``step`` would actually have experienced.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .miner import MinerConfig, MiningReport, PatternResult, mine_patterns, panel_arrays, rows_in, simulate_trades, window_stats


@dataclass
class FoldResult:
    train_window: Tuple[str, str]
    oos_window: Tuple[str, str]
    report: MiningReport
    oos_trades: pd.DataFrame
    oos_stats: Dict


@dataclass
class WalkForwardResult:
    folds: List[FoldResult]
    oos_trades: pd.DataFrame
    oos_stats: Dict
    stability: pd.DataFrame  # how often each pattern was selected across folds

    def to_dict(self) -> Dict:
        return {
            "n_folds": len(self.folds),
            "oos_stats": self.oos_stats,
            "folds": [
                {
                    "train": f.train_window,
                    "oos": f.oos_window,
                    "n_selected": len(f.report.selected),
                    "oos_stats": f.oos_stats,
                    "selected": [r.name for r in f.report.selected],
                }
                for f in self.folds
            ],
            "stability": self.stability.to_dict(orient="records"),
        }


def signature(r: PatternResult) -> str:
    """Pattern identity up to the exact threshold values, so that the same
    setup re-discovered with slightly different quantiles counts as stable."""
    parts = sorted(f"{c.feature}{c.op}" for c in r.pattern.conditions)
    return f"[{r.pattern.side}] " + " AND ".join(parts) + f" | hold {r.horizon}d"


def walk_forward(
    panel: pd.DataFrame,
    cfg: MinerConfig,
    n_folds: int = 4,
    min_train_years: float = 5.0,
    verbose: bool = True,
) -> WalkForwardResult:
    """Expanding-window walk-forward.

    The period after the first ``min_train_years`` is cut into ``n_folds``
    equal out-of-sample slices.  For each slice the miner sees only the data
    before it (split internally into train/valid with the configured
    fractions, test disabled), and its selections are traded on the slice.
    """
    dates = panel.index.get_level_values("Date").unique().sort_values()
    first_oos = dates[0] + pd.Timedelta(days=int(round(365.25 * min_train_years)))
    oos_dates = dates[dates >= first_oos]
    if len(oos_dates) < n_folds * 60:
        raise ValueError("not enough history for the requested number of folds")
    bounds = np.array_split(np.arange(len(oos_dates)), n_folds)
    arrs = panel_arrays(panel, cfg.horizons)
    e = cfg.embargo()

    inner_cfg = replace(cfg, train_frac=0.75, valid_frac=0.25)
    folds: List[FoldResult] = []
    all_trades = []
    counts: Dict[str, int] = {}
    for k, b in enumerate(bounds):
        oos_start, oos_end = oos_dates[b[0]], oos_dates[b[-1]]
        train_end_pos = np.searchsorted(dates.values, oos_start.to_datetime64()) - 1 - e
        train_dates = dates[: train_end_pos + 1]
        n = len(train_dates)
        i1 = int(n * inner_cfg.train_frac)
        windows = {
            "train": (train_dates[0], train_dates[i1 - 1]),
            "valid": (train_dates[min(i1 + e, n - 1)], train_dates[-1]),
        }
        if verbose:
            print(f"[walk-forward] fold {k + 1}/{n_folds}: train {windows['train'][0].date()}..{windows['valid'][1].date()} -> oos {oos_start.date()}..{oos_end.date()}")
        report = mine_patterns(panel, inner_cfg, windows=windows, verbose=verbose)
        oos_rows = rows_in(panel, (oos_start, oos_end))
        years = max((oos_end - oos_start).days / 365.25, 1 / 365.25)
        trades = []
        fold_sigs = set()
        for r in report.selected:
            t = simulate_trades(panel, arrs, r.pattern, r.horizon, oos_rows, cfg.cost_bps)
            t["pattern"] = r.name
            trades.append(t)
            fold_sigs.add(signature(r))
        for sig in fold_sigs:  # count each setup once per fold
            counts[sig] = counts.get(sig, 0) + 1
        tr = pd.concat(trades) if trades else pd.DataFrame(columns=["date", "symbol", "ret", "excess", "horizon", "pattern"])
        # if several patterns fire on the same symbol/day, count the trade once
        tr = tr.drop_duplicates(subset=["date", "symbol"]) if len(tr) else tr
        tr["fold"] = k
        st = window_stats(tr, years) if len(tr) else {"n": 0, "trades_per_year": 0.0, "mean": 0.0, "mean_excess": 0.0, "win_rate": 0.0, "t_stat": 0.0}
        folds.append(FoldResult((str(windows["train"][0].date()), str(windows["valid"][1].date())), (str(oos_start.date()), str(oos_end.date())), report, tr, st))
        all_trades.append(tr)

    oos = pd.concat(all_trades) if all_trades else pd.DataFrame()
    total_years = max((oos_dates[-1] - oos_dates[0]).days / 365.25, 1 / 365.25)
    oos_stats = window_stats(oos, total_years) if len(oos) else {"n": 0}
    stability = pd.DataFrame(
        sorted(({"pattern": k, "folds_selected": v} for k, v in counts.items()), key=lambda d: -d["folds_selected"])
    )
    return WalkForwardResult(folds, oos, oos_stats, stability)
