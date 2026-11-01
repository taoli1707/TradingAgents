"""Beam-search pattern miner with walk-forward validation.

Pipeline
--------
1. **Screen** (train window): enumerate single conditions, then pairs and
   triples of the strongest ones (beam search).  Each (pattern, horizon) is
   scored by the mean *excess* forward return - the pattern's return minus the
   symbol's unconditional return for the same horizon - with a t-statistic
   whose standard error is clustered by holding-window block, so that
   overlapping signals do not inflate the evidence.
2. **Correct** for the number of trials with Benjamini-Hochberg FDR control.
3. **Confirm** survivors by simulating realistic, non-overlapping trades
   (one position per symbol, next-open entry, cost per side) on the train
   window and on an embargoed validation window the miner never saw.
4. **Report** the test window once, for the final selection only.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import BOOLEAN_FEATURES, feature_columns
from .rules import Condition, Pattern, compatible, generate_conditions, DEFAULT_QUANTILES
from .stats import TradeStats, benjamini_hochberg, deflated_sharpe, t_to_p, trade_stats


@dataclass
class MinerConfig:
    horizons: Tuple[int, ...] = (1, 3, 5, 10, 20)
    max_conditions: int = 3
    beam_width: int = 40
    quantiles: Tuple[float, ...] = DEFAULT_QUANTILES
    features: Optional[List[str]] = None  # None = every feature column
    exclude_features: Tuple[str, ...] = ("dollar_vol_20_log",)
    # support / frequency
    min_signals: int = 60              # overlapping signal rows in train
    max_signal_fraction: float = 0.35  # a "pattern" should be selective
    min_trades_per_year: float = 6.0   # non-overlapping trades, whole universe
    # significance
    min_t_screen: float = 2.5
    fdr_q: float = 0.10
    top_k_confirm: int = 400
    # validation gates (applied on the validation window)
    min_valid_t: float = 1.0
    min_valid_trades: int = 15
    min_valid_win_rate: float = 0.5
    max_overlap: float = 0.7           # Jaccard overlap of signal bars above which two patterns are one family
    # costs: per side, in basis points of notional (10 bps = 0.10 %)
    cost_bps: float = 5.0
    # splits (fractions of the date range)
    train_frac: float = 0.6
    valid_frac: float = 0.2
    embargo_days: Optional[int] = None  # defaults to max(horizons)
    allow_short: bool = True
    random_seed: int = 7

    def embargo(self) -> int:
        return self.embargo_days if self.embargo_days is not None else max(self.horizons)


# --------------------------------------------------------------------------- #
# Panel helpers
# --------------------------------------------------------------------------- #

@dataclass
class PanelArrays:
    """Numpy views of the panel used by the screener."""

    dates: np.ndarray            # datetime64 per row
    day_idx: np.ndarray          # dense integer day index per row
    sym_idx: np.ndarray          # dense integer symbol index per row
    pos_in_sym: np.ndarray       # row position within its symbol's series
    fwd: Dict[int, np.ndarray]   # horizon -> forward return (nan where unknown)
    base: Dict[int, np.ndarray]  # horizon -> per-symbol base rate, broadcast per row
    n_days: int
    symbols: List[str]


def panel_arrays(panel: pd.DataFrame, horizons: Iterable[int], base_rows: Optional[np.ndarray] = None) -> PanelArrays:
    dates = panel.index.get_level_values("Date").values
    syms = panel.index.get_level_values("Symbol")
    uniq_dates, day_idx = np.unique(dates, return_inverse=True)
    uniq_syms, sym_idx = np.unique(syms.values, return_inverse=True)
    order = np.lexsort((day_idx, sym_idx))
    pos = np.empty(len(panel), dtype=int)
    # position within symbol = rank of day among that symbol's rows
    counts = np.bincount(sym_idx)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    pos[order] = np.arange(len(panel)) - np.repeat(starts, counts)
    fwd, base = {}, {}
    for h in horizons:
        col = panel[f"fwd_{h}"].to_numpy(dtype=float)
        fwd[h] = col
        rows = base_rows if base_rows is not None else np.ones(len(col), dtype=bool)
        valid = rows & ~np.isnan(col)
        sums = np.bincount(sym_idx[valid], weights=col[valid], minlength=len(uniq_syms))
        cnts = np.bincount(sym_idx[valid], minlength=len(uniq_syms))
        pooled = float(col[valid].mean()) if valid.any() else 0.0
        # symbols that have no history inside the base window fall back to the pooled mean
        per_sym = np.divide(sums, cnts, out=np.full_like(sums, pooled), where=cnts > 0)
        base[h] = per_sym[sym_idx]
    return PanelArrays(dates, day_idx, sym_idx, pos, fwd, base, len(uniq_dates), list(uniq_syms))


def date_splits(panel: pd.DataFrame, cfg: MinerConfig) -> Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]:
    """Chronological train / valid / test windows with an embargo gap."""
    dates = panel.index.get_level_values("Date").unique().sort_values()
    n = len(dates)
    e = cfg.embargo()
    i1 = int(n * cfg.train_frac)
    i2 = int(n * (cfg.train_frac + cfg.valid_frac))
    out = {"train": (dates[0], dates[i1 - 1])}
    if i2 - (i1 + e) > 50:
        out["valid"] = (dates[i1 + e], dates[i2 - 1])
    if n - (i2 + e) > 50:
        out["test"] = (dates[i2 + e], dates[n - 1])
    return out


def rows_in(panel: pd.DataFrame, window: Tuple[pd.Timestamp, pd.Timestamp]) -> np.ndarray:
    d = panel.index.get_level_values("Date")
    return np.asarray((d >= window[0]) & (d <= window[1]))


# --------------------------------------------------------------------------- #
# Screening
# --------------------------------------------------------------------------- #

def _screen_one(mask: np.ndarray, arrs: PanelArrays, h: int, window_rows: np.ndarray) -> Optional[Tuple[int, float, float]]:
    """Return (n, mean_excess, t) for a signal mask at horizon ``h``."""
    fwd = arrs.fwd[h]
    rows = mask & window_rows & ~np.isnan(fwd)
    n = int(rows.sum())
    if n == 0:
        return None
    excess = fwd[rows] - arrs.base[h][rows]
    m = excess.mean()
    e = excess - m
    cl = arrs.day_idx[rows] // h
    sums = np.bincount(cl, weights=e)
    g = int((np.bincount(cl) > 0).sum())
    if g < 2:
        return n, float(m), 0.0
    var = (sums ** 2).sum() / n ** 2 * g / (g - 1)
    t = float(m / math.sqrt(var)) if var > 0 else 0.0
    return n, float(m), t


@dataclass
class ScreenResult:
    pattern: Pattern
    horizon: int
    n_signals: int
    mean_excess: float
    t_stat: float
    p_value: float


def screen(
    panel: pd.DataFrame,
    train_rows: np.ndarray,
    cfg: MinerConfig,
    arrs: Optional[PanelArrays] = None,
) -> Tuple[List[ScreenResult], int]:
    """Beam-search over conjunctions.  Returns all evaluated results and the
    number of (pattern, horizon) trials, which is needed for deflation."""
    arrs = arrs or panel_arrays(panel, cfg.horizons, base_rows=train_rows)
    feats = cfg.features or [f for f in feature_columns(panel) if f not in cfg.exclude_features]
    train = panel.loc[train_rows]
    conds = generate_conditions(train, feats, cfg.quantiles)
    n_train = int(train_rows.sum())
    max_rows = int(cfg.max_signal_fraction * n_train)

    masks: Dict[Condition, np.ndarray] = {}
    for c in conds:
        m = c.mask(panel)
        supp = int((m & train_rows).sum())
        if cfg.min_signals <= supp <= max_rows:
            masks[c] = m

    results: List[ScreenResult] = []
    trials = 0

    def evaluate(pattern: Pattern, mask: np.ndarray) -> float:
        nonlocal trials
        best = 0.0
        for h in cfg.horizons:
            r = _screen_one(mask, arrs, h, train_rows)
            trials += 1
            if r is None or r[0] < cfg.min_signals:
                continue
            n, me, t = r
            results.append(ScreenResult(pattern, h, n, me, t, t_to_p(t)))
            best = max(best, abs(t))
        return best

    # level 1
    level: List[Tuple[Pattern, np.ndarray, float]] = []
    for c, m in masks.items():
        p = Pattern((c,))
        level.append((p, m, evaluate(p, m)))
    level.sort(key=lambda x: -x[2])
    seeds = [(p.conditions[0], m) for p, m, _ in level[: cfg.beam_width]]
    beam = level[: cfg.beam_width]

    # deeper levels
    for depth in range(2, cfg.max_conditions + 1):
        seen = set()
        new_level: List[Tuple[Pattern, np.ndarray, float]] = []
        for pat, pmask, _ in beam:
            for cond, cmask in seeds:
                if any(not compatible(cond, pc) for pc in pat.conditions) or cond in pat.conditions:
                    continue
                newp = Pattern(pat.conditions + (cond,))
                if newp in seen:
                    continue
                seen.add(newp)
                m = pmask & cmask
                supp = int((m & train_rows).sum())
                if supp < cfg.min_signals:
                    continue
                new_level.append((newp, m, evaluate(newp, m)))
        if not new_level:
            break
        new_level.sort(key=lambda x: -x[2])
        beam = new_level[: cfg.beam_width]

    return results, trials


# --------------------------------------------------------------------------- #
# Exact trade simulation
# --------------------------------------------------------------------------- #

def simulate_trades(
    panel: pd.DataFrame,
    arrs: PanelArrays,
    pattern: Pattern,
    horizon: int,
    rows: np.ndarray,
    cost_bps: float,
    mask: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Non-overlapping trades: one position per symbol at a time.

    Entry at the next open after the signal, exit at the close ``horizon``
    bars later.  Returns a frame with one row per trade.
    """
    sign = 1.0 if pattern.side == "long" else -1.0
    fwd = arrs.fwd[horizon]
    m = (pattern.mask(panel) if mask is None else mask) & rows & ~np.isnan(fwd)
    idx = np.flatnonzero(m)
    if len(idx) == 0:
        return pd.DataFrame(columns=["date", "symbol", "ret", "excess", "horizon"])
    # sort by symbol then position so the greedy pass is sequential per symbol
    order = np.lexsort((arrs.pos_in_sym[idx], arrs.sym_idx[idx]))
    idx = idx[order]
    keep = np.zeros(len(idx), dtype=bool)
    last_sym, free_from = -1, -1
    syms, poss = arrs.sym_idx[idx], arrs.pos_in_sym[idx]
    for k in range(len(idx)):
        s, p = syms[k], poss[k]
        if s != last_sym:
            last_sym, free_from = s, -1
        if p >= free_from:
            keep[k] = True
            free_from = p + horizon + 1  # exit bar is p+h; next entry signal must be on/after exit bar
    sel = idx[keep]
    cost = 2.0 * cost_bps / 1e4
    raw = fwd[sel]
    ret = sign * raw - cost
    excess = sign * (raw - arrs.base[horizon][sel]) - cost
    return pd.DataFrame(
        {
            "date": arrs.dates[sel],
            "symbol": np.asarray(arrs.symbols)[arrs.sym_idx[sel]],
            "ret": ret,
            "excess": excess,
            "horizon": horizon,
        }
    ).sort_values("date").reset_index(drop=True)


def _years(window_rows: np.ndarray, arrs: PanelArrays) -> float:
    d = arrs.dates[window_rows]
    if len(d) == 0:
        return 0.0
    span = (d.max() - d.min()) / np.timedelta64(1, "D")
    return max(float(span) / 365.25, 1 / 365.25)


def window_stats(trades: pd.DataFrame, years: float) -> Dict:
    ts = trade_stats(trades["ret"].to_numpy(), trades["excess"].to_numpy(), trades["date"].to_numpy())
    d = ts.to_dict()
    d["trades_per_year"] = float(ts.n / years) if years > 0 else 0.0
    d["total_return_sum"] = float(trades["ret"].sum())
    return d


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #

@dataclass
class PatternResult:
    pattern: Pattern
    horizon: int
    screen: Dict
    train: Dict
    valid: Optional[Dict]
    test: Optional[Dict]
    passed_fdr: bool
    deflated_sharpe: float
    selected: bool
    reject_reason: str = ""

    @property
    def name(self) -> str:
        return f"{self.pattern.describe()} | hold {self.horizon}d"

    def to_dict(self) -> Dict:
        return {
            "pattern": self.pattern.to_dict(),
            "description": self.pattern.describe(),
            "horizon": self.horizon,
            "screen": self.screen,
            "train": self.train,
            "valid": self.valid,
            "test": self.test,
            "passed_fdr": self.passed_fdr,
            "deflated_sharpe": self.deflated_sharpe,
            "selected": self.selected,
            "reject_reason": self.reject_reason,
        }


@dataclass
class MiningReport:
    config: MinerConfig
    splits: Dict[str, Tuple[str, str]]
    n_trials: int
    n_screen_candidates: int
    n_fdr_survivors: int
    results: List[PatternResult]
    symbols: List[str]

    @property
    def selected(self) -> List[PatternResult]:
        return [r for r in self.results if r.selected]

    def to_dict(self) -> Dict:
        return {
            "config": asdict(self.config),
            "splits": self.splits,
            "n_trials": self.n_trials,
            "n_screen_candidates": self.n_screen_candidates,
            "n_fdr_survivors": self.n_fdr_survivors,
            "symbols": self.symbols,
            "results": [r.to_dict() for r in self.results],
        }


def _prune_redundant(results: List[PatternResult], panel: pd.DataFrame, train_rows: np.ndarray, max_overlap: float) -> None:
    """Keep one representative per family of patterns that fire on nearly the
    same bars.  Variants of one setup with slightly different thresholds would
    otherwise crowd the report and the portfolio."""
    kept: List[Tuple[PatternResult, np.ndarray]] = []
    for r in results:
        if not r.selected:
            continue
        m = r.pattern.mask(panel) & train_rows
        dup = None
        for k, km in kept:
            inter = np.count_nonzero(m & km)
            union = np.count_nonzero(m | km)
            if union and inter / union >= max_overlap and k.pattern.side == r.pattern.side:
                dup = k
                break
        if dup is None:
            kept.append((r, m))
        else:
            r.selected = False
            r.reject_reason = f"redundant with: {dup.name}"
    # re-sort so that selected patterns come first again
    results.sort(key=lambda x: (not x.selected, -(x.valid["t_stat"] if x.valid else x.train["t_stat"])))


def mine_patterns(
    panel: pd.DataFrame,
    cfg: Optional[MinerConfig] = None,
    windows: Optional[Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]] = None,
    verbose: bool = True,
) -> MiningReport:
    """Run screen -> FDR -> confirm -> select on a feature/label panel."""
    cfg = cfg or MinerConfig()
    windows = windows or date_splits(panel, cfg)
    train_rows = rows_in(panel, windows["train"])
    valid_rows = rows_in(panel, windows["valid"]) if "valid" in windows else None
    test_rows = rows_in(panel, windows["test"]) if "test" in windows else None
    arrs = panel_arrays(panel, cfg.horizons, base_rows=train_rows)

    if verbose:
        print(f"[miner] rows={len(panel):,} symbols={len(arrs.symbols)} train={int(train_rows.sum()):,}")
    screened, n_trials = screen(panel, train_rows, cfg, arrs)
    if verbose:
        print(f"[miner] screened {n_trials:,} (pattern, horizon) trials")

    # multiple-testing correction over every trial that produced a result
    pvals = [r.p_value for r in screened]
    passed = benjamini_hochberg(pvals, cfg.fdr_q) if pvals else np.zeros(0, dtype=bool)
    for r, ok in zip(screened, passed):
        r.passed_fdr = bool(ok)  # type: ignore[attr-defined]
    cands = [r for r in screened if abs(r.t_stat) >= cfg.min_t_screen]
    cands.sort(key=lambda r: -abs(r.t_stat))
    cands = cands[: cfg.top_k_confirm]
    n_fdr = int(sum(1 for r in cands if getattr(r, "passed_fdr", False)))
    if verbose:
        print(f"[miner] {len(cands)} candidates with |t|>={cfg.min_t_screen}; {n_fdr} pass FDR q={cfg.fdr_q}")

    years_train = _years(train_rows, arrs)
    years_valid = _years(valid_rows, arrs) if valid_rows is not None else 0.0
    years_test = _years(test_rows, arrs) if test_rows is not None else 0.0

    results: List[PatternResult] = []
    for r in cands:
        side = "long" if r.mean_excess > 0 else "short"
        if side == "short" and not cfg.allow_short:
            continue
        pat = Pattern(r.pattern.conditions, side=side)
        mask = pat.mask(panel)
        tr = simulate_trades(panel, arrs, pat, r.horizon, train_rows, cfg.cost_bps, mask)
        train_s = window_stats(tr, years_train)
        dsr = deflated_sharpe(tr["ret"].to_numpy(), n_trials) if len(tr) >= 10 else 0.0
        valid_s = test_s = None
        if valid_rows is not None:
            va = simulate_trades(panel, arrs, pat, r.horizon, valid_rows, cfg.cost_bps, mask)
            valid_s = window_stats(va, years_valid)
        if test_rows is not None:
            te = simulate_trades(panel, arrs, pat, r.horizon, test_rows, cfg.cost_bps, mask)
            test_s = window_stats(te, years_test)

        reason = ""
        if not getattr(r, "passed_fdr", False):
            reason = "failed FDR"
        elif train_s["trades_per_year"] < cfg.min_trades_per_year:
            reason = f"too rare: {train_s['trades_per_year']:.1f} trades/yr"
        elif train_s["mean_excess"] <= 0 or train_s["mean"] <= 0:
            # a pattern must make money outright, not only relative to the symbol's drift
            reason = "no edge after costs"
        elif valid_s is not None:
            if valid_s["n"] < cfg.min_valid_trades:
                reason = f"validation: only {valid_s['n']} trades"
            elif valid_s["mean"] <= 0:
                reason = f"validation: net return {valid_s['mean']:.4f} <= 0"
            elif valid_s["mean_excess"] <= 0 or valid_s["t_stat"] < cfg.min_valid_t:
                reason = f"validation: excess={valid_s['mean_excess']:.4f} t={valid_s['t_stat']:.2f}"
            elif valid_s["win_rate"] < cfg.min_valid_win_rate:
                reason = f"validation: win rate {valid_s['win_rate']:.2f}"
        results.append(
            PatternResult(
                pattern=pat,
                horizon=r.horizon,
                screen={"n_signals": r.n_signals, "mean_excess": r.mean_excess, "t_stat": r.t_stat, "p_value": r.p_value},
                train=train_s,
                valid=valid_s,
                test=test_s,
                passed_fdr=bool(getattr(r, "passed_fdr", False)),
                deflated_sharpe=dsr,
                selected=(reason == ""),
                reject_reason=reason,
            )
        )

    # rank: selected first, then by validation t (or train t when no validation)
    def rank_key(x: PatternResult):
        v = x.valid["t_stat"] if x.valid else x.train["t_stat"]
        return (not x.selected, -v)

    results.sort(key=rank_key)
    _prune_redundant(results, panel, train_rows, cfg.max_overlap)
    if verbose:
        print(f"[miner] selected {sum(r.selected for r in results)} patterns")
    return MiningReport(
        config=cfg,
        splits={k: (str(v[0].date()), str(v[1].date())) for k, v in windows.items()},
        n_trials=n_trials,
        n_screen_candidates=len(cands),
        n_fdr_survivors=n_fdr,
        results=results,
        symbols=arrs.symbols,
    )
