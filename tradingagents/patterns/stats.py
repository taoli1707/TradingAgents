"""Statistics that keep the pattern miner honest.

* cluster-robust t-statistics (signals on the same day, or inside the same
  holding window, are not independent observations);
* Benjamini-Hochberg false-discovery-rate control over all tested patterns;
* the Deflated Sharpe Ratio of Bailey & Lopez de Prado (2014), which asks how
  likely the best of ``n_trials`` strategies is to look this good by chance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from scipy import stats as sps


def cluster_t(x: np.ndarray, clusters: np.ndarray) -> float:
    """t-statistic of ``mean(x)`` with standard errors clustered by ``clusters``.

    Var(mean) = sum over clusters of (sum of within-cluster residuals)^2 / n^2.
    Falls back to the classical t when every observation is its own cluster.
    """
    n = len(x)
    if n < 2:
        return 0.0
    m = x.mean()
    e = x - m
    _, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=e)
    g = len(sums)
    if g < 2:
        return 0.0
    var = (sums ** 2).sum() / n ** 2
    var *= g / (g - 1)  # small-sample cluster correction
    if var <= 0:
        return 0.0
    return float(m / np.sqrt(var))


def t_to_p(t: float, two_sided: bool = True) -> float:
    p = sps.norm.sf(abs(t))
    return float(2 * p if two_sided else p)


def benjamini_hochberg(pvals: Sequence[float], q: float = 0.10) -> np.ndarray:
    """Return a boolean array: which hypotheses are rejected at FDR ``q``."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresh = q * (np.arange(1, n + 1) / n)
    passed = ranked <= thresh
    if not passed.any():
        return np.zeros(n, dtype=bool)
    k = np.max(np.where(passed)[0])
    out = np.zeros(n, dtype=bool)
    out[order[: k + 1]] = True
    return out


def expected_max_sharpe(n_trials: int, sr_var: float) -> float:
    """Expected maximum Sharpe ratio among ``n_trials`` zero-mean strategies
    whose Sharpe ratios have variance ``sr_var`` (False Strategy Theorem)."""
    if n_trials <= 1 or sr_var <= 0:
        return 0.0
    gamma = 0.5772156649015329
    z1 = sps.norm.ppf(1 - 1.0 / n_trials)
    z2 = sps.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(sr_var) * ((1 - gamma) * z1 + gamma * z2))


def probabilistic_sharpe(sr: float, sr0: float, n: int, skew: float, kurt: float) -> float:
    """P(true SR > sr0) given an observed per-period SR over ``n`` observations.

    ``kurt`` is the (non-excess) kurtosis, so 3 for a Gaussian.
    """
    if n < 3:
        return 0.0
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4.0 * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(n - 1) / denom
    return float(sps.norm.cdf(z))


def deflated_sharpe(returns: np.ndarray, n_trials: int, sr_var: Optional[float] = None) -> float:
    """Deflated Sharpe Ratio: PSR against the expected max SR of ``n_trials``.

    ``returns`` are per-trade (or per-period) returns of the selected
    strategy.  ``sr_var`` is the variance of the per-trade Sharpe across the
    trials; when unknown we use the selected strategy's own squared standard
    error as a conservative stand-in.
    """
    r = np.asarray(returns, dtype=float)
    n = len(r)
    if n < 10 or r.std() == 0:
        return 0.0
    sr = r.mean() / r.std(ddof=1)
    skew = float(sps.skew(r))
    kurt = float(sps.kurtosis(r, fisher=False))
    if sr_var is None:
        sr_var = 1.0 / n
    sr0 = expected_max_sharpe(n_trials, sr_var)
    return probabilistic_sharpe(sr, sr0, n, skew, kurt)


@dataclass
class TradeStats:
    n: int
    mean: float
    mean_excess: float
    median: float
    win_rate: float
    profit_factor: float
    t_stat: float
    p_value: float
    sharpe_per_trade: float
    worst: float
    best: float

    def to_dict(self):
        return self.__dict__.copy()


def trade_stats(ret: np.ndarray, excess: np.ndarray, clusters: np.ndarray) -> TradeStats:
    """Summarise a set of trade returns (already net of costs if desired)."""
    ret = np.asarray(ret, dtype=float)
    excess = np.asarray(excess, dtype=float)
    n = len(ret)
    if n == 0:
        return TradeStats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    gains = ret[ret > 0].sum()
    losses = -ret[ret < 0].sum()
    pf = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0
    t = cluster_t(excess, clusters)
    sd = ret.std(ddof=1) if n > 1 else 0.0
    return TradeStats(
        n=int(n),
        mean=float(ret.mean()),
        mean_excess=float(excess.mean()),
        median=float(np.median(ret)),
        win_rate=float((ret > 0).mean()),
        profit_factor=pf,
        t_stat=t,
        p_value=t_to_p(t),
        sharpe_per_trade=float(ret.mean() / sd) if sd > 0 else 0.0,
        worst=float(ret.min()),
        best=float(ret.max()),
    )
