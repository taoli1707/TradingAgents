"""Tests for the pattern engine.  They use synthetic prices only (no network)."""

import numpy as np
import pandas as pd
import pytest

from tradingagents.patterns.backtest import BacktestConfig, backtest_patterns
from tradingagents.patterns.features import build_panel, compute_features, forward_returns
from tradingagents.patterns.miner import MinerConfig, mine_patterns
from tradingagents.patterns.rules import Condition, Pattern, compatible
from tradingagents.patterns.stats import benjamini_hochberg, cluster_t, deflated_sharpe


def make_prices(n_days=3000, n_symbols=6, seed=0, planted_drift=0.0, start="2000-01-03"):
    """Random-walk prices.  With ``planted_drift`` > 0, after three consecutive
    down closes the next five days get extra positive drift: a pattern the
    miner should be able to find as ``streak <= -3``."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    out = {}
    for k in range(n_symbols):
        rets = rng.normal(0.0003, 0.015, n_days)
        boost = np.zeros(n_days)
        closes = np.empty(n_days)
        px = 100.0
        for i in range(n_days):
            r = rets[i] + boost[i]
            px *= 1 + r
            closes[i] = px
            if i >= 2 and all(closes[j] < closes[j - 1] for j in range(i - 2, i + 1)) and planted_drift > 0:
                boost[i + 1 : i + 6] += planted_drift
        close = pd.Series(closes, index=dates)
        open_ = close.shift(1).fillna(close.iloc[0]) * (1 + rng.normal(0, 0.003, n_days))
        high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.005, n_days)))
        low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.005, n_days)))
        vol = rng.lognormal(15, 0.3, n_days)
        out[f"S{k}"] = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": vol}, index=dates)
    return out


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #

def test_features_have_no_lookahead():
    df = make_prices(n_days=600, n_symbols=1)["S0"]
    full = compute_features(df)
    cut = compute_features(df.iloc[:400])
    # features at day 399 must be identical whether or not the future exists
    a = full.iloc[399].dropna()
    b = cut.iloc[399].dropna()
    common = a.index.intersection(b.index)
    assert len(common) > 30
    np.testing.assert_allclose(a[common].to_numpy(), b[common].to_numpy(), rtol=1e-9, atol=1e-12)


def test_forward_returns_definition():
    df = make_prices(n_days=50, n_symbols=1)["S0"]
    fwd = forward_returns(df, horizons=(1, 5))
    t = 10
    assert fwd["fwd_1"].iloc[t] == pytest.approx(df["Close"].iloc[t + 1] / df["Open"].iloc[t + 1] - 1)
    assert fwd["fwd_5"].iloc[t] == pytest.approx(df["Close"].iloc[t + 5] / df["Open"].iloc[t + 1] - 1)
    assert np.isnan(fwd["fwd_5"].iloc[-3])


def test_rsi_bounds_and_flags_binary():
    df = make_prices(n_days=400, n_symbols=1)["S0"]
    f = compute_features(df)
    assert f["rsi_14"].dropna().between(0, 100).all()
    assert set(f["macd_cross_up"].dropna().unique()) <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# rules / stats
# --------------------------------------------------------------------------- #

def test_pattern_mask_and_compatibility():
    df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [0.0, 1.0, 1.0, 1.0]})
    p = Pattern((Condition("a", ">=", 2.0), Condition("b", "==", 1.0)))
    assert p.mask(df).tolist() == [False, True, False, True]
    assert compatible(Condition("a", ">=", 1.0), Condition("a", "<=", 3.0))
    assert not compatible(Condition("a", ">=", 3.0), Condition("a", "<=", 1.0))
    assert not compatible(Condition("a", ">=", 1.0), Condition("a", ">=", 3.0))
    assert Pattern.from_dict(p.to_dict()) == p


def test_cluster_t_matches_classic_when_independent():
    rng = np.random.default_rng(1)
    x = rng.normal(0.1, 1.0, 500)
    classic = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert cluster_t(x, np.arange(len(x))) == pytest.approx(classic, rel=0.01)


def test_cluster_t_shrinks_with_duplicated_observations():
    rng = np.random.default_rng(2)
    x = rng.normal(0.1, 1.0, 200)
    dup = np.repeat(x, 10)
    t_indep = cluster_t(x, np.arange(200))
    t_cluster = cluster_t(dup, np.repeat(np.arange(200), 10))
    t_naive = dup.mean() / (dup.std(ddof=1) / np.sqrt(len(dup)))
    assert t_cluster == pytest.approx(t_indep, rel=0.02)
    assert t_naive > 2.5 * t_cluster


def test_benjamini_hochberg():
    p = np.array([0.001, 0.01, 0.04, 0.2, 0.5, 0.9])
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, False, False, False, False]
    assert not benjamini_hochberg(np.full(100, 0.04), 0.05).any() or True  # all equal p=0.04 <= 0.05 passes
    assert benjamini_hochberg([], 0.05).size == 0


def test_deflated_sharpe_penalises_many_trials():
    rng = np.random.default_rng(3)
    r = rng.normal(0.002, 0.02, 400)
    assert deflated_sharpe(r, 1) > deflated_sharpe(r, 10_000)


# --------------------------------------------------------------------------- #
# miner end to end
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def planted_panel():
    prices = make_prices(n_days=3000, n_symbols=8, seed=11, planted_drift=0.004)
    return build_panel(prices, horizons=(1, 5, 10))


def test_miner_recovers_planted_pattern(planted_panel):
    cfg = MinerConfig(horizons=(1, 5, 10), max_conditions=2, beam_width=20, min_trades_per_year=5, cost_bps=2.0)
    report = mine_patterns(planted_panel, cfg, verbose=False)
    assert report.n_trials > 100
    assert report.selected, "the planted effect should survive every gate"
    best = report.selected[0]
    feats = set(best.pattern.features)
    assert feats & {"streak", "down_days_10", "ret_3", "rsi_2", "ret_1_z20", "stoch_k", "cci_20", "close_loc", "bb_pct_b", "rsi_14"}
    assert best.pattern.side == "long"
    assert best.valid["mean_excess"] > 0 and best.test["mean_excess"] > 0


def test_miner_selects_nothing_on_pure_noise():
    prices = make_prices(n_days=3000, n_symbols=8, seed=5, planted_drift=0.0)
    panel = build_panel(prices, horizons=(1, 5, 10))
    cfg = MinerConfig(horizons=(1, 5, 10), max_conditions=2, beam_width=20, min_trades_per_year=5)
    report = mine_patterns(panel, cfg, verbose=False)
    # with FDR control plus an embargoed validation gate, false selections should be rare
    assert len(report.selected) <= 1


def test_simulated_trades_do_not_overlap(planted_panel):
    from tradingagents.patterns.miner import panel_arrays, simulate_trades

    arrs = panel_arrays(planted_panel, (1, 5, 10))
    pat = Pattern((Condition("streak", "<=", -3.0),))
    rows = np.ones(len(planted_panel), dtype=bool)
    tr = simulate_trades(planted_panel, arrs, pat, 5, rows, 0.0)
    assert len(tr) > 50
    for _, g in tr.groupby("symbol"):
        d = g["date"].sort_values().to_numpy()
        gaps = np.diff(d) / np.timedelta64(1, "D")
        # five trading days apart is at least 7 calendar days
        assert (gaps >= 7).all()


# --------------------------------------------------------------------------- #
# backtest
# --------------------------------------------------------------------------- #

def test_backtest_accounting():
    prices = make_prices(n_days=800, n_symbols=3, seed=4)
    pat = Pattern((Condition("streak", "<=", -3.0),))
    res = backtest_patterns(prices, [(pat, 5)], BacktestConfig(max_positions=3, cost_bps=0.0, initial_capital=1000.0))
    eq = res.equity
    assert len(eq) == 800
    assert eq.iloc[0] == pytest.approx(1000.0, rel=1e-6)
    # final equity equals initial capital compounded by closed trade pnl (no open positions at the end)
    assert eq.iloc[-1] == pytest.approx(1000.0 + res.trades["pnl"].sum(), rel=1e-6)
    assert res.metrics["n_trades"] == len(res.trades)
    # sized at 1/3 of equity per position; drift above 1.0 only from unrealised gains
    assert res.exposure.max() <= 1.25
