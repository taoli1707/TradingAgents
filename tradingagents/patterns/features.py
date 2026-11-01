"""Point-in-time technical features and forward return labels.

Every feature at row ``t`` uses only bars up to and including ``t``'s close.
Forward returns use bars strictly after ``t``: the engine assumes a signal is
read after the close of day ``t`` and the position is opened at the open of
day ``t+1``.  Keeping those two halves separate is what makes the backtests
free of look-ahead bias.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

DEFAULT_HORIZONS = (1, 3, 5, 10, 20)

# Features that are 0/1 flags.  The miner tests them as ``== 1`` rather than
# through quantile thresholds.
BOOLEAN_FEATURES = [
    "macd_cross_up",
    "macd_cross_down",
    "sma50_cross_up_200",
    "sma50_cross_down_200",
    "close_cross_up_sma20",
    "close_cross_down_sma20",
    "rsi14_cross_up_30",
    "rsi14_cross_down_70",
    "new_52w_high",
    "new_52w_low",
    "gap_up_1pct",
    "gap_down_1pct",
    "above_sma200",
    "mkt_above_sma200",
]


def _rsi(close: pd.Series, n: int) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    # Wilder smoothing
    roll_up = up.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    roll_down = down.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.fillna(100.0).where(roll_up.notna(), np.nan)


def _true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up_move = df["High"].diff()
    down_move = -df["Low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = _true_range(df)
    atr = tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _streak(sign: pd.Series) -> pd.Series:
    """Length of the current run of equal non-zero signs (positive for up runs,
    negative for down runs)."""
    s = sign.fillna(0).astype(int).values
    out = np.zeros(len(s), dtype=float)
    run = 0
    for i, v in enumerate(s):
        if v == 0:
            run = 0
        elif i > 0 and np.sign(run) == v:
            run += v
        else:
            run = v
        out[i] = run
    return pd.Series(out, index=sign.index)


def compute_features(df: pd.DataFrame, market: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Compute the indicator feature matrix for one symbol.

    Parameters
    ----------
    df : OHLCV frame as returned by :func:`tradingagents.patterns.data.load_prices`.
    market : optional OHLCV frame of a market proxy (for example SPY).  When
        given, regime features prefixed ``mkt_`` are added.
    """
    c, o, h, l, v = df["Close"], df["Open"], df["High"], df["Low"], df["Volume"]
    f = pd.DataFrame(index=df.index)
    ret = c.pct_change()

    # --- returns / momentum --------------------------------------------- #
    for n in (1, 3, 5, 10, 20, 60, 120):
        f[f"ret_{n}"] = c.pct_change(n)
    f["ret_120_ex20"] = c.shift(20) / c.shift(120) - 1.0  # 12-1 style momentum

    # --- moving averages ------------------------------------------------- #
    sma20 = c.rolling(20).mean()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    ema10 = c.ewm(span=10, adjust=False).mean()
    f["close_vs_sma20"] = c / sma20 - 1.0
    f["close_vs_sma50"] = c / sma50 - 1.0
    f["close_vs_sma200"] = c / sma200 - 1.0
    f["close_vs_ema10"] = c / ema10 - 1.0
    f["sma50_vs_sma200"] = sma50 / sma200 - 1.0
    f["sma20_slope_5"] = sma20 / sma20.shift(5) - 1.0
    f["above_sma200"] = (c > sma200).astype(float)

    # --- oscillators ----------------------------------------------------- #
    f["rsi_14"] = _rsi(c, 14)
    f["rsi_2"] = _rsi(c, 2)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    macds = macd.ewm(span=9, adjust=False).mean()
    f["macd_norm"] = macd / c
    f["macd_hist_norm"] = (macd - macds) / c
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    f["stoch_k"] = 100 * (c - low14) / (high14 - low14).replace(0.0, np.nan)
    tp = (h + l + c) / 3.0
    f["cci_20"] = (tp - tp.rolling(20).mean()) / (0.015 * tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True))
    f["adx_14"] = _adx(df, 14)

    # --- volatility / bands ---------------------------------------------- #
    std20 = c.rolling(20).std()
    f["bb_pct_b"] = (c - (sma20 - 2 * std20)) / (4 * std20).replace(0.0, np.nan)
    f["bb_width"] = 4 * std20 / sma20
    atr14 = _true_range(df).ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    f["atr_pct"] = atr14 / c
    f["rvol_20"] = ret.rolling(20).std() * np.sqrt(252)
    f["rvol_ratio_20_100"] = ret.rolling(20).std() / ret.rolling(100).std()
    f["range_pct"] = (h - l) / c
    f["close_loc"] = (c - l) / (h - l).replace(0.0, np.nan)  # 0 = closed at low, 1 = at high
    f["gap"] = o / c.shift(1) - 1.0
    f["ret_1_z20"] = (ret - ret.rolling(20).mean()) / ret.rolling(20).std()

    # --- volume ---------------------------------------------------------- #
    vol20 = v.rolling(20).mean()
    f["vol_ratio_20"] = v / vol20
    f["vol_z_20"] = (v - vol20) / v.rolling(20).std()
    f["dollar_vol_20_log"] = np.log((c * v).rolling(20).mean())
    obv = (np.sign(ret.fillna(0)) * v).cumsum()
    f["obv_slope_10"] = (obv - obv.shift(10)) / vol20.replace(0.0, np.nan) / 10.0

    # --- 52 week levels / streaks ----------------------------------------- #
    hi252 = c.rolling(252).max()
    lo252 = c.rolling(252).min()
    f["dist_52w_high"] = c / hi252 - 1.0
    f["dist_52w_low"] = c / lo252 - 1.0
    f["new_52w_high"] = (c >= hi252).astype(float)
    f["new_52w_low"] = (c <= lo252).astype(float)
    f["streak"] = _streak(np.sign(ret))
    f["down_days_10"] = (ret < 0).rolling(10).sum()

    # --- crossover flags ------------------------------------------------- #
    def cross_up(a: pd.Series, b) -> pd.Series:
        return ((a > b) & (a.shift(1) <= (b.shift(1) if isinstance(b, pd.Series) else b))).astype(float)

    def cross_down(a: pd.Series, b) -> pd.Series:
        return ((a < b) & (a.shift(1) >= (b.shift(1) if isinstance(b, pd.Series) else b))).astype(float)

    f["macd_cross_up"] = cross_up(macd, macds)
    f["macd_cross_down"] = cross_down(macd, macds)
    f["sma50_cross_up_200"] = cross_up(sma50, sma200)
    f["sma50_cross_down_200"] = cross_down(sma50, sma200)
    f["close_cross_up_sma20"] = cross_up(c, sma20)
    f["close_cross_down_sma20"] = cross_down(c, sma20)
    f["rsi14_cross_up_30"] = cross_up(f["rsi_14"], 30.0)
    f["rsi14_cross_down_70"] = cross_down(f["rsi_14"], 70.0)
    f["gap_up_1pct"] = (f["gap"] > 0.01).astype(float)
    f["gap_down_1pct"] = (f["gap"] < -0.01).astype(float)

    # --- market regime --------------------------------------------------- #
    if market is not None:
        mc = market["Close"].reindex(df.index).ffill()
        msma200 = mc.rolling(200).mean()
        f["mkt_above_sma200"] = (mc > msma200).astype(float)
        f["mkt_ret_20"] = mc.pct_change(20)
        f["mkt_rvol_20"] = mc.pct_change().rolling(20).std() * np.sqrt(252)
        f["rel_ret_20"] = f["ret_20"] - f["mkt_ret_20"]

    f = f.replace([np.inf, -np.inf], np.nan)
    return f


def forward_returns(df: pd.DataFrame, horizons: Iterable[int] = DEFAULT_HORIZONS) -> pd.DataFrame:
    """Forward returns for a signal observed at the close of day ``t``.

    ``fwd_{h}`` = Close[t+h] / Open[t+1] - 1.  Entry at the next open is the
    earliest realistic fill for a signal computed on closing data.  Rows whose
    horizon runs past the end of the data are ``NaN``.
    """
    entry = df["Open"].shift(-1)
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        out[f"fwd_{h}"] = df["Close"].shift(-h) / entry - 1.0
    # worst drawdown during the holding window (for stop-loss analysis)
    return out


def build_panel(
    prices: Dict[str, pd.DataFrame],
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    market_symbol: Optional[str] = None,
) -> pd.DataFrame:
    """Stack features and labels for all symbols into one long frame.

    The result has a ``MultiIndex`` of (Date, Symbol).  Feature columns come
    from :func:`compute_features`; label columns are ``fwd_{h}``.
    """
    market = prices.get(market_symbol) if market_symbol else None
    frames: List[pd.DataFrame] = []
    for sym, df in prices.items():
        feats = compute_features(df, market=market)
        labels = forward_returns(df, horizons)
        joined = pd.concat([feats, labels], axis=1)
        joined["Symbol"] = sym
        frames.append(joined)
    panel = pd.concat(frames)
    panel = panel.set_index("Symbol", append=True)
    panel.index.names = ["Date", "Symbol"]
    return panel.sort_index()


def feature_columns(panel: pd.DataFrame) -> List[str]:
    return [c for c in panel.columns if not c.startswith("fwd_")]


def label_columns(panel: pd.DataFrame) -> List[str]:
    return [c for c in panel.columns if c.startswith("fwd_")]
