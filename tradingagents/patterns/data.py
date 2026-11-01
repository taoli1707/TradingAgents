"""Price loading with a local cache and several interchangeable sources.

All sources return one ``pandas.DataFrame`` per symbol, indexed by a
timezone-naive ``DatetimeIndex`` with the columns
``Open, High, Low, Close, Volume``.  Prices are adjusted for splits and
dividends so that return calculations are continuous across corporate
actions.
"""

from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

PRICE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

LEAN_RAW_BASE = "https://raw.githubusercontent.com/QuantConnect/Lean/master/Data/equity/usa"


def _default_cache_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "dataflows", "data_cache", "patterns")


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise a raw frame to the engine's canonical layout."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    rename = {c: c.title() for c in out.columns}
    out = out.rename(columns=rename)
    if "Adj Close" in out.columns and "Close" in out.columns:
        # yfinance with auto_adjust=False: rescale OHLC by the adjustment ratio
        ratio = out["Adj Close"] / out["Close"]
        for c in ["Open", "High", "Low", "Close"]:
            out[c] = out[c] * ratio
        out = out.drop(columns=["Adj Close"])
    missing = [c for c in PRICE_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError(f"price frame is missing columns {missing}")
    out = out[PRICE_COLUMNS].astype(float)
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])
    out = out[(out["Close"] > 0) & (out["Open"] > 0)]
    return out


# --------------------------------------------------------------------------- #
# Sources
# --------------------------------------------------------------------------- #

def fetch_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download adjusted daily bars from Yahoo Finance."""
    import yfinance as yf  # imported lazily: optional dependency for tests

    df = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        multi_level_index=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no rows for {symbol}")
    return _clean(df)


def _lean_adjust(raw: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Apply a QuantConnect factor file to raw (unadjusted) bars.

    A factor file row ``date, price_factor, split_factor, reference`` applies
    to every bar dated on or before ``date`` and after the previous row's date.
    Adjusted price = raw * price_factor * split_factor.
    Adjusted volume = raw / split_factor.
    """
    f = factors.sort_values("date").reset_index(drop=True)
    idx = np.searchsorted(f["date"].values, raw.index.values, side="left")
    idx = np.clip(idx, 0, len(f) - 1)
    pf = f["price_factor"].values[idx]
    sf = f["split_factor"].values[idx]
    out = raw.copy()
    for c in ["Open", "High", "Low", "Close"]:
        out[c] = out[c] * pf * sf
    out["Volume"] = out["Volume"] / sf
    return out


def drop_before_last_gap(df: pd.DataFrame, max_gap_days: int = 60) -> pd.DataFrame:
    """Keep only the rows after the last gap longer than ``max_gap_days``.

    A multi-month hole in a daily series means the vendor stitched two
    different histories together; indicators across the hole are meaningless.
    """
    if len(df) < 2:
        return df
    gaps = pd.Series(df.index).diff().dt.days.to_numpy()
    big = np.flatnonzero(gaps > max_gap_days)
    return df.iloc[big[-1]:] if len(big) else df


def fetch_lean_github(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Load QuantConnect's sample daily data from GitHub (about 13 symbols, 1998-2021).

    Used as a zero-key fallback for development and tests.  Raw prices are in
    units of 1/10000 dollar and are unadjusted; the companion factor file is
    applied so that the output matches the other sources.
    """
    import requests

    sym = symbol.lower()
    r = requests.get(f"{LEAN_RAW_BASE}/daily/{sym}.zip", timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        csv_bytes = z.read(z.namelist()[0])
    raw = pd.read_csv(
        io.BytesIO(csv_bytes),
        header=None,
        names=["Date", "Open", "High", "Low", "Close", "Volume"],
    )
    raw["Date"] = pd.to_datetime(raw["Date"].str.slice(0, 8), format="%Y%m%d")
    raw = raw.set_index("Date")
    for c in ["Open", "High", "Low", "Close"]:
        raw[c] = raw[c] / 10000.0

    # The sample files reuse tickers of defunct companies before the entity's
    # own listing date; the map file's first row gives that date.
    mr = requests.get(f"{LEAN_RAW_BASE}/map_files/{sym}.csv", timeout=60)
    if mr.status_code == 200 and mr.text.strip():
        first = pd.to_datetime(mr.text.strip().splitlines()[0].split(",")[0], format="%Y%m%d")
        raw = raw.loc[first:]
    raw = drop_before_last_gap(raw)

    fr = requests.get(f"{LEAN_RAW_BASE}/factor_files/{sym}.csv", timeout=60)
    if fr.status_code == 200 and fr.text.strip():
        factors = pd.read_csv(
            io.StringIO(fr.text),
            header=None,
            names=["date", "price_factor", "split_factor", "reference"],
        )
        factors["date"] = pd.to_datetime(factors["date"].astype(str), format="%Y%m%d")
        raw = _lean_adjust(raw, factors)
    df = _clean(raw)
    return df.loc[start:end]


def fetch_csv_dir(directory: str) -> Callable[[str, str, str], pd.DataFrame]:
    """Return a fetcher that reads ``<directory>/<SYMBOL>.csv`` files.

    The CSV needs a date column (``Date`` or the first column) and the columns
    ``Open, High, Low, Close, Volume``; ``Adj Close`` is honoured if present.
    """

    def _fetch(symbol: str, start: str, end: str) -> pd.DataFrame:
        path = os.path.join(directory, f"{symbol}.csv")
        if not os.path.exists(path):
            path = os.path.join(directory, f"{symbol.lower()}.csv")
        df = pd.read_csv(path)
        date_col = "Date" if "Date" in df.columns else df.columns[0]
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)
        return _clean(df).loc[start:end]

    return _fetch


SOURCES: Dict[str, Callable[[str, str, str], pd.DataFrame]] = {
    "yfinance": fetch_yfinance,
    "lean": fetch_lean_github,
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

@dataclass
class LoadReport:
    loaded: List[str]
    failed: Dict[str, str]


def load_prices(
    symbols: Iterable[str],
    start: str = "2000-01-01",
    end: Optional[str] = None,
    source: str = "yfinance",
    cache_dir: Optional[str] = None,
    refresh: bool = False,
    csv_dir: Optional[str] = None,
    min_rows: int = 500,
    report: Optional[LoadReport] = None,
) -> Dict[str, pd.DataFrame]:
    """Load adjusted daily bars for ``symbols``.

    Parameters
    ----------
    source : ``"yfinance"`` (default), ``"lean"`` (QuantConnect sample data on
        GitHub, no key needed) or ``"csv"`` (requires ``csv_dir``).
    cache_dir : Parquet cache.  Defaults to ``tradingagents/dataflows/data_cache/patterns``.
    min_rows : symbols with fewer bars than this are dropped; short histories
        make the statistics unreliable.
    """
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    cache_dir = cache_dir or _default_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)

    if source == "csv":
        if not csv_dir:
            raise ValueError("source='csv' needs csv_dir")
        fetch = fetch_csv_dir(csv_dir)
    else:
        try:
            fetch = SOURCES[source]
        except KeyError as exc:
            raise ValueError(f"unknown source {source!r}; choose from {sorted(SOURCES) + ['csv']}") from exc

    out: Dict[str, pd.DataFrame] = {}
    rep = report if report is not None else LoadReport([], {})
    for sym in symbols:
        sym = sym.upper()
        cache_path = os.path.join(cache_dir, f"{sym}.{source}.parquet")
        df = None
        if not refresh and os.path.exists(cache_path):
            cached = pd.read_parquet(cache_path)
            # a cache hit must cover the requested window
            if len(cached) and cached.index[0] <= pd.Timestamp(start) + pd.Timedelta(days=7) and \
                    cached.index[-1] >= pd.Timestamp(end) - pd.Timedelta(days=7):
                df = cached
        if df is None:
            try:
                df = fetch(sym, start, end)
                df.to_parquet(cache_path)
            except Exception as exc:  # noqa: BLE001 - report and continue
                rep.failed[sym] = f"{type(exc).__name__}: {exc}"
                continue
        df = df.loc[start:end]
        if len(df) < min_rows:
            rep.failed[sym] = f"only {len(df)} rows (< {min_rows})"
            continue
        out[sym] = df
        rep.loaded.append(sym)
    return out
