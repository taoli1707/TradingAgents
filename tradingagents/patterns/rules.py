"""Pattern primitives: a pattern is a conjunction of threshold conditions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .features import BOOLEAN_FEATURES

OPS = {
    "<=": np.less_equal,
    ">=": np.greater_equal,
    "<": np.less,
    ">": np.greater,
    "==": np.equal,
}


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    value: float

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        col = frame[self.feature].to_numpy()
        with np.errstate(invalid="ignore"):
            m = OPS[self.op](col, self.value)
        return m & ~np.isnan(col)

    def describe(self) -> str:
        if self.feature in BOOLEAN_FEATURES:
            return self.feature if self.value == 1 else f"not {self.feature}"
        v = self.value
        txt = f"{v:.4g}" if abs(v) < 1e4 else f"{v:.3e}"
        return f"{self.feature} {self.op} {txt}"

    def to_dict(self) -> Dict:
        return {"feature": self.feature, "op": self.op, "value": float(self.value)}

    @staticmethod
    def from_dict(d: Dict) -> "Condition":
        return Condition(d["feature"], d["op"], float(d["value"]))


@dataclass(frozen=True)
class Pattern:
    conditions: Tuple[Condition, ...]
    side: str = "long"  # "long" or "short"

    def __post_init__(self):
        object.__setattr__(self, "conditions", tuple(sorted(self.conditions, key=lambda c: (c.feature, c.op, c.value))))

    def mask(self, frame: pd.DataFrame) -> np.ndarray:
        m = np.ones(len(frame), dtype=bool)
        for c in self.conditions:
            m &= c.mask(frame)
        return m

    def describe(self) -> str:
        body = " AND ".join(c.describe() for c in self.conditions)
        return f"[{self.side}] {body}"

    @property
    def features(self) -> List[str]:
        return [c.feature for c in self.conditions]

    def to_dict(self) -> Dict:
        return {"side": self.side, "conditions": [c.to_dict() for c in self.conditions]}

    @staticmethod
    def from_dict(d: Dict) -> "Pattern":
        return Pattern(tuple(Condition.from_dict(c) for c in d["conditions"]), d.get("side", "long"))

    def __len__(self) -> int:
        return len(self.conditions)


# --------------------------------------------------------------------------- #
# Candidate condition generation
# --------------------------------------------------------------------------- #

# Well-known threshold conventions.  Mining these makes the resulting
# patterns read like classic technical-analysis setups.
NAMED_THRESHOLDS: Dict[str, List[Tuple[str, float]]] = {
    "rsi_14": [("<=", 30.0), ("<=", 40.0), (">=", 60.0), (">=", 70.0)],
    "rsi_2": [("<=", 10.0), ("<=", 25.0), (">=", 75.0), (">=", 90.0)],
    "stoch_k": [("<=", 20.0), (">=", 80.0)],
    "bb_pct_b": [("<=", 0.0), ("<=", 0.2), (">=", 0.8), (">=", 1.0)],
    "cci_20": [("<=", -100.0), (">=", 100.0)],
    "adx_14": [("<=", 20.0), (">=", 25.0), (">=", 40.0)],
    "close_vs_sma200": [("<=", 0.0), (">=", 0.0)],
    "sma50_vs_sma200": [("<=", 0.0), (">=", 0.0)],
    "streak": [("<=", -3.0), ("<=", -5.0), (">=", 3.0), (">=", 5.0)],
    "down_days_10": [(">=", 7.0), ("<=", 3.0)],
    "dist_52w_high": [(">=", -0.02), ("<=", -0.2)],
    "dist_52w_low": [("<=", 0.05)],
}

DEFAULT_QUANTILES = (0.1, 0.2, 0.8, 0.9)


def generate_conditions(
    train: pd.DataFrame,
    features: Sequence[str],
    quantiles: Iterable[float] = DEFAULT_QUANTILES,
    include_named: bool = True,
) -> List[Condition]:
    """Build the universe of single conditions from the *training* data.

    Continuous features get ``<= q`` for lower quantiles and ``>= q`` for
    upper quantiles, where ``q`` is the training-set quantile.  Boolean flags
    get ``== 1``.  Thresholds are fixed from the training window so that the
    out-of-sample evaluation never looks at future distributions.
    """
    conds: List[Condition] = []
    for feat in features:
        if feat not in train.columns:
            continue
        col = train[feat].dropna()
        if col.empty:
            continue
        if feat in BOOLEAN_FEATURES:
            if col.mean() > 0:  # flag fires at least once
                conds.append(Condition(feat, "==", 1.0))
            continue
        if include_named and feat in NAMED_THRESHOLDS:
            for op, val in NAMED_THRESHOLDS[feat]:
                conds.append(Condition(feat, op, val))
        for q in quantiles:
            val = float(col.quantile(q))
            op = "<=" if q < 0.5 else ">="
            conds.append(Condition(feat, op, round(val, 6)))
    # drop exact duplicates
    seen = set()
    out = []
    for c in conds:
        key = (c.feature, c.op, c.value)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def compatible(a: Condition, b: Condition) -> bool:
    """Two conditions can be combined when they are on different features, or
    bracket the same feature from both sides (``x >= lo AND x <= hi``)."""
    if a.feature != b.feature:
        return True
    if a.op == b.op or "==" in (a.op, b.op):
        return False
    lo, hi = (a, b) if a.op.startswith(">") else (b, a)
    return lo.value < hi.value
