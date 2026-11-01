"""Gradient-boosting layer: learn where the indicator space is profitable,
then translate the model's confident region back into explicit patterns.

The model is trained walk-forward (expanding window with an embargo) so every
reported number is out of sample.  A shallow surrogate decision tree fitted to
the model's high-confidence region yields human-readable rules, which are
then handed to the same exact trade simulator the rule miner uses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .features import feature_columns
from .rules import Condition, Pattern
from .stats import cluster_t, t_to_p


@dataclass
class MLConfig:
    horizon: int = 5
    n_folds: int = 4
    min_train_years: float = 5.0
    embargo_days: int = 20
    top_fraction: float = 0.10       # trade when predicted probability is in the top decile of train
    params: Dict = field(default_factory=lambda: {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 15,
        "min_child_samples": 300,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 5.0,
        "verbose": -1,
        "seed": 7,
    })
    n_estimators: int = 300
    surrogate_depth: int = 3
    surrogate_min_leaf: int = 400
    cost_bps: float = 5.0
    exclude_features: Tuple[str, ...] = ("dollar_vol_20_log",)


@dataclass
class MLResult:
    horizon: int
    oos: pd.DataFrame            # date, symbol, prob, fwd, excess, fold
    fold_stats: List[Dict]
    overall: Dict
    importance: pd.DataFrame
    surrogate_patterns: List[Pattern]

    def to_dict(self) -> Dict:
        return {
            "horizon": self.horizon,
            "overall": self.overall,
            "fold_stats": self.fold_stats,
            "importance": self.importance.head(20).to_dict(orient="records"),
            "surrogate_patterns": [p.describe() for p in self.surrogate_patterns],
        }


def _labels(panel: pd.DataFrame, horizon: int, train_rows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    fwd = panel[f"fwd_{horizon}"].to_numpy(dtype=float)
    sym = panel.index.get_level_values("Symbol")
    masked = pd.Series(fwd).where(pd.Series(train_rows))
    base = masked.groupby(sym.values).transform("mean").to_numpy()
    base = np.where(np.isnan(base), np.nanmean(masked.to_numpy()), base)
    excess = fwd - base
    return fwd, excess


def _top_decile_stats(df: pd.DataFrame, years: float, cost: float) -> Dict:
    if df.empty:
        return {"n": 0, "trades_per_year": 0.0, "mean_excess": 0.0, "win_rate": 0.0, "t_stat": 0.0, "p_value": 1.0, "auc": float("nan")}
    net = df["fwd"].to_numpy() - cost
    exc = df["excess"].to_numpy() - cost
    t = cluster_t(exc, df["date"].to_numpy())
    return {
        "n": int(len(df)),
        "trades_per_year": float(len(df) / years),
        "mean_net": float(net.mean()),
        "mean_excess": float(exc.mean()),
        "win_rate": float((net > 0).mean()),
        "t_stat": float(t),
        "p_value": t_to_p(t),
    }


def train_walk_forward(panel: pd.DataFrame, cfg: Optional[MLConfig] = None, verbose: bool = True) -> MLResult:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score

    cfg = cfg or MLConfig()
    feats = [f for f in feature_columns(panel) if f not in cfg.exclude_features]
    X_all = panel[feats].to_numpy(dtype=float)
    dates = panel.index.get_level_values("Date")
    syms = panel.index.get_level_values("Symbol")
    uniq = dates.unique().sort_values()
    first_oos = uniq[0] + pd.Timedelta(days=int(round(365.25 * cfg.min_train_years)))
    oos_dates = uniq[uniq >= first_oos]
    if len(oos_dates) < cfg.n_folds * 60:
        raise ValueError("not enough history for walk-forward ML")
    chunks = np.array_split(np.arange(len(oos_dates)), cfg.n_folds)
    cost = 2.0 * cfg.cost_bps / 1e4

    preds, fold_stats, importances = [], [], []
    for k, ch in enumerate(chunks):
        oos_start, oos_end = oos_dates[ch[0]], oos_dates[ch[-1]]
        train_end = uniq[np.searchsorted(uniq.values, oos_start.to_datetime64()) - 1 - cfg.embargo_days]
        train_rows = np.asarray(dates <= train_end)
        oos_rows = np.asarray((dates >= oos_start) & (dates <= oos_end))
        fwd, excess = _labels(panel, cfg.horizon, train_rows)
        y = (excess > 0).astype(int)
        ok = ~np.isnan(fwd) & ~np.isnan(X_all).all(axis=1)
        tr = train_rows & ok
        te = oos_rows & ok
        model = lgb.LGBMClassifier(n_estimators=cfg.n_estimators, **cfg.params)
        model.fit(X_all[tr], y[tr])
        p_tr = model.predict_proba(X_all[tr])[:, 1]
        thresh = float(np.quantile(p_tr, 1 - cfg.top_fraction))
        p_te = model.predict_proba(X_all[te])[:, 1]
        df = pd.DataFrame({
            "date": dates[te], "symbol": syms[te], "prob": p_te,
            "fwd": fwd[te], "excess": excess[te], "fold": k, "threshold": thresh,
        })
        preds.append(df)
        years = max((oos_end - oos_start).days / 365.25, 1 / 365.25)
        top = df[df["prob"] >= thresh]
        st = _top_decile_stats(top, years, cost)
        try:
            st["auc"] = float(roc_auc_score((df["excess"] > 0).astype(int), df["prob"]))
        except ValueError:
            st["auc"] = float("nan")
        st.update({"fold": k, "oos": (str(oos_start.date()), str(oos_end.date())), "threshold": thresh})
        fold_stats.append(st)
        importances.append(pd.Series(model.booster_.feature_importance("gain"), index=feats))
        if verbose:
            print(f"[ml] fold {k + 1}/{cfg.n_folds} oos {oos_start.date()}..{oos_end.date()} auc={st['auc']:.3f} top-decile n={st['n']} excess={st['mean_excess']:.4f} t={st['t_stat']:.2f}")

    oos = pd.concat(preds, ignore_index=True)
    top = oos[oos["prob"] >= oos["threshold"]]
    total_years = max((oos_dates[-1] - oos_dates[0]).days / 365.25, 1 / 365.25)
    overall = _top_decile_stats(top, total_years, cost)
    try:
        overall["auc"] = float(roc_auc_score((oos["excess"] > 0).astype(int), oos["prob"]))
    except ValueError:
        overall["auc"] = float("nan")
    imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    importance = imp.reset_index()
    importance.columns = ["feature", "gain"]

    # surrogate rules from the final model's confident region
    surrogate = extract_rules(panel, feats, model, thresh, cfg)
    return MLResult(cfg.horizon, oos, fold_stats, overall, importance, surrogate)


def extract_rules(panel: pd.DataFrame, feats: List[str], model, threshold: float, cfg: MLConfig) -> List[Pattern]:
    """Fit a shallow tree to ``model_prob >= threshold`` and return the leaves
    that are mostly inside the confident region as explicit patterns."""
    from sklearn.tree import DecisionTreeClassifier

    X = panel[feats].to_numpy(dtype=float)
    ok = ~np.isnan(X).any(axis=1)
    X = X[ok]
    target = (model.predict_proba(X)[:, 1] >= threshold).astype(int)
    if target.sum() < cfg.surrogate_min_leaf:
        return []
    tree = DecisionTreeClassifier(max_depth=cfg.surrogate_depth, min_samples_leaf=cfg.surrogate_min_leaf, random_state=0)
    tree.fit(X, target)
    t = tree.tree_
    patterns: List[Pattern] = []

    def walk(node: int, conds: List[Condition]):
        if t.children_left[node] == -1:
            counts = t.value[node][0]
            purity = counts[1] / counts.sum() if counts.sum() else 0.0
            if purity >= 0.5 and conds:
                patterns.append(Pattern(tuple(conds)))
            return
        f = feats[t.feature[node]]
        thr = float(t.threshold[node])
        walk(t.children_left[node], conds + [Condition(f, "<=", round(thr, 6))])
        walk(t.children_right[node], conds + [Condition(f, ">", round(thr, 6))])

    walk(0, [])
    return patterns
