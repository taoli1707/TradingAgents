"""Markdown / JSON reporting for a mining run."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd

from .miner import MiningReport, PatternResult


def _pct(x: Optional[float]) -> str:
    return "-" if x is None else f"{100 * x:+.2f}%"


def _row(r: PatternResult) -> str:
    def seg(d: Optional[Dict]) -> str:
        if not d:
            return "- | - | - | -"
        return f"{d['n']} | {d['win_rate']:.0%} | {_pct(d['mean_excess'])} | {d['t_stat']:.2f}"

    return (
        f"| {r.pattern.describe()} | {r.horizon} | {r.train['trades_per_year']:.1f} | "
        f"{seg(r.train)} | {seg(r.valid)} | {seg(r.test)} | {r.deflated_sharpe:.2f} | "
        f"{'yes' if r.selected else r.reject_reason} |"
    )


TABLE_HEADER = (
    "| Pattern | Hold (days) | Trades / yr (train) | Train n | Train win | Train excess | Train t "
    "| Valid n | Valid win | Valid excess | Valid t | Test n | Test win | Test excess | Test t | DSR | Selected |\n"
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
)


def render_markdown(
    report: MiningReport,
    walk_forward: Optional[Dict] = None,
    ml: Optional[Dict] = None,
    backtest: Optional[Dict] = None,
    benchmark: Optional[Dict] = None,
    data_note: str = "",
    max_rejected: int = 25,
) -> str:
    cfg = report.config
    lines: List[str] = []
    lines.append("# Equity pattern mining report\n")
    if data_note:
        lines.append(data_note + "\n")
    lines.append(f"Universe ({len(report.symbols)} symbols): {', '.join(report.symbols)}\n")
    lines.append("## Windows\n")
    for k, (a, b) in report.splits.items():
        lines.append(f"- **{k}**: {a} to {b}")
    lines.append("")
    lines.append("## Selection funnel\n")
    lines.append(f"| Stage | Count |\n|---|---|")
    lines.append(f"| (pattern, horizon) trials screened | {report.n_trials:,} |")
    lines.append(f"| candidates with screen abs(t) >= {cfg.min_t_screen} | {report.n_screen_candidates} |")
    lines.append(f"| pass Benjamini-Hochberg FDR q={cfg.fdr_q} | {report.n_fdr_survivors} |")
    lines.append(f"| pass frequency gate (>= {cfg.min_trades_per_year} trades/yr) and validation gates | {len(report.selected)} |")
    lines.append("")
    lines.append(
        f"Costs: {cfg.cost_bps} basis points per side. Excess = trade return minus the symbol's unconditional "
        f"return over the same horizon, net of costs. t-statistics use standard errors clustered by entry date. "
        f"DSR = Deflated Sharpe Ratio against {report.n_trials:,} trials (probability the edge is not a "
        f"selection artefact; 0.95 or higher is the usual bar).\n"
    )
    lines.append("## Selected patterns\n")
    if report.selected:
        lines.append(TABLE_HEADER)
        for r in report.selected:
            lines.append(_row(r))
    else:
        lines.append("No pattern passed every gate. The strongest in-sample patterns are listed below with the gate that rejected them.")
    lines.append("")
    rejected = [r for r in report.results if not r.selected][:max_rejected]
    if rejected:
        lines.append(f"## Strongest rejected patterns (top {len(rejected)})\n")
        lines.append(TABLE_HEADER)
        for r in rejected:
            lines.append(_row(r))
        lines.append("")
    if walk_forward:
        lines.append("## Walk-forward (re-mine at each fold, trade only afterwards)\n")
        o = walk_forward["oos_stats"]
        lines.append(f"Folds: {walk_forward['n_folds']}. Pooled out-of-sample trades: {o.get('n', 0)}, "
                     f"{o.get('trades_per_year', 0):.1f} per year, win rate {o.get('win_rate', 0):.0%}, "
                     f"mean net return {_pct(o.get('mean'))}, mean excess {_pct(o.get('mean_excess'))}, t = {o.get('t_stat', 0):.2f}.\n")
        lines.append("| Fold | Train | Out of sample | Patterns selected | OOS trades | OOS win | OOS excess | OOS t |\n|---|---|---|---|---|---|---|---|")
        for i, f in enumerate(walk_forward["folds"]):
            s = f["oos_stats"]
            lines.append(f"| {i + 1} | {f['train'][0]} to {f['train'][1]} | {f['oos'][0]} to {f['oos'][1]} | {f['n_selected']} | {s.get('n', 0)} | {s.get('win_rate', 0):.0%} | {_pct(s.get('mean_excess'))} | {s.get('t_stat', 0):.2f} |")
        lines.append("")
        repeated = [s for s in walk_forward.get("stability", []) if s["folds_selected"] > 1]
        lines.append("Setups (same features and directions, thresholds may differ) selected in more than one fold:\n")
        if repeated:
            for s in repeated:
                lines.append(f"- {s['pattern']} ({s['folds_selected']} folds)")
        else:
            lines.append("- none; every fold picked a different setup, which is itself evidence against a stable edge")
        lines.append("")
    if ml:
        lines.append(f"## Gradient-boosting model (hold {ml['horizon']} days, walk-forward)\n")
        o = ml["overall"]
        lines.append(f"Out-of-sample AUC {o.get('auc', float('nan')):.3f}. Trading only the top-decile predictions: "
                     f"{o.get('n', 0)} trades ({o.get('trades_per_year', 0):.1f}/yr), win rate {o.get('win_rate', 0):.0%}, "
                     f"mean net return {_pct(o.get('mean_net'))}, mean excess {_pct(o.get('mean_excess'))}, t = {o.get('t_stat', 0):.2f}.\n")
        lines.append("| Fold | Out of sample | AUC | Top-decile trades | Win | Excess | t |\n|---|---|---|---|---|---|---|")
        for f in ml["fold_stats"]:
            lines.append(f"| {f['fold'] + 1} | {f['oos'][0]} to {f['oos'][1]} | {f.get('auc', float('nan')):.3f} | {f['n']} | {f['win_rate']:.0%} | {_pct(f['mean_excess'])} | {f['t_stat']:.2f} |")
        lines.append("")
        lines.append("Top features by gain: " + ", ".join(f"{d['feature']}" for d in ml["importance"][:10]) + ".\n")
        if ml.get("surrogate_patterns"):
            lines.append("Rules extracted from the model's confident region, traded with the exact simulator on the test window:\n")
            tested = {d["pattern"]: d for d in ml.get("surrogate_test", [])}
            lines.append("| Rule | Test trades | Trades / yr | Win | Excess | t |\n|---|---|---|---|---|---|")
            for p in ml["surrogate_patterns"]:
                d = tested.get(p)
                if d:
                    lines.append(f"| {p} | {d['n']} | {d['trades_per_year']:.1f} | {d['win_rate']:.0%} | {_pct(d['mean_excess'])} | {d['t_stat']:.2f} |")
                else:
                    lines.append(f"| {p} | - | - | - | - | - |")
            lines.append("")
    if backtest:
        lines.append("## Portfolio backtest of the selected patterns (test window)\n")
        m = backtest["metrics"]
        lines.append("| Metric | Patterns | Equal-weight buy and hold |\n|---|---|---|")
        b = benchmark or {}
        for key, label in [("cagr", "CAGR"), ("sharpe", "Sharpe"), ("max_drawdown", "Max drawdown"), ("annual_vol", "Annual volatility"), ("avg_exposure", "Average gross exposure")]:
            lines.append(f"| {label} | {_pct(m.get(key)) if key != 'sharpe' else f'{m.get(key, 0):.2f}'} | {_pct(b.get(key)) if key != 'sharpe' else f'{b.get(key, 0):.2f}' if b else '-'} |")
        lines.append(f"| Trades | {m.get('n_trades', 0)} ({m.get('trades_per_year', 0):.1f}/yr) | - |")
        lines.append(f"| Win rate | {m.get('win_rate', 0):.0%} | - |")
        lines.append(f"| Profit factor | {m.get('profit_factor', 0):.2f} | - |")
        lines.append("")
    lines.append("## How to read this\n")
    lines.append(
        "- A pattern is only worth trading if it is **selected**, its validation and test columns agree in sign, "
        "its DSR is high, and the trade count is large enough for your account to compound.\n"
        "- Patterns that look great in train and die in valid/test are the normal outcome of data mining; "
        "the engine shows them so you can see what was tried.\n"
        "- Everything here is gross of taxes and assumes fills at the next open at the stated cost."
    )
    return "\n".join(lines) + "\n"


def save_outputs(out_dir: str, report: MiningReport, markdown: str, extras: Optional[Dict] = None) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    paths["report_md"] = os.path.join(out_dir, "report.md")
    with open(paths["report_md"], "w") as fh:
        fh.write(markdown)
    paths["report_json"] = os.path.join(out_dir, "report.json")
    payload = report.to_dict()
    payload.update(extras or {})
    with open(paths["report_json"], "w") as fh:
        json.dump(payload, fh, indent=2, default=str)
    paths["patterns_json"] = os.path.join(out_dir, "patterns.json")
    with open(paths["patterns_json"], "w") as fh:
        json.dump(
            [{"pattern": r.pattern.to_dict(), "horizon": r.horizon, "description": r.pattern.describe()} for r in report.selected],
            fh,
            indent=2,
        )
    return paths
