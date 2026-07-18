"""Run a TradingAgents analysis over the Claude Code relay (no API key).

Usage:
    python scripts/run_relay_analysis.py TICKER [YYYY-MM-DD]

The run blocks while a Claude Code session serves the relay requests in
~/.tradingagents/claude_relay/pending/ — see CLAUDE_RELAY.md.
"""

import sys
from datetime import date
from pathlib import Path

# Windows consoles/redirects default to cp1252; agent output is UTF-8.
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)
    ticker = sys.argv[1].upper()
    trade_date = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()

    config = DEFAULT_CONFIG.copy()
    print(f"[relay] provider={config['llm_provider']} ticker={ticker} date={trade_date}",
          flush=True)

    ta = TradingAgentsGraph(debug=True, config=config)
    final_state, decision = ta.propagate(ticker, trade_date)

    print("\n" + "=" * 72)
    print(f"FINAL RATING for {ticker} on {trade_date}: {decision}")
    print("=" * 72)
    print(final_state["final_trade_decision"])


if __name__ == "__main__":
    main()
