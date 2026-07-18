# Claude Code relay — run TradingAgents with no API key

This fork adds an LLM provider, **`claude_code`** (now the default), that
needs **no API key**. Instead of calling a hosted LLM API, every agent
invocation is written as a JSON file to a relay directory; a Claude Code
session acts as the model, answering each request by writing a response
file. All data tools (yfinance, StockTwits, Reddit) are keyless, so the
whole pipeline runs without any credentials.

## How to run an analysis

1. Open Claude Code in this repo and say something like:
   *"Serve the TradingAgents relay and analyze DOCN for 2026-07-13."*
2. Claude Code starts the graph in the background:
   ```bash
   python scripts/run_relay_analysis.py DOCN 2026-07-13
   ```
3. The graph blocks on its first LLM call and drops a request file in
   `~/.tradingagents/claude_relay/pending/`. The Claude Code session
   reads each `req_*.json`, reasons, writes the matching `res_<id>.json`,
   and repeats (~20–30 requests per run) until the final decision prints.

## Relay protocol (for the serving session)

Directory: `~/.tradingagents/claude_relay/` (override with
`TRADINGAGENTS_RELAY_DIR`). Requests appear in `pending/` as
`req_<seq>_<id>.json`; answer by atomically writing `res_<id>.json` next
to it (write a `.tmp`, then rename). Consumed pairs are moved to `done/`.

Request fields: `id`, `kind`, `messages` (role/content list; roles are
`system|user|assistant|tool`), plus `tools` (OpenAI-style schemas) when
`kind == "chat+tools"`, or `response_schema` (JSON Schema) when
`kind == "structured"`.

Response shapes — exactly one of:

| kind        | expected response file                                          |
|-------------|-----------------------------------------------------------------|
| chat        | `{"content": "your full text answer"}`                          |
| chat+tools  | `{"tool_calls": [{"name": "...", "args": {...}}]}` to call tools, or `{"content": "..."}` when the report is finished |
| structured  | `{"structured": {…object matching response_schema…}}`           |

Serving guidance:
- **Stay in character.** Each request's system/user messages define the
  agent role (Market Analyst, Bear Researcher, Portfolio Manager, …).
  Answer as that agent would, fully and concretely.
- **chat+tools** requests are ReAct steps: request tool calls until you
  have the data you need, then emit the final `content` report. Tool
  results come back in the next request as `role: "tool"` messages.
- **structured** requests must satisfy the JSON Schema exactly (enum
  casing matters, e.g. the 5-tier rating `Buy/Overweight/Hold/Underweight/Sell`).

Timeout: a blocked call waits `TRADINGAGENTS_RELAY_TIMEOUT` seconds
(default 3600) before failing.

## Reverting to a hosted provider

Set env vars (or edit `default_config.py`):

```bash
TRADINGAGENTS_LLM_PROVIDER=openai   # or anthropic, google, ...
TRADINGAGENTS_DEEP_THINK_LLM=gpt-5.5
TRADINGAGENTS_QUICK_THINK_LLM=gpt-5.4-mini
```
