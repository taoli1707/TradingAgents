"""Smoke test for the claude_code relay client (no API key, no network).

Exercises all three response shapes by playing both sides of the relay
in-process: a background thread acts as the responder.

    python scripts/smoke_relay.py
"""

import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Isolated relay dir so a real serving session never sees these requests.
_tmp = tempfile.mkdtemp(prefix="relay_smoke_")
os.environ["TRADINGAGENTS_RELAY_DIR"] = _tmp

from langchain_core.tools import tool
from pydantic import BaseModel

from tradingagents.llm_clients import create_llm_client

PENDING = Path(_tmp) / "pending"


def responder():
    """Answer every request with a canned, shape-appropriate response."""
    answered: set[str] = set()
    deadline = time.time() + 30
    while len(answered) < 3 and time.time() < deadline:
        for req_path in sorted(PENDING.glob("req_*.json")):
            req = json.loads(req_path.read_text(encoding="utf-8"))
            if req["id"] in answered:
                continue
            kind = req["kind"]
            if kind == "chat":
                res = {"content": "PONG"}
            elif kind == "chat+tools":
                res = {"tool_calls": [{"name": req["tools"][0]["function"]["name"],
                                       "args": {"ticker": "DOCN"}}]}
            else:  # structured
                res = {"structured": {"rating": "Buy", "reasoning": "canned"}}
            out = PENDING / f"res_{req['id']}.json"
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(res), encoding="utf-8")
            tmp.replace(out)
            answered.add(req["id"])
        time.sleep(0.1)


@tool
def get_stock_data(ticker: str) -> str:
    """Fetch stock data for a ticker."""
    return "not actually called"


class MiniPlan(BaseModel):
    rating: str
    reasoning: str


def main():
    client = create_llm_client(provider="claude_code", model="claude-code-relay")
    llm = client.get_llm()
    llm.timeout_s = 30  # fail fast in the smoke test

    t = threading.Thread(target=responder, daemon=True)
    t.start()

    # 1. plain chat
    r1 = llm.invoke("ping")
    assert r1.content == "PONG", f"chat failed: {r1.content!r}"
    print("[1/3] chat            OK ->", r1.content)

    # 2. tool-calling (the analyst ReAct pattern)
    r2 = llm.bind_tools([get_stock_data]).invoke("analyze DOCN")
    assert r2.tool_calls and r2.tool_calls[0]["name"] == "get_stock_data", r2
    assert r2.tool_calls[0]["args"] == {"ticker": "DOCN"}
    print("[2/3] chat+tools      OK ->", r2.tool_calls[0]["name"], r2.tool_calls[0]["args"])

    # 3. structured output (the manager/trader pattern)
    r3 = llm.with_structured_output(MiniPlan).invoke("decide")
    assert isinstance(r3, MiniPlan) and r3.rating == "Buy", r3
    print("[3/3] structured      OK ->", r3)

    print("\nALL RELAY SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
