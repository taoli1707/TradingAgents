"""Claude Code relay client — run TradingAgents with NO API key.

This provider does not call any LLM HTTP API. Instead, every model
invocation is serialized to a JSON *request file* in a relay directory,
and the client blocks until a matching *response file* appears. The
responder is a Claude Code session (the interactive agent the user is
already talking to), which watches the relay directory, reasons over
each request, and writes the response. Authentication rides on the
user's existing Claude Code login — no ANTHROPIC_API_KEY / OPENAI_API_KEY
is ever read.

Protocol (all paths inside the relay dir, default
``~/.tradingagents/claude_relay``, override with ``TRADINGAGENTS_RELAY_DIR``):

  pending/req_<seq>_<id>.json   written by this client, atomically
  pending/res_<id>.json         written by the responder, atomically
  done/                         both files are moved here after consumption

Request JSON::

  {
    "id": "a1b2c3d4",
    "model": "claude-code-relay",
    "kind": "chat" | "chat+tools" | "structured",
    "messages": [{"role": "system|user|assistant|tool", "content": "...",
                   "tool_calls": [...]?, "tool_call_id": "..."?}],
    "tools": [<openai-style tool schemas>]?,        # kind == chat+tools
    "response_schema": {<json schema>}?             # kind == structured
  }

Response JSON (exactly one of):

  {"content": "final text answer"}
  {"tool_calls": [{"name": "get_stock_data", "args": {...}}, ...]}
  {"structured": {<object matching response_schema>}}

The three shapes map onto the three ways agents in this repo consume an
LLM: plain generation (researchers, risk debators), ReAct tool loops
(market/news/fundamentals analysts, via ``bind_tools`` +
``result.tool_calls``), and typed outputs (managers/trader/sentiment,
via ``with_structured_output``).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langchain_core.utils.function_calling import convert_to_openai_tool

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

DEFAULT_RELAY_DIR = os.path.join(os.path.expanduser("~"), ".tradingagents", "claude_relay")
# One graph run makes ~20-30 sequential LLM calls and the responder is an
# interactive agent, so individual waits must be generous.
DEFAULT_TIMEOUT_S = 3600.0
POLL_INTERVAL_S = 0.5


def _relay_dir() -> Path:
    return Path(os.environ.get("TRADINGAGENTS_RELAY_DIR", DEFAULT_RELAY_DIR))


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def _normalize_text(content: Any) -> str:
    """Flatten LangChain message content (str or block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _serialize_messages(messages: list[BaseMessage]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": _normalize_text(m.content)})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": _normalize_text(m.content)})
        elif isinstance(m, AIMessage):
            entry: dict = {"role": "assistant", "content": _normalize_text(m.content)}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {"id": tc.get("id"), "name": tc.get("name"), "args": tc.get("args")}
                    for tc in m.tool_calls
                ]
            out.append(entry)
        elif isinstance(m, ToolMessage):
            out.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": _normalize_text(m.content),
                }
            )
        else:  # ("human", "...") tuples arrive pre-coerced by LangChain; be lenient anyway
            out.append({"role": "user", "content": _normalize_text(getattr(m, "content", m))})
    return out


def _extract_json_object(text: str) -> dict:
    """Parse a JSON object out of text that may carry markdown fences or prose."""
    text = text.strip()
    if text.startswith("```"):
        first_nl = text.index("\n")
        text = text[first_nl + 1 :]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in relay response")
    return json.loads(text[start : end + 1])


class ChatClaudeCode(BaseChatModel):
    """LangChain chat model backed by the Claude Code file relay."""

    model_name: str = "claude-code-relay"
    timeout_s: float = DEFAULT_TIMEOUT_S
    temperature: float | None = None  # accepted for config parity; advisory only

    @property
    def _llm_type(self) -> str:
        return "claude-code-relay"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    # -- transport ---------------------------------------------------------

    def _roundtrip(self, request: dict) -> dict:
        relay = _relay_dir()
        pending = relay / "pending"
        done = relay / "done"
        pending.mkdir(parents=True, exist_ok=True)
        done.mkdir(parents=True, exist_ok=True)

        req_id = uuid.uuid4().hex[:12]
        request["id"] = req_id
        request["model"] = self.model_name
        request["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        seq = f"{time.time():017.6f}".replace(".", "")
        req_path = pending / f"req_{seq}_{req_id}.json"
        res_path = pending / f"res_{req_id}.json"
        _atomic_write_json(req_path, request)
        logger.info("relay: wrote %s (kind=%s), waiting for response", req_path.name, request["kind"])

        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            if res_path.exists():
                try:
                    payload = json.loads(res_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    time.sleep(POLL_INTERVAL_S)  # responder mid-write; retry
                    continue
                # archive the exchange for auditability
                try:
                    req_path.replace(done / req_path.name)
                    res_path.replace(done / res_path.name)
                except OSError:
                    pass
                return payload
            time.sleep(POLL_INTERVAL_S)

        raise TimeoutError(
            f"Claude Code relay: no response for request {req_id} within "
            f"{self.timeout_s:.0f}s. Is a Claude Code session serving "
            f"{pending}? (See CLAUDE_RELAY.md)"
        )

    # -- generation --------------------------------------------------------

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools")
        response_schema = kwargs.get("response_schema")

        request: dict = {"messages": _serialize_messages(messages)}
        if tools:
            request["kind"] = "chat+tools"
            request["tools"] = tools
        elif response_schema:
            request["kind"] = "structured"
            request["response_schema"] = response_schema
        else:
            request["kind"] = "chat"

        payload = self._roundtrip(request)

        if "tool_calls" in payload and payload["tool_calls"]:
            tool_calls = [
                {
                    "id": tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    "name": tc["name"],
                    "args": tc.get("args", {}),
                    "type": "tool_call",
                }
                for tc in payload["tool_calls"]
            ]
            message = AIMessage(content=payload.get("content", ""), tool_calls=tool_calls)
        elif "structured" in payload:
            message = AIMessage(content=json.dumps(payload["structured"]))
        else:
            message = AIMessage(content=str(payload.get("content", "")))

        return ChatResult(generations=[ChatGeneration(message=message)])

    # -- agent-facing surface ------------------------------------------------

    def bind_tools(self, tools: list, **kwargs: Any) -> Any:
        openai_tools = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=openai_tools, **kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        json_schema = schema.model_json_schema() if hasattr(schema, "model_json_schema") else schema

        def _parse(message: AIMessage):
            data = _extract_json_object(_normalize_text(message.content))
            if hasattr(schema, "model_validate"):
                return schema.model_validate(data)
            return data

        return self.bind(response_schema=json_schema) | RunnableLambda(_parse)


class ClaudeCodeClient(BaseLLMClient):
    """No-API-key client: serves LLM calls through a Claude Code session."""

    provider = "claude_code"

    def get_llm(self) -> Any:
        kwargs = dict(self.kwargs)
        timeout_s = float(os.environ.get("TRADINGAGENTS_RELAY_TIMEOUT", DEFAULT_TIMEOUT_S))
        return ChatClaudeCode(model_name=self.model, timeout_s=timeout_s, **kwargs)

    def validate_model(self) -> bool:
        # Any model label is fine — the responder is whatever Claude model
        # is driving the serving session.
        return True
