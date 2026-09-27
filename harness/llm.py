"""LLM clients: an OpenAI-compatible/vLLM backend and a deterministic mock.

The whole harness talks to this one interface, so every non-LLM component is
testable on a laptop with no GPU and no torch (which is the local situation:
RTX 5060 Laptop 8GB, no torch, cannot host Qwen3.8-27B).

vLLM traps encoded here, all verified upstream:

* ``reasoning_effort`` accepts only ``{xhigh, medium, low}``; passing ``high``
  returns **HTTP 500** (QwenLM/Qwen3.8#217) and the default is ``xhigh``, which
  is unaffordable in a multi-turn loop.  We always send an explicit effort and,
  if the server 500s on it, retry once without the field rather than dying.
* Structured output is unreliable in thinking mode (vLLM#35700), so we never ask
  for ``response_format``/``guided_json``.  The only structure we rely on is
  native tool calling (``--enable-auto-tool-choice --tool-call-parser
  qwen3_coder``), and even that gets a text-markup fallback: the Qwen coder
  template's tool-call tags are recovered from raw text with regexes when the
  server-side parser drops them, exactly as the Duck does.
"""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence

VALID_REASONING_EFFORTS = ("low", "medium", "xhigh")

# The Qwen-coder tool-call markup, assembled from pieces so this source file does
# not contain a literal closing tag (which confuses some code-transport layers).
_LT = chr(60)
_GT = chr(62)
_SLASH = chr(47)
_TC_OPEN = f"{_LT}tool_call{_GT}"
_TC_CLOSE = f"{_LT}{_SLASH}tool_call{_GT}"
_FN_CLOSE = f"{_LT}{_SLASH}function{_GT}"
_PARAM_CLOSE = f"{_LT}{_SLASH}parameter{_GT}"

_TOOL_BLOCK_RE = re.compile(
    re.escape(_TC_OPEN)
    + r"\s*"
    + re.escape(f"{_LT}function=")
    + r"([^>\n]+)"
    + re.escape(_GT)
    + r"\s*(.*?)\s*"
    + re.escape(_FN_CLOSE)
    + r"\s*"
    + re.escape(_TC_CLOSE),
    flags=re.DOTALL | re.IGNORECASE,
)
_PARAM_RE = re.compile(
    re.escape(f"{_LT}parameter=")
    + r"([^>]+)"
    + re.escape(_GT)
    + r"\s*(.*?)\s*"
    + re.escape(_PARAM_CLOSE),
    flags=re.DOTALL | re.IGNORECASE,
)
_THINK_CLOSE = f"{_LT}{_SLASH}think{_GT}"
_THINK_TAG_RE = re.compile(re.escape(f"{_LT}think{_GT}") + "|" + re.escape(_THINK_CLOSE), re.IGNORECASE)
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_+-]*\s*\n?|\n?```\s*$")


class LLMError(RuntimeError):
    pass


class ContextOverflowError(LLMError):
    pass


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    raw_arguments: str = ""


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    text: str = ""
    reasoning: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    def to_usage(self):
        from .budget import Usage as BudgetUsage

        return BudgetUsage(
            prompt_tokens=self.usage.prompt_tokens,
            completion_tokens=self.usage.completion_tokens,
            reasoning_tokens=self.usage.reasoning_tokens,
            total_tokens=self.usage.total_tokens or (self.usage.prompt_tokens + self.usage.completion_tokens),
            calls=1,
        )


@dataclass
class LLMRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = "auto"
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int | None = None
    reasoning_effort: str | None = None
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    """Injectable LLM backend. Every non-LLM component depends only on this."""

    def complete(self, request: LLMRequest) -> LLMResponse:  # pragma: no cover
        ...


def strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_tool_arguments(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        pass
    fenced = strip_code_fence(raw)
    try:
        parsed = json.loads(fenced)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except json.JSONDecodeError:
        return {"code": fenced}


def recover_tool_calls(text: str) -> tuple[ToolCall, ...]:
    """Recover Qwen-coder tool-call markup emitted as plain text.

    The Duck ships exactly this fallback (``_recover_tool_calls_from_markup``):
    the server-side tool parser occasionally leaves the markup in the content
    channel, and losing a turn to that is pure wasted budget.
    """
    if not text or "tool_call" not in text.lower():
        return ()
    calls: list[ToolCall] = []
    seen: set[str] = set()
    for match in _TOOL_BLOCK_RE.finditer(text):
        name = match.group(1).strip()
        if not name:
            continue
        arguments = {
            key.strip(): value
            for key, value in _PARAM_RE.findall(match.group(2) or "")
            if key.strip()
        }
        raw = json.dumps(arguments, ensure_ascii=True)
        if raw in seen:
            continue
        seen.add(raw)
        calls.append(
            ToolCall(
                id=f"markup-{len(calls) + 1}",
                name=name,
                arguments=arguments,
                raw_arguments=raw,
            )
        )
    return tuple(calls)


def _extract_reasoning(message: dict[str, Any]) -> str:
    for key in ("reasoning_content", "reasoning", "thinking"):
        value = message.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _normalize_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return _THINK_TAG_RE.sub("", content).strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, str):
                parts.append(item)
        return _THINK_TAG_RE.sub("", "\n".join(parts)).strip()
    return str(content)


def _is_context_error(text: str) -> bool:
    lowered = (text or "").lower()
    return "maximum context length" in lowered or "too many tokens" in lowered or (
        "context" in lowered and "exceed" in lowered
    )


def _parse_response(payload: dict[str, Any]) -> LLMResponse:
    choices = payload.get("choices") or []
    if not choices:
        raise LLMError(f"server returned no choices: {str(payload)[:300]}")
    message = choices[0].get("message") or {}
    text = _normalize_content(message.get("content"))
    reasoning = _extract_reasoning(message)

    calls: list[ToolCall] = []
    for raw_call in message.get("tool_calls") or []:
        function = raw_call.get("function") or {}
        name = str(function.get("name") or "").strip()
        if not name:
            continue
        raw_arguments = function.get("arguments")
        if isinstance(raw_arguments, dict):
            arguments, raw_text = raw_arguments, json.dumps(raw_arguments)
        else:
            raw_text = str(raw_arguments or "")
            arguments = parse_tool_arguments(raw_text)
        calls.append(
            ToolCall(
                id=str(raw_call.get("id") or f"call-{len(calls) + 1}"),
                name=name,
                arguments=arguments,
                raw_arguments=raw_text,
            )
        )

    if not calls:
        calls = list(recover_tool_calls(text) or recover_tool_calls(reasoning))
        if calls:
            # The markup was the tool call; do not also show it as prose.
            text = _TOOL_BLOCK_RE.sub("", text).strip()

    raw_usage = payload.get("usage") or {}
    details = raw_usage.get("completion_tokens_details") or {}
    prompt_tokens = int(raw_usage.get("prompt_tokens") or 0)
    completion_tokens = int(raw_usage.get("completion_tokens") or 0)
    usage = Usage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=int(raw_usage.get("total_tokens") or (prompt_tokens + completion_tokens)),
        reasoning_tokens=int(details.get("reasoning_tokens") or 0),
    )
    return LLMResponse(
        text=text,
        reasoning=reasoning,
        tool_calls=tuple(calls),
        usage=usage,
        finish_reason=str(choices[0].get("finish_reason") or ""),
        raw=payload,
    )


class VLLMClient:
    """Blocking OpenAI-compatible client (vLLM ``/v1/chat/completions``)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "qwen3.8-27b",
        api_key: str = "EMPTY",
        *,
        timeout: float = 600.0,
        max_retries: int = 2,
        retry_backoff: float = 2.0,
        extra_body: dict[str, Any] | None = None,
        session: Any | None = None,
        drop_effort_on_500: bool = True,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = retry_backoff
        self.extra_body = dict(extra_body or {})
        self.drop_effort_on_500 = drop_effort_on_500
        self._session = session
        self._lock = threading.Lock()
        self._drop_reasoning_effort = False
        self.stats = {"requests": 0, "retries": 0, "effort_dropped": 0}

    # -- injection point for tests -----------------------------------------
    @property
    def session(self):
        if self._session is None:
            import requests  # local import: keeps the package importable without requests

            self._session = requests.Session()
        return self._session

    def complete(self, request: LLMRequest) -> LLMResponse:
        payload = self.build_payload(request)
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        attempt = 0
        last_error: Exception | None = None
        while attempt <= self.max_retries:
            self.stats["requests"] += 1
            try:
                response = self.session.post(url, headers=headers, json=payload, timeout=self.timeout)
            except Exception as exc:
                last_error = exc
                attempt += 1
                self.stats["retries"] += 1
                time.sleep(self.retry_backoff * attempt)
                continue

            status = getattr(response, "status_code", 0)
            if status == 200:
                try:
                    return _parse_response(response.json())
                except Exception as exc:
                    last_error = exc
                    attempt += 1
                    self.stats["retries"] += 1
                    continue

            body_text = ""
            try:
                body_text = response.text[:2000]
            except Exception:
                pass

            if status == 400 and _is_context_error(body_text):
                raise ContextOverflowError(body_text[:500])

            if (
                status == 500
                and self.drop_effort_on_500
                and "reasoning_effort" in payload
                and not self._drop_reasoning_effort
            ):
                # QwenLM/Qwen3.8#217: an unaccepted effort value 500s.  Drop it for
                # the rest of the run instead of losing every turn.
                with self._lock:
                    self._drop_reasoning_effort = True
                payload.pop("reasoning_effort", None)
                self.stats["effort_dropped"] += 1
                continue

            if status in (429, 500, 502, 503, 504):
                last_error = LLMError(f"HTTP {status}: {body_text[:400]}")
                attempt += 1
                self.stats["retries"] += 1
                time.sleep(self.retry_backoff * attempt)
                continue

            raise LLMError(f"HTTP {status}: {body_text[:600]}")

        raise LLMError(f"request failed after {attempt} attempts: {last_error}")

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": request.messages,
            "stream": False,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
        if request.max_tokens:
            payload["max_tokens"] = int(request.max_tokens)
        if request.tools:
            payload["tools"] = request.tools
            payload["tool_choice"] = request.tool_choice or "auto"
        if request.top_k and request.top_k > 0:
            payload["top_k"] = int(request.top_k)
        if request.seed is not None and request.seed >= 0:
            payload["seed"] = int(request.seed)
        payload["chat_template_kwargs"] = {"enable_thinking": True}
        effort = (request.reasoning_effort or "").strip().lower()
        if effort:
            if effort not in VALID_REASONING_EFFORTS:
                raise LLMError(
                    f"reasoning_effort {effort!r} is invalid; Qwen3.8 accepts only "
                    f"{VALID_REASONING_EFFORTS} (passing 'high' returns HTTP 500)"
                )
            if not self._drop_reasoning_effort:
                payload["reasoning_effort"] = effort
        payload.update(request.extra)
        payload.update(self.extra_body)
        return payload


class MockLLM:
    """Deterministic, scriptable backend used by the test suite and ``--mock``.

    Two modes:

    * ``script=[...]`` -- a fixed sequence of responses, cycled if the agent asks
      for more (so a runaway loop cannot hang a test);
    * ``responder=fn`` -- a callable ``LLMRequest -> LLMResponse``, which lets a
      test inspect the exact message list it was sent.
    """

    def __init__(
        self,
        script: Sequence[LLMResponse | dict[str, Any]] | None = None,
        *,
        responder: Callable[[LLMRequest], LLMResponse] | None = None,
        name: str = "mock",
    ) -> None:
        if script is None and responder is None:
            raise ValueError("MockLLM needs either a script or a responder")
        self.script: list[LLMResponse] = [_coerce(item) for item in (script or [])]
        self.responder = responder
        self.name = name
        self.requests: list[LLMRequest] = []
        self.calls = 0

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        index = self.calls
        self.calls += 1
        if self.responder is not None:
            return self.responder(request)
        if not self.script:
            return LLMResponse(text="no script", finish_reason="stop")
        return self.script[index % len(self.script)]

    # convenience ---------------------------------------------------------
    @property
    def last_messages(self) -> list[dict[str, Any]]:
        return self.requests[-1].messages if self.requests else []

    def message_texts(self) -> list[str]:
        from .grids import sequence_text

        out = []
        for request in self.requests:
            out.append([sequence_text(m.get("content")) for m in request.messages])
        return out


def _coerce(item: LLMResponse | dict[str, Any]) -> LLMResponse:
    if isinstance(item, LLMResponse):
        return item
    if not isinstance(item, dict):
        raise TypeError(f"cannot build an LLMResponse from {type(item)!r}")
    data = dict(item)
    calls = []
    for call in data.pop("tool_calls", ()) or ():
        if isinstance(call, ToolCall):
            calls.append(call)
            continue
        if not isinstance(call, dict):
            raise TypeError("tool call must be a dict or ToolCall")
        arguments = call.get("arguments") or {}
        calls.append(
            ToolCall(
                id=str(call.get("id") or f"mock-{len(calls) + 1}"),
                name=str(call.get("name") or "python"),
                arguments=arguments,
                raw_arguments=json.dumps(arguments),
            )
        )
    usage = data.pop("usage", None) or {}
    if not isinstance(usage, Usage):
        usage = Usage(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            total_tokens=int(usage.get("total_tokens") or 0),
            reasoning_tokens=int(usage.get("reasoning_tokens") or 0),
        )
    return LLMResponse(
        text=str(data.pop("text", "") or ""),
        reasoning=str(data.pop("reasoning", "") or ""),
        tool_calls=tuple(calls),
        usage=usage,
        finish_reason=str(data.pop("finish_reason", "") or ""),
        raw=data,
    )


def text_response(text: str, **kwargs: Any) -> LLMResponse:
    return _coerce({"text": text, **kwargs})


def tool_response(code: str, *, text: str = "", call_id: str | None = None, **kwargs: Any) -> LLMResponse:
    return _coerce(
        {
            "text": text,
            "tool_calls": [
                {
                    "id": call_id or "mock-call-1",
                    "name": "python",
                    "arguments": {"code": code},
                }
            ],
            **kwargs,
        }
    )


class ConcurrencyLimitedClient:
    """Wraps a client with a semaphore so N workers cannot stampede one server."""

    def __init__(self, inner: LLMClient, max_concurrent: int) -> None:
        self.inner = inner
        self._semaphore = threading.Semaphore(max(1, int(max_concurrent)))

    def complete(self, request: LLMRequest) -> LLMResponse:
        with self._semaphore:
            return self.inner.complete(request)
