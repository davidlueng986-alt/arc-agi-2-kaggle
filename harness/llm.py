"""LLM client interface: OpenAI-compatible HTTP (vLLM) plus an offline MockLLM.

Every component downstream of `LLMClient` is backend-agnostic, which is what makes
the whole harness testable with no GPU. The HTTP client uses only `urllib` from the
standard library so the Kaggle wheelhouse needs nothing beyond vLLM itself.
"""

from __future__ import annotations

import itertools
import json
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .grids import Grid, to_csv

VALID_EFFORTS: Tuple[str, ...] = ("xhigh", "medium", "low")
EFFORT_RANK = {"low": 0, "medium": 1, "xhigh": 2}


class EffortError(ValueError):
    """Raised for a reasoning_effort value the Qwen3.8 chat template rejects."""


def validate_effort(effort: str) -> str:
    if effort not in VALID_EFFORTS:
        raise EffortError(
            f"reasoning_effort={effort!r} is invalid; the Qwen3.8 chat template "
            f"accepts only {VALID_EFFORTS} and raises HTTP 500 otherwise "
            f"(QwenLM/Qwen3.8#217)"
        )
    return effort


@dataclass
class ChatRequest:
    messages: List[Dict[str, str]]
    max_tokens: int = 16384
    temperature: float = 0.9
    top_p: float = 0.95
    seed: Optional[int] = None
    reasoning_effort: str = "xhigh"
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_effort(self.reasoning_effort)


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass
class ChatResponse:
    content: str = ""
    reasoning: str = ""
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""
    elapsed: float = 0.0
    error: Optional[str] = None
    attempts: int = 1
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def truncated(self) -> bool:
        return self.finish_reason == "length"


class LLMClient:
    """Interface every backend implements."""

    name = "base"

    def complete(self, request: ChatRequest) -> ChatResponse:
        raise NotImplementedError

    def health(self) -> bool:
        return True

    def close(self) -> None:
        return None


class HTTPChatClient(LLMClient):
    """Blocking OpenAI-compatible `/v1/chat/completions` client.

    Concurrency is supplied by the caller's thread pool, which keeps the runtime
    dependency list empty and makes each request independently cancellable by the
    budget governor.
    """

    name = "vllm-http"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "qwen3.8-27b",
        api_key: str = "EMPTY",
        timeout: float = 1800.0,
        max_retries: int = 3,
        backoff: float = 2.0,
        effort_mechanism: str = "chat_template_kwargs",
        extra_payload: Optional[Dict[str, Any]] = None,
        logger: Optional[Callable[[str], None]] = None,
    ) -> None:
        if effort_mechanism not in ("chat_template_kwargs", "top_level", "none"):
            raise ValueError(f"unknown effort_mechanism {effort_mechanism!r}")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.backoff = backoff
        self.effort_mechanism = effort_mechanism
        self.extra_payload = dict(extra_payload or {})
        self._log = logger or (lambda _m: None)

    def _payload(self, request: ChatRequest) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": request.messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": False,
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if self.effort_mechanism == "chat_template_kwargs":
            payload["chat_template_kwargs"] = {"reasoning_effort": request.reasoning_effort}
        elif self.effort_mechanism == "top_level":
            payload["reasoning_effort"] = request.reasoning_effort
        payload.update(self.extra_payload)
        return payload

    def complete(self, request: ChatRequest) -> ChatResponse:
        payload = self._payload(request)
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        url = f"{self.base_url}/chat/completions"
        started = time.monotonic()
        last_error = "unknown"

        for attempt in range(1, self.max_retries + 1):
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                return self._parse(raw, time.monotonic() - started, attempt, request.meta)
            except urllib.error.HTTPError as exc:
                detail = ""
                try:
                    detail = exc.read().decode("utf-8", "replace")[:400]
                except Exception:
                    pass
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code in (400, 404, 422):
                    break
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                time.sleep(self.backoff ** attempt)

        return ChatResponse(
            error=last_error,
            elapsed=time.monotonic() - started,
            attempts=self.max_retries,
            meta=dict(request.meta),
        )

    def _parse(
        self, raw: Dict[str, Any], elapsed: float, attempts: int, meta: Dict[str, Any]
    ) -> ChatResponse:
        choices = raw.get("choices") or [{}]
        message = choices[0].get("message") or {}
        usage_raw = raw.get("usage") or {}
        details = usage_raw.get("completion_tokens_details") or {}
        content = message.get("content")
        if content is None:
            content = ""
        elif isinstance(content, list):
            content = "".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        usage = Usage(
            prompt_tokens=int(usage_raw.get("prompt_tokens") or 0),
            completion_tokens=int(usage_raw.get("completion_tokens") or 0),
            reasoning_tokens=int(
                details.get("reasoning_tokens") or usage_raw.get("reasoning_tokens") or 0
            ),
        )
        return ChatResponse(
            content=str(content),
            reasoning=str(reasoning),
            usage=usage,
            finish_reason=str(choices[0].get("finish_reason") or ""),
            elapsed=elapsed,
            attempts=attempts,
            meta=dict(meta),
        )

    def health(self) -> bool:
        url = f"{self.base_url}/models"
        try:
            req = urllib.request.Request(
                url, headers={"Authorization": f"Bearer {self.api_key}"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception:
            return False

    def wait_healthy(self, deadline_ts: float, poll: float = 5.0) -> bool:
        while time.monotonic() < deadline_ts:
            if self.health():
                return True
            time.sleep(poll)
        return self.health()


class RoundRobinClient(LLMClient):
    """Fans requests across several endpoints.

    Used for a 2 x (TP=2) serving topology: FP8 27B needs 13.5 GB/GPU at TP=2, which
    leaves ~6.6 GB for KV, so two independent 2-GPU instances each run a shallower
    batch but the pair recovers most of the aggregate throughput of a single TP=4
    instance while halving the blast radius of an NCCL failure.
    """

    name = "round-robin"

    def __init__(self, clients: Sequence[LLMClient]) -> None:
        if not clients:
            raise ValueError("RoundRobinClient needs at least one client")
        self.clients: List[LLMClient] = list(clients)
        self._counter = itertools.count()

    def complete(self, request: ChatRequest) -> ChatResponse:
        index = next(self._counter) % len(self.clients)
        return self.clients[index].complete(request)

    def health(self) -> bool:
        return any(c.health() for c in self.clients)

    def healthy(self) -> List[LLMClient]:
        return [c for c in self.clients if c.health()]

    def close(self) -> None:
        for client in self.clients:
            client.close()


class MockLLM(LLMClient):
    """Deterministic offline backend used by the test-suite and `eval_local --mock`.

    `answers` maps task_id -> list of ground-truth output grids. Each call returns
    the correct grids with probability `correct_prob`, else a deterministic
    perturbation, so voting, shape-diverse selection and budget accounting all get
    exercised on realistic-ish inputs.
    """

    name = "mock"

    def __init__(
        self,
        answers: Optional[Dict[str, Sequence[Grid]]] = None,
        correct_prob: float = 0.5,
        malformed_prob: float = 0.0,
        thinking_tokens: Optional[Dict[str, int]] = None,
        latency: float = 0.0,
        seed: int = 1234,
        tok_per_sec: float = 400.0,
        answer_fn: Optional[Callable[[ChatRequest], Optional[List[Grid]]]] = None,
    ) -> None:
        self.answers: Dict[str, List[Grid]] = {k: [list(r) for r in v] for k, v in (answers or {}).items()}
        self.correct_prob = correct_prob
        self.malformed_prob = malformed_prob
        self.thinking_tokens = dict(thinking_tokens or {"low": 900, "medium": 4000, "xhigh": 10000})
        self.latency = latency
        self.tok_per_sec = tok_per_sec
        self.rng = random.Random(seed)
        self.answer_fn = answer_fn
        self.calls: List[ChatRequest] = []
        self.tokens_emitted = 0

    def _perturb(self, grid: Grid, salt: int) -> Grid:
        rng = random.Random(salt)
        out = [list(row) for row in grid]
        for _ in range(max(1, len(out) * len(out[0]) // 8)):
            r = rng.randrange(len(out))
            c = rng.randrange(len(out[0]))
            out[r][c] = rng.randrange(10)
        return out

    def _render(self, grids: List[Grid]) -> str:
        lines = ["<think>", "Let me work through the transformation step by step.", "</think>"]
        for i, g in enumerate(grids, start=1):
            lines.append("")
            lines.append(f"ANSWER {i}:")
            lines.append(to_csv(g))
        return "\n".join(lines)

    def complete(self, request: ChatRequest) -> ChatResponse:
        started = time.monotonic()
        self.calls.append(request)
        task_id = str(request.meta.get("task_id", ""))
        truth = self.answers.get(task_id)

        if self.answer_fn is not None:
            grids = self.answer_fn(request)
        elif truth is None:
            grids = [[[0]]]
        else:
            roll = self.rng.random()
            if roll < self.correct_prob:
                grids = [list(map(list, g)) for g in truth]
            elif roll < self.correct_prob + self.malformed_prob:
                grids = None
            else:
                salt = abs(hash((task_id, request.seed))) % 100000
                grids = [self._perturb(g, salt + j) for j, g in enumerate(truth)]

        think = self.thinking_tokens.get(request.reasoning_effort, 1000)
        think = max(64, int(think * self.rng.uniform(0.7, 1.3)))

        if grids is None:
            content = "I could not determine a consistent rule for this puzzle."
        else:
            content = self._render(grids)

        if self.latency:
            time.sleep(self.latency)
        elif self.tok_per_sec:
            time.sleep(min(0.05, think / self.tok_per_sec))

        completion = think + max(8, len(content) // 4)
        self.tokens_emitted += completion
        return ChatResponse(
            content=content,
            reasoning="thinking " * min(think // 2, 50),
            usage=Usage(
                prompt_tokens=int(sum(len(m["content"]) for m in request.messages) / 3.2),
                completion_tokens=completion,
                reasoning_tokens=think,
            ),
            finish_reason="stop",
            elapsed=time.monotonic() - started,
            meta=dict(request.meta),
        )
