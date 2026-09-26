"""LLM clients: injectable interface + vLLM (OpenAI-compatible) + MockLLM.

Everything above this module is LLM-agnostic, which is what makes the entire
disambiguation layer testable offline (no GPU): tests and `eval_local --mock`
inject MockLLM; the Kaggle notebook injects VLLMClient pointed at a local
vLLM server.

vLLM traps handled here (verified upstream):
- Qwen3.8 `reasoning_effort` accepts ONLY {xhigh, medium, low}; passing "high"
  raises HTTP 500 (QwenLM/Qwen3.8#217). Default is xhigh — we always pin
  low/medium explicitly to stay inside the token budget.
- Structured output is unreliable in thinking mode (vLLM#35700) -> we never
  request JSON schema; we extract markdown code blocks tolerantly instead.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional, Protocol

from .types import LLMResponse


class LLMClient(Protocol):
    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        reasoning_effort: str = "low",
        seed: Optional[int] = None,
    ) -> LLMResponse: ...


# ---------------------------------------------------------------------------
# vLLM / OpenAI-compatible client (stdlib only — works on Kaggle & locally)
# ---------------------------------------------------------------------------


class VLLMClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "qwen3.8-27b",
        request_timeout: float = 900.0,
        max_retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.request_timeout = request_timeout
        self.max_retries = max_retries
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        reasoning_effort: str = "low",
        seed: Optional[int] = None,
    ) -> LLMResponse:
        assert reasoning_effort in ("low", "medium", "xhigh"), (
            "Qwen3.8 reasoning_effort accepts only low/medium/xhigh ('high' -> HTTP 500)"
        )
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": True,
                "reasoning_effort": reasoning_effort,
            },
        }
        if seed is not None:
            payload["seed"] = seed
        data = json.dumps(payload).encode()
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/chat/completions",
                    data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self.request_timeout) as resp:
                    body = json.loads(resp.read().decode())
                text = body["choices"][0]["message"].get("content") or ""
                reasoning = body["choices"][0]["message"].get("reasoning_content") or ""
                usage = body.get("usage", {})
                pt = int(usage.get("prompt_tokens", 0))
                ct = int(usage.get("completion_tokens", 0))
                self.total_prompt_tokens += pt
                self.total_completion_tokens += ct
                # reasoning_content arrives as a separate field with the qwen3
                # parser; keep it out of `text` (we only parse code blocks).
                del reasoning
                return LLMResponse(text=text, prompt_tokens=pt, completion_tokens=ct, model=self.model)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, KeyError) as e:
                last_err = e
                time.sleep(min(2 ** attempt * 2.0, 30.0))
        raise RuntimeError(f"LLM request failed after {self.max_retries} retries: {last_err}")


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------


class MockLLM:
    """Scriptable stand-in for a real LLM. `responder(messages, call_index)`
    returns the raw assistant text; token counts are approximated as len//4 so
    budget accounting is exercised in tests."""

    def __init__(self, responder: Callable[[list[dict], int], str], model: str = "mock"):
        self.responder = responder
        self.model = model
        self.calls = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 1.0,
        max_tokens: int = 4096,
        reasoning_effort: str = "low",
        seed: Optional[int] = None,
    ) -> LLMResponse:
        text = self.responder(messages, self.calls)
        self.calls += 1
        pt = sum(len(m.get("content", "")) for m in messages) // 4
        ct = len(text) // 4
        self.total_prompt_tokens += pt
        self.total_completion_tokens += ct
        return LLMResponse(text=text, prompt_tokens=pt, completion_tokens=ct, model=self.model)


def scripted_responder(scripts: list[str]) -> Callable[[list[dict], int], str]:
    """Returns responses in order; repeats the last one when exhausted."""

    def responder(messages: list[dict], call_index: int) -> str:
        if not scripts:
            return "no script"
        return scripts[min(call_index, len(scripts) - 1)]

    return responder


# A small library of generic programs used by LibraryMockLLM / --mock runs and
# as the CPU-only fallback inside the Kaggle notebook. These exercise the full
# oracle+disambiguation path; on hard ARC-AGI-2 tasks most fail verification,
# which is exactly the fallback path we must validate.
TEMPLATE_PROGRAMS: list[tuple[str, str]] = [
    (
        "identity",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return grid.copy()
""",
    ),
    (
        "rot90",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(grid, 1))
""",
    ),
    (
        "rot180",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(grid, 2))
""",
    ),
    (
        "rot270",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(grid, 3))
""",
    ),
    (
        "fliplr",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(grid[:, ::-1])
""",
    ),
    (
        "flipud",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(grid[::-1, :])
""",
    ),
    (
        "transpose",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(grid.T)
""",
    ),
    (
        "top_half",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return grid[: grid.shape[0] // 2, :].copy()
""",
    ),
    (
        "left_half",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return grid[:, : grid.shape[1] // 2].copy()
""",
    ),
    (
        "zoom2x",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    return np.kron(grid, np.ones((2, 2), dtype=grid.dtype))
""",
    ),
    (
        "bg_to_zero",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    vals, counts = np.unique(grid, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    out = grid.copy()
    out[out == bg] = 0
    return out
""",
    ),
    (
        "least_to_most",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(counts)
    out = grid.copy()
    out[out == int(vals[order[0]])] = int(vals[order[-1]])
    return out
""",
    ),
    (
        "keep_largest_component",
        """import numpy as np
from collections import deque

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    h, w = g.shape
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    seen = np.zeros_like(g, dtype=bool)
    best, best_size = None, 0
    for i in range(h):
        for j in range(w):
            if seen[i, j] or g[i, j] == bg:
                continue
            color = int(g[i, j])
            q = deque([(i, j)])
            seen[i, j] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if len(cells) > best_size:
                best_size, best = len(cells), cells
    out = np.full_like(g, bg)
    if best:
        for r, c in best:
            out[r, c] = g[r, c]
    return out
""",
    ),
    (
        "frame_border",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    out = np.zeros_like(g)
    out[0, :] = g[0, :]
    out[-1, :] = g[-1, :]
    out[:, 0] = g[:, 0]
    out[:, -1] = g[:, -1]
    return out
""",
    ),
    (
        "gravity_down",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    out = np.full_like(g, bg)
    for c in range(g.shape[1]):
        col = [v for v in g[:, c] if v != bg]
        for k, v in enumerate(reversed(col)):
            out[g.shape[0] - 1 - k, c] = v
    return out
""",
    ),
    (
        "crop_to_content",
        """import numpy as np

def transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid)
    vals, counts = np.unique(g, return_counts=True)
    bg = int(vals[np.argmax(counts)])
    mask = g != bg
    if not mask.any():
        return g[:1, :1].copy()
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return g[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()
""",
    ),
]


def library_responder(order: Optional[list[int]] = None) -> Callable[[list[dict], int], str]:
    """Cycles through TEMPLATE_PROGRAMS (optionally in a custom order)."""
    idxs = order if order is not None else list(range(len(TEMPLATE_PROGRAMS)))

    def responder(messages: list[dict], call_index: int) -> str:
        name, code = TEMPLATE_PROGRAMS[idxs[call_index % len(idxs)]]
        return f"The rule looks like `{name}`.\n\n```python\n{code}```\n"

    return responder


@dataclass
class MockConfig:
    responder: Callable[[list[dict], int], str] = field(default_factory=library_responder)
