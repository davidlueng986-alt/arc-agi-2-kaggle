"""LLM clients: an injectable interface with three backends.

- OpenAICompatClient: any OpenAI-compatible endpoint (vLLM). Stdlib-only
  (urllib via asyncio.to_thread) so it works in an offline Kaggle rerun with no
  extra packages. Passes reasoning_effort through chat_template_kwargs
  (vLLM/qwen3 convention; only {xhigh, medium, low} are legal — `high` is HTTP 500).
- LiteLLMClient: keeps Poetiq's acompletion path working (optional import).
- MockLLM: deterministic scriptable backend for tests and dry runs.
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Callable, Optional

from harness.budget import TokenLedger
from harness.types import LLMResponse

VALID_EFFORTS = ("xhigh", "medium", "low")


class LLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        prompt: str,
        *,
        tag: str = "generic",
        temperature: float = 1.0,
        max_tokens: int = 2048,
        reasoning_effort: Optional[str] = "low",
        timeout: float = 300.0,
        cache_key: Optional[str] = None,
    ) -> LLMResponse:
        """Single-turn completion. `tag` drives accounting/mocking; `cache_key`
        marks requests sharing an identical prefix (prefix-cache accounting)."""


def _check_effort(effort: Optional[str]) -> Optional[str]:
    if effort is None:
        return None
    if effort not in VALID_EFFORTS:
        raise ValueError(
            f"reasoning_effort={effort!r} invalid: Qwen3.8 accepts only {VALID_EFFORTS} "
            "('high' returns HTTP 500 upstream)"
        )
    return effort


class OpenAICompatClient(LLMClient):
    """vLLM / OpenAI-compatible chat completions via stdlib urllib."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000/v1",
        model: str = "qwen3.8-27b",
        api_key: str = "EMPTY",
        ledger: Optional[TokenLedger] = None,
        retries: int = 2,
        retry_delay: float = 3.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.ledger = ledger
        self.retries = retries
        self.retry_delay = retry_delay

    def _post_sync(self, payload: dict, timeout: float) -> dict:
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    async def complete(
        self,
        prompt: str,
        *,
        tag: str = "generic",
        temperature: float = 1.0,
        max_tokens: int = 2048,
        reasoning_effort: Optional[str] = "low",
        timeout: float = 300.0,
        cache_key: Optional[str] = None,
    ) -> LLMResponse:
        effort = _check_effort(reasoning_effort)
        payload: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if effort is not None:
            payload["chat_template_kwargs"] = {"reasoning_effort": effort}

        last_err: Optional[str] = None
        start = time.monotonic()
        for attempt in range(self.retries + 1):
            try:
                data = await asyncio.to_thread(self._post_sync, payload, timeout)
                text = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
                usage = data.get("usage") or {}
                resp = LLMResponse(
                    text=text.strip(),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    duration=time.monotonic() - start,
                )
                if self.ledger is not None:
                    self.ledger.add(tag, resp.prompt_tokens, resp.completion_tokens, cache_key=cache_key)
                return resp
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, json.JSONDecodeError) as e:
                last_err = f"{type(e).__name__}: {e}"
                if attempt < self.retries:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))
        return LLMResponse(error=last_err or "request failed", duration=time.monotonic() - start)


class LiteLLMClient(LLMClient):
    """Poetiq's original path: LiteLLM acompletion (model e.g. 'openai/qwen3.8-27b'
    with api_base pointed at a local vLLM server, or any provider)."""

    def __init__(self, model: str, ledger: Optional[TokenLedger] = None, **props):
        import litellm  # optional dependency; raises ImportError if absent

        self._litellm = litellm
        litellm.suppress_debug_info = True
        self.model = model
        self.ledger = ledger
        self.props = props

    async def complete(
        self,
        prompt: str,
        *,
        tag: str = "generic",
        temperature: float = 1.0,
        max_tokens: int = 2048,
        reasoning_effort: Optional[str] = "low",
        timeout: float = 300.0,
        cache_key: Optional[str] = None,
    ) -> LLMResponse:
        kwargs = dict(self.props)
        if reasoning_effort is not None:
            _check_effort(reasoning_effort)
            kwargs["chat_template_kwargs"] = {"reasoning_effort": reasoning_effort}
        start = time.monotonic()
        try:
            resp = await self._litellm.acompletion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                num_retries=2,
                **kwargs,
            )
            usage = getattr(resp, "usage", None)
            out = LLMResponse(
                text=(resp["choices"][0]["message"]["content"] or "").strip(),
                prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
                duration=time.monotonic() - start,
            )
            if self.ledger is not None:
                self.ledger.add(tag, out.prompt_tokens, out.completion_tokens, cache_key=cache_key)
            return out
        except Exception as e:  # noqa: BLE001 — client must never kill a task
            return LLMResponse(error=f"{type(e).__name__}: {e}", duration=time.monotonic() - start)


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------

class MockLLM(LLMClient):
    """Deterministic mock for tests/dry runs.

    Two composable modes:
      - scripted: `responses` is a list (consumed cyclically or per-tag dict) of
        str | Exception | callable(prompt, tag) -> str;
      - lookup: built via MockLLM.from_tasks(challenges, solutions) — solver
        prompts are answered with a program that exactly reproduces the known
        outputs (dispatch on exact input match), judge prompts echo the two most
        frequent <PREDICTED_GRID> grids, text prompts echo the test input.
        `fail_every` injects a malformed response every k-th solver call to
        exercise the parsing/refinement paths.
    """

    def __init__(
        self,
        responses: Optional[list | dict] = None,
        lookup: Optional[dict] = None,
        fail_every: int = 0,
        seed: int = 0,
        ledger: Optional[TokenLedger] = None,
        latency: float = 0.0,
    ):
        self.responses = responses
        self.lookup = lookup or {}
        self.fail_every = fail_every
        self.rng = random.Random(seed)
        self.ledger = ledger
        self.latency = latency
        self.calls: list[dict] = []
        self._script_idx = 0
        self._solver_calls = 0

    @classmethod
    def from_tasks(
        cls,
        challenges: dict,
        solutions: Optional[dict] = None,
        fail_every: int = 0,
        seed: int = 0,
        ledger: Optional[TokenLedger] = None,
    ) -> "MockLLM":
        lookup = {}
        for tid, task in challenges.items():
            lookup[tid] = {
                "train_in": [ex["input"] for ex in task["train"]],
                "train_out": [ex["output"] for ex in task["train"]],
                "test_in": [ex["input"] for ex in task["test"]],
                "outputs": (solutions or {}).get(tid),
            }
        return cls(lookup=lookup, fail_every=fail_every, seed=seed, ledger=ledger)

    # -- response construction ---------------------------------------------
    def _solve_code(self, t: dict) -> str:
        pairs = list(zip(t["train_in"], t["train_out"]))
        if t["outputs"]:
            pairs += list(zip(t["test_in"], t["outputs"]))
        table = ",\n    ".join(f"({json.dumps(i)}, {json.dumps(o)})" for i, o in pairs)
        return (
            "import numpy as np\n\n"
            "def transform(grid: np.ndarray) -> np.ndarray:\n"
            '    """Mock lookup solution."""\n'
            "    table = [\n    " + table + ",\n    ]\n"
            "    g = np.asarray(grid).tolist()\n"
            "    for i, o in table:\n"
            "        if g == i:\n"
            "            return np.array(o)\n"
            "    return np.asarray(grid)\n"
        )

    def _identify_task(self, prompt: str) -> Optional[dict]:
        from harness.grids import grid_to_csv, grid_to_diagram

        for tid, t in self.lookup.items():
            for g in t["test_in"]:
                if grid_to_diagram(g) in prompt or grid_to_csv(g) in prompt:
                    return t
        return None

    def _mock_solver_response(self, prompt: str) -> str:
        t = self._identify_task(prompt)
        if t is None or not t["outputs"]:
            return "I could not find a rule.\n```python\nimport numpy as np\ndef transform(grid):\n    return grid\n```"
        code = self._solve_code(t)
        return (
            "The rule maps each known input to its observed output.\n\n"
            f"```python\n{code}\n```"
        )

    def _mock_text_response(self, prompt: str) -> str:
        t = self._identify_task(prompt)
        from harness.grids import grid_to_csv

        if t is None:
            return "0,0\n0,0"
        g = t["test_in"][0]
        return (
            "The pattern looks like it preserves the layout.\n\n"
            "Final answer:\n```\n" + grid_to_csv(g) + "\n```"
        )

    def _mock_judge_response(self, prompt: str) -> str:
        from harness.grids import grid_to_csv

        blocks = re.findall(r"<PREDICTED_GRID>\n(.*?)\n</PREDICTED_GRID>", prompt, re.DOTALL)
        counts: dict[str, int] = {}
        for b in blocks:
            counts[b] = counts.get(b, 0) + 1
        # most frequent first; stable order for ties
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], blocks.index(kv[0])))
        picks = [g for g, _ in ranked[:2]]
        while len(picks) < 2:
            picks.append("0,0")
        return (
            "Meta-conclusion: the majority mechanic is the most plausible.\n\n"
            "First choice:\n```\n" + picks[0] + "\n```\n\n"
            "Second choice:\n```\n" + picks[1] + "\n```"
        )

    async def complete(
        self,
        prompt: str,
        *,
        tag: str = "generic",
        temperature: float = 1.0,
        max_tokens: int = 2048,
        reasoning_effort: Optional[str] = "low",
        timeout: float = 300.0,
        cache_key: Optional[str] = None,
    ) -> LLMResponse:
        if reasoning_effort is not None:
            _check_effort(reasoning_effort)
        if self.latency:
            await asyncio.sleep(self.latency)
        self.calls.append({"tag": tag, "prompt_chars": len(prompt), "cache_key": cache_key})

        # scripted mode takes priority
        if self.responses is not None:
            item: object
            if isinstance(self.responses, dict):
                pool = self.responses.get(tag) or self.responses.get("*")
                item = pool[self._script_idx % len(pool)] if isinstance(pool, list) else pool
            else:
                item = self.responses[self._script_idx % len(self.responses)]
            self._script_idx += 1
            if isinstance(item, Exception):
                return LLMResponse(error=str(item))
            text = item(prompt, tag) if callable(item) else str(item)
        else:
            self._solver_calls += 1
            if (
                self.fail_every
                and tag.startswith("solver")
                and self._solver_calls % self.fail_every == 0
            ):
                text = "Here is my answer but no code block at all, just prose."
            elif tag.startswith("solver"):
                text = self._mock_solver_response(prompt)
            elif tag == "judge":
                text = self._mock_judge_response(prompt)
            elif tag == "text":
                text = self._mock_text_response(prompt)
            else:
                text = "0,0"

        pt, ct = max(1, len(prompt) // 4), max(1, len(text) // 4)
        if self.ledger is not None:
            self.ledger.add(tag, pt, ct, cache_key=cache_key)
        return LLMResponse(text=text.strip(), prompt_tokens=pt, completion_tokens=ct, duration=0.0)
