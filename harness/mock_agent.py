"""A deterministic stand-in agent, for offline plumbing validation.

``MockLLM`` supplies canned *responses*; this module supplies a canned
*behaviour* -- a responder that plays a plausible five-turn REPL session against
any ARC-AGI-2 task.  It exists so the entire non-LLM stack (sandbox, oracle,
pool, eviction, budget, watchdog, submission writer, scorer) can be exercised
end-to-end on a laptop with no GPU and no torch, which is the local situation.

It is deliberately dumb: it tries identity, a constant fill, and a
crop-to-bounding-box rule, then finishes.  On the real eval split that scores
near zero.  That is the point -- a ``--mock`` run validates plumbing and schema,
never accuracy, and nobody should mistake its score for a prediction.
"""

from __future__ import annotations

from typing import Any, Callable

from .context import estimate_message_tokens
from .llm import LLMRequest, LLMResponse, MockLLM, ToolCall, Usage
from .render import estimate_vision_tokens

INSPECT_CODE = """
print("shapes", shapes())
print("colors in", TRAIN[0].input.colors(), "out", TRAIN[0].output.colors())
print(TRAIN[0].input.seg(8))
print(TRAIN[0].output.seg(8))
result = {"n_train": len(TRAIN), "n_test": len(TEST)}
"""

CANDIDATES: tuple[tuple[str, str], ...] = (
    (
        "identity",
        """
def transform(g):
    return [list(row) for row in g]

rep = verify(transform)
print(rep["summary"])
for f in rep["failures"][:2]:
    print({k: v for k, v in f.items() if k in ("pair","kind","expected_shape","got_shape","n_diff","diff_frac","bbox","samples")})
""",
    ),
    (
        "bbox_crop",
        """
def transform(g):
    rows, cols = len(g), len(g[0])
    bg = max(set(v for row in g for v in row), key=lambda v: sum(r.count(v) for r in g))
    pts = [(r, c) for r in range(rows) for c in range(cols) if g[r][c] != bg]
    if not pts:
        return [list(row) for row in g]
    r0 = min(p[0] for p in pts); r1 = max(p[0] for p in pts)
    c0 = min(p[1] for p in pts); c1 = max(p[1] for p in pts)
    return [list(row[c0:c1 + 1]) for row in g[r0:r1 + 1]]

rep = verify(transform)
print(rep["summary"])
for f in rep["failures"][:2]:
    print({k: v for k, v in f.items() if k in ("pair","kind","expected_shape","got_shape","n_diff","diff_frac","bbox")})
""",
    ),
    (
        "constant_fill",
        """
def transform(g):
    counts = {}
    for row in g:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    colour = max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return [[colour] * len(g[0]) for _ in g]

rep = verify(transform)
print(rep["summary"])
""",
    ),
)

FINISH_CODE = """
finish("mock agent exhausted its candidate list")
"""


def mock_responder(max_candidate_turns: int | None = None) -> Callable[[LLMRequest], LLMResponse]:
    """Responder: inspect, then each candidate, then finish.

    Stateless -- the turn index is derived from the request's own message list,
    not from a closure counter.  One ``MockLLM`` instance is shared by every
    worker thread (exactly as one vLLM server would be), so any per-instance
    state would leak between tasks and make the mock non-deterministic under
    concurrency.

    Reported usage comes from the same estimator the harness uses for eviction,
    with the image charged at its real vision-token cost rather than at the
    length of its base64 text.  That makes ``--mock --project`` produce a
    per-turn prompt-token figure comparable to a live run.
    """
    limit = len(CANDIDATES) if max_candidate_turns is None else max(0, max_candidate_turns)

    def respond(request: LLMRequest) -> LLMResponse:
        index = sum(1 for m in request.messages if m.get("role") == "assistant")
        if index == 0:
            code = INSPECT_CODE
            text = "Hypothesis: need the shapes and palettes first.\nNext: inspect."
        elif index <= limit:
            name, code = CANDIDATES[index - 1]
            text = f"Hypothesis: {name}.\nNext: verify it."
        else:
            code = FINISH_CODE
            text = "Next: finish with the best registered candidate."
        image_tokens = _image_tokens_in(request.messages)
        prompt_tokens = estimate_message_tokens(
            request.messages, tools=request.tools, image_tokens=image_tokens
        )
        return LLMResponse(
            text=text,
            reasoning="",
            tool_calls=[ToolCall(id=f"call_{index}", name="python", arguments={"code": code})],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=max(1, (len(text) + len(code)) // 3),
            ),
            finish_reason="tool_calls",
            raw={"mock": True, "index": index},
        )

    return respond


def _image_tokens_in(messages) -> int:
    """Vision-token cost of an attached panel, recovered from its pixel size."""
    import base64
    import io

    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = str((part.get("image_url") or {}).get("url", ""))
                if "," not in url:
                    continue
                try:
                    from PIL import Image

                    blob = base64.b64decode(url.split(",", 1)[1])
                    with Image.open(io.BytesIO(blob)) as image:
                        width, height = image.size
                    return estimate_vision_tokens(width, height)
                except Exception:
                    return 0
    return 0


def build_mock_llm(**kwargs: Any) -> MockLLM:
    return MockLLM(responder=mock_responder(**kwargs))
