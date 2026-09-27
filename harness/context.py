"""Context estimation and eviction.

The Duck enables "infinite play" by continuously popping the oldest messages
while keeping the system prompt and recent history.  We need the same thing, but
our message list has a different shape and one extra invariant:

* indices ``[0]`` (system) and ``[1]`` (the fixed task block, with the single
  image) are **never** dropped -- they are the prefix-cacheable part and they are
  the only place the task data lives;
* an OpenAI-compatible API rejects a ``tool`` message that does not immediately
  follow the ``assistant`` message carrying its ``tool_call_id``, so eviction has
  to remove whole blocks (assistant + its tool results + any status user message),
  never a partial block.

The last resort, before giving up, is to strip the image part out of the task
block: it is the largest single object in the prefix and the model can always
re-derive the same information from ``TRAIN[i].ascii``.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from .grids import sequence_text

SYSTEM_INDEX = 0
TASK_INDEX = 1
PROTECTED_PREFIX = 2


def estimate_tokens(value: Any, *, chars_per_token: float = 3.0) -> int:
    """Chars/N estimator -- the Duck's ``_estimate_tokens``.

    Only used when the server does not report usage.  For ASCII grid art this
    slightly *under*-counts (repetitive symbol strings merge less well than
    prose), which is the unsafe direction, so callers that matter pass the
    server-reported numbers instead.
    """
    try:
        rendered = json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        rendered = str(value)
    return max(1, int((len(rendered) + chars_per_token - 1) // chars_per_token))


def estimate_message_tokens(
    messages: Sequence[dict[str, Any]],
    *,
    tools: Sequence[dict[str, Any]] | None = None,
    image_tokens: int = 0,
    per_message_overhead: int = 4,
) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "image_url":
                        total += image_tokens
                    else:
                        total += estimate_tokens(str(part.get("text", "")))
        else:
            total += estimate_tokens(sequence_text(content))
        for call in message.get("tool_calls") or []:
            total += estimate_tokens(call)
        total += per_message_overhead
    if tools:
        total += estimate_tokens(list(tools))
    return total


def _is_assistant(message: dict[str, Any]) -> bool:
    return str(message.get("role", "")).strip() == "assistant"


def _is_tool(message: dict[str, Any]) -> bool:
    return str(message.get("role", "")).strip() == "tool"


def block_starts(messages: Sequence[dict[str, Any]]) -> list[int]:
    """Indices that begin a droppable block (each assistant turn and its tail)."""
    return [i for i, m in enumerate(messages) if _is_assistant(m) and i >= PROTECTED_PREFIX]


def count_assistant_turns(messages: Sequence[dict[str, Any]]) -> int:
    return sum(1 for m in messages if _is_assistant(m))


def drop_oldest_block(messages: list[dict[str, Any]], *, preserve_recent_turns: int) -> bool:
    """Pop the oldest assistant block. Returns False when nothing is droppable."""
    starts = [i for i in block_starts(messages)]
    if not starts:
        return False
    if count_assistant_turns(messages) <= preserve_recent_turns:
        return False
    first = starts[0]
    second = starts[1] if len(starts) > 1 else len(messages)
    # Never leave the list starting with an orphan `tool` message: the API
    # rejects a tool result whose assistant/tool_call parent has been dropped.
    del messages[first:second]
    while len(messages) > PROTECTED_PREFIX and _is_tool(messages[PROTECTED_PREFIX]):
        del messages[PROTECTED_PREFIX]
    return True


def strip_image(messages: list[dict[str, Any]]) -> bool:
    """Replace the image part of the task block with a one-line note."""
    if len(messages) <= TASK_INDEX:
        return False
    content = messages[TASK_INDEX].get("content")
    if not isinstance(content, list):
        return False
    text_parts = [p for p in content if isinstance(p, dict) and p.get("type") == "text"]
    had_image = any(isinstance(p, dict) and p.get("type") == "image_url" for p in content)
    if not had_image:
        return False
    text = "\n".join(str(p.get("text", "")) for p in text_parts)
    messages[TASK_INDEX] = {
        "role": messages[TASK_INDEX].get("role", "user"),
        "content": text + "\n[image dropped to save context: use TRAIN[i].ascii / .seg()]",
    }
    return True


def trim_to_budget(
    messages: Sequence[dict[str, Any]],
    *,
    context_budget: int,
    tools: Sequence[dict[str, Any]] | None = None,
    image_tokens: int = 0,
    preserve_recent_turns: int = 6,
    allow_image_strip: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return a copy of ``messages`` that fits ``context_budget``.

    Order of operations: drop oldest blocks (keeping the most recent
    ``preserve_recent_turns`` assistant turns), then strip the image, then keep
    dropping.  Never touches the protected prefix.
    """
    out = [dict(m) for m in messages]
    stats = {"blocks_dropped": 0, "image_stripped": 0}
    if not out:
        return out, stats

    def size() -> int:
        return estimate_message_tokens(out, tools=tools, image_tokens=image_tokens)

    current_image_tokens = image_tokens
    while size() > context_budget:
        if drop_oldest_block(out, preserve_recent_turns=preserve_recent_turns):
            stats["blocks_dropped"] += 1
            continue
        if allow_image_strip and current_image_tokens and strip_image(out):
            stats["image_stripped"] += 1
            current_image_tokens = 0
            continue
        break
    stats["tokens"] = size()
    return out, stats
