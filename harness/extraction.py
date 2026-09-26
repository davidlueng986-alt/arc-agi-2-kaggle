"""Tolerant extraction of a `transform` program from model output.

The model is asked for a single markdown-fenced python block, but thinking-mode
models routinely emit extra prose, several blocks, or forget fences entirely
(vLLM structured output is unreliable in thinking mode, vLLM#35700), so we
parse tolerantly and prefer the LAST block that defines `transform`.
"""
from __future__ import annotations

import re
from typing import Optional

_FENCE_RE = re.compile(r"^[\s>]*(```+|~~~+)[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?$")


def extract_code_blocks(text: str) -> list[str]:
    """Return all fenced code block bodies, in order. Handles unclosed fences."""
    blocks: list[str] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _FENCE_RE.match(lines[i])
        if m:
            fence = m.group(1)[0] * 3  # normalize ```` -> ```
            body: list[str] = []
            i += 1
            closed = False
            while i < len(lines):
                m2 = _FENCE_RE.match(lines[i])
                if m2 and m2.group(1)[0] * 3 == fence:
                    closed = True
                    i += 1
                    break
                body.append(lines[i])
                i += 1
            blocks.append("\n".join(body))
            if not closed:
                break
        else:
            i += 1
    return blocks


def pick_program(text: str) -> Optional[str]:
    """Pick the best candidate program source from a model response.

    Preference order:
    1. last fenced block containing `def transform`
    2. last fenced block containing any `def `
    3. raw text, when it itself contains `def transform`
    """
    if not text:
        return None
    blocks = extract_code_blocks(text)
    with_transform = [b for b in blocks if re.search(r"^\s*def\s+transform\b", b, re.M)]
    if with_transform:
        return with_transform[-1].strip() + "\n"
    with_def = [b for b in blocks if re.search(r"^\s*def\s+\w+", b, re.M)]
    if with_def:
        return with_def[-1].strip() + "\n"
    if re.search(r"^\s*def\s+transform\b", text, re.M):
        return text.strip() + "\n"
    return None


def compiles(code: str) -> bool:
    try:
        compile(code, "<candidate>", "exec")
        return True
    except SyntaxError:
        return False
