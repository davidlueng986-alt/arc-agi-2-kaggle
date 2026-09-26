"""Tolerant extraction of code blocks (and transform entry points) from model output.

Local 27B models produce malformed output far more often than frontier APIs:
unfenced code, ```py fences, multiple blocks, prose after code, `def solver`
instead of `def transform`, and refusals. Everything here is best-effort and
returns None instead of raising.
"""

from __future__ import annotations

import re
from typing import Optional

FENCED_RE = re.compile(r"```([a-zA-Z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
REFUSAL_PATTERNS = (
    r"\bi (?:cannot|can't|am unable to|'m unable to)\b",
    r"\bi'?m sorry\b",
    r"\bas an ai\b",
    r"\bi (?:do not|don't) have (?:the )?(?:ability|access)\b",
    r"\bcannot (?:assist|help|comply)\b",
)


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """All fenced blocks as (lang, body), in order. Tolerates ``` without lang."""
    if not text:
        return []
    return [(m.group(1).strip().lower(), m.group(2)) for m in FENCED_RE.finditer(text)]


def _has_entry(body: str) -> bool:
    return re.search(r"^\s*def\s+(transform|solver)\s*\(", body, re.MULTILINE) is not None


def _extract_unfenced(text: str) -> Optional[str]:
    """Last resort: code without any fence. Capture from the first import/def line."""
    m = re.search(r"^(?:import\s+\w+|from\s+\w+|def\s+(?:transform|solver)\s*\()", text, re.MULTILINE)
    if not m:
        return None
    body = text[m.start():]
    lines: list[str] = []
    for line in body.splitlines():
        s = line.strip()
        if lines and s and not line[0].isspace() and not s.startswith(("def ", "import ", "from ", "@", "#")):
            # Non-indented prose after code started -> stop.
            if not s.endswith((":","\\", ")", "]")) and "return" not in s:
                break
        lines.append(line)
    code = "\n".join(lines).strip()
    return code if _has_entry(code) else None


def pick_transform_code(text: str) -> Optional[tuple[str, str]]:
    """Find the best (code, entry_name) pair in a model response.

    Preference order:
      1. python/py fenced block defining transform (last one wins — models often
         draft, then emit a final version);
      2. any fenced block defining transform or solver;
      3. unfenced `def transform` / `def solver` region.
    `solver` is accepted and bound to `transform` by the sandbox shim.
    """
    if not text:
        return None
    blocks = extract_code_blocks(text)
    py_blocks = [b for lang, b in blocks if lang in ("python", "py", "python3")]

    for pool in (py_blocks, [b for _, b in blocks]):
        with_transform = [b for b in pool if re.search(r"^\s*def\s+transform\s*\(", b, re.MULTILINE)]
        if with_transform:
            return with_transform[-1].strip(), "transform"
        with_solver = [b for b in pool if re.search(r"^\s*def\s+solver\s*\(", b, re.MULTILINE)]
        if with_solver:
            return with_solver[-1].strip(), "solver"

    unfenced = _extract_unfenced(text)
    if unfenced:
        entry = "transform" if re.search(r"^\s*def\s+transform\s*\(", unfenced, re.MULTILINE) else "solver"
        return unfenced, entry
    return None


def looks_like_refusal(text: str) -> bool:
    if not text or len(text.strip()) < 3:
        return True  # empty/near-empty treated as refusal
    lowered = text.lower()
    return any(re.search(p, lowered) for p in REFUSAL_PATTERNS)


def truncate(text: str, max_chars: int, marker: str = "\n...[truncated]\n") -> str:
    if not text or len(text) <= max_chars:
        return text or ""
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head] + marker + text[-tail:]
