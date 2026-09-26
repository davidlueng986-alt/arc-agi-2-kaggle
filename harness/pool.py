"""Solution pool: the diversity-preserving memory of a refinement expert.

Poetiq semantics: every scored partial solution is kept; the feedback context
shows the top-K by soft score (K = max_solutions = 5), worst-first when
improving_order is set, so the next sample is conditioned on a *distribution*
of hypotheses rather than a single chain (see HYPOTHESIS.md §2).
"""

from __future__ import annotations

import hashlib
import re

from harness.prompts import build_feedback_block


def _code_fingerprint(code: str) -> str:
    norm = re.sub(r"\s+", " ", code.strip())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


class SolutionPool:
    def __init__(self, max_solutions: int = 5, improving_order: bool = True):
        self.max_solutions = max_solutions
        self.improving_order = improving_order
        self._entries: list[dict] = []
        self._fingerprints: set[str] = set()

    def __len__(self) -> int:
        return len(self._entries)

    def add(self, code: str, feedback: str, score: float) -> bool:
        """Add a scored partial solution. Returns False if an identical program
        (whitespace-normalised) is already pooled — duplicate blocks would waste
        feedback context without adding diversity."""
        fp = _code_fingerprint(code)
        if fp in self._fingerprints:
            return False
        self._fingerprints.add(fp)
        self._entries.append({"code": code, "feedback": feedback, "score": float(score)})
        return True

    @property
    def entries(self) -> list[dict]:
        return list(self._entries)

    def best(self) -> dict | None:
        if not self._entries:
            return None
        return max(self._entries, key=lambda e: e["score"])

    def feedback_block(self) -> str:
        """Top-K scored entries rendered as the Poetiq <solution_i> block."""
        return build_feedback_block(
            self._entries, max_examples=self.max_solutions, improving_order=self.improving_order
        )
