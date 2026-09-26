"""Land's holistic trace judge: ONE long-context prompt with ALL candidate
traces per test input, a council of N samples of the same model at higher
reasoning effort, each picking its top-2 grids (synthesis allowed), aggregated
2 points per first choice / 1 point per second choice per distinct grid.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from harness.budget import Deadline, TokenLedger
from harness.grids import distinct_grids, extract_all_grids, grid_key, normalize_grid
from harness.llm import LLMClient
from harness.prompts import build_judge_prompt
from harness.types import Candidate, Grid, JudgePick, ScoreEntry


def build_judge_entries(
    candidates: list[Candidate],
    test_idx: int,
    max_entries: int = 12,
) -> list[dict]:
    """Distinct grids for one test input with a representative trace each.

    Ordering: verified candidates first, then by soft score — the cap must drop
    the weakest traces, never the verified ones. Duplicates (same grid) are
    collapsed but their traces' sources noted, so the judge still sees natural
    multiplicity through repeated solution entries only up to the cap.
    """
    usable = [
        c for c in candidates
        if test_idx < len(c.grids) and c.grids[test_idx] is not None
    ]
    usable.sort(key=lambda c: (not c.verified, -c.soft, c.source, c.iteration))

    entries: list[dict] = []
    seen: set[tuple] = set()
    for c in usable:
        g = c.grids[test_idx]
        k = grid_key(g)
        if k in seen:
            continue
        seen.add(k)
        entries.append({"content": c.trace or c.code or "", "grid": g, "source": c.source})
        if len(entries) >= max_entries:
            break
    return entries


def parse_judge_response(text: str) -> list[Grid]:
    """Land's run_duo_pick_judge parsing: all CSV blocks, last two distinct."""
    grids = [g for g in (normalize_grid(b) for b in extract_all_grids(text)) if g]
    unique = distinct_grids(grids)
    return unique[-2:] if len(unique) >= 2 else unique


async def run_council(
    task_id: str,
    train_in: list[Grid],
    train_out: list[Grid],
    test_input: Grid,
    entries: list[dict],
    client: LLMClient,
    n_judges: int = 3,
    temperature: float = 1.0,
    reasoning_effort: str = "medium",
    max_tokens: int = 4096,
    request_timeout: float = 600.0,
    ledger: Optional[TokenLedger] = None,
    deadline: Optional[Deadline] = None,
    trace_cap_chars: int = 2500,
) -> list[JudgePick]:
    """Run n_judges samples over the SAME prompt (prefix cache => one prefill)."""
    if not entries:
        return []
    prompt = build_judge_prompt(
        train_in, train_out, test_input, entries,
        total_attempts=len(entries), trace_cap_chars=trace_cap_chars,
    )
    cache_key = f"judge:{task_id}:{grid_key(test_input)[:4]}"

    async def one(idx: int) -> JudgePick:
        if deadline is not None and deadline.level() >= 4:
            return JudgePick(error="deadline")
        resp = await client.complete(
            prompt,
            tag="judge",
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            timeout=request_timeout,
            cache_key=cache_key,
        )
        if not resp.ok:
            return JudgePick(error=resp.error)
        grids = parse_judge_response(resp.text)
        return JudgePick(grids=grids, prompt_tokens=resp.prompt_tokens,
                          completion_tokens=resp.completion_tokens)

    results = await asyncio.gather(*[one(i) for i in range(n_judges)], return_exceptions=True)
    picks: list[JudgePick] = []
    for r in results:
        if isinstance(r, JudgePick):
            picks.append(r)
        else:
            picks.append(JudgePick(error=str(r)))
    return picks


def aggregate_picks(picks: list[JudgePick]) -> list[ScoreEntry]:
    """2 points for each judge's first choice, 1 point for each judge's second
    choice, summed per distinct output grid (Land's selection_advanced scoreboard).
    Grids not produced by any candidate program are 'synthesized'."""
    scoreboard: dict[tuple, ScoreEntry] = {}
    for run_idx, pick in enumerate(picks):
        for slot, grid in enumerate(pick.grids[:2]):
            points = 2 if slot == 0 else 1
            k = grid_key(grid)
            if k not in scoreboard:
                scoreboard[k] = ScoreEntry(grid=grid, origin="candidate", matched=True)
            scoreboard[k].points += points
            scoreboard[k].voters.append(run_idx)
    ranked = sorted(scoreboard.values(), key=lambda e: (-e.points,))
    return ranked


def mark_synthesized(entries_list: list[ScoreEntry], candidate_keys: set[tuple]) -> None:
    for e in entries_list:
        if grid_key(e.grid) not in candidate_keys:
            e.origin = "synthesized"
            e.matched = False
