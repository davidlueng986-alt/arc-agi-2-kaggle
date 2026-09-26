import asyncio

from harness.judge import (
    aggregate_picks,
    build_judge_entries,
    mark_synthesized,
    parse_judge_response,
    run_council,
)
from harness.llm import MockLLM
from harness.prompts import build_judge_prompt
from harness.types import Candidate, JudgePick

GA = [[1, 1], [1, 1]]
GB = [[2, 2], [2, 2]]
GC = [[3, 3], [3, 3]]


def _cand(source, grid, verified=False, soft=0.0, trace="trace", it=1):
    return Candidate(source=source, iteration=it, trace=trace, grids=[grid],
                     verified=verified, soft=soft)


class TestJudgeEntries:
    def test_verified_first_and_dedup(self):
        cands = [
            _cand("expert_a", GA, verified=False, soft=0.5),
            _cand("expert_b", GA, verified=True, soft=1.0),   # duplicate grid
            _cand("text[0]", GB),
        ]
        entries = build_judge_entries(cands, 0)
        assert len(entries) == 2  # GA deduped
        assert entries[0]["grid"] == GA  # verified version leads
        assert entries[0]["source"] == "expert_b"

    def test_cap(self):
        cands = [_cand("expert_a", [[i]], it=i) for i in range(20)]
        entries = build_judge_entries(cands, 0, max_entries=12)
        assert len(entries) == 12

    def test_missing_grid_skipped(self):
        c = _cand("expert_a", None)
        c.grids = [None]
        assert build_judge_entries([c], 0) == []


class TestJudgeParsing:
    def test_last_two_distinct(self):
        text = (
            "Reasoning...\n```\n1,1\n1,1\n```\nThen I reconsider.\n"
            "```\n2,2\n2,2\n```\nFinal:\n```\n3,3\n3,3\n```"
        )
        picks = parse_judge_response(text)
        assert picks == [GB, GC]

    def test_duplicate_final_grids_collapse(self):
        text = "```\n2,2\n2,2\n```\n```\n2,2\n2,2\n```"
        assert parse_judge_response(text) == [GB]

    def test_single_grid(self):
        assert parse_judge_response("```\n3,3\n3,3\n```") == [GC]
        # a one-row grid is parsed as-is (1x2)
        assert parse_judge_response("```\n3,3\n```") == [[[3, 3]]]

    def test_no_grid(self):
        assert parse_judge_response("I refuse to answer.") == []


class TestAggregation:
    def test_two_one_point_arithmetic(self):
        # judge0: A then B; judge1: A then C; judge2: B then A
        picks = [
            JudgePick(grids=[GA, GB]),
            JudgePick(grids=[GA, GC]),
            JudgePick(grids=[GB, GA]),
        ]
        board = aggregate_picks(picks)
        pts = {tuple(map(tuple, e.grid)): e.points for e in board}
        assert pts[((1, 1), (1, 1))] == 2 + 2 + 1  # A: two firsts + one second
        assert pts[((2, 2), (2, 2))] == 1 + 2      # B: one second + one first
        assert pts[((3, 3), (3, 3))] == 1          # C: one second
        assert board[0].grid == GA

    def test_failed_judges_ignored(self):
        picks = [JudgePick(grids=[GA, GB]), JudgePick(error="timeout"), JudgePick(grids=[])]
        board = aggregate_picks(picks)
        assert board[0].points == 2 and board[0].grid == GA
        assert board[1].points == 1

    def test_voters_recorded(self):
        picks = [JudgePick(grids=[GA, GB]), JudgePick(grids=[GA, GC])]
        board = aggregate_picks(picks)
        ga = next(e for e in board if e.grid == GA)
        assert ga.voters == [0, 1]

    def test_synthesized_marking(self):
        picks = [JudgePick(grids=[GC, GA])]
        board = aggregate_picks(picks)
        mark_synthesized(board, {tuple(map(tuple, GA))})
        by_grid = {tuple(map(tuple, e.grid)): e for e in board}
        assert by_grid[((3, 3), (3, 3))].origin == "synthesized"
        assert not by_grid[((3, 3), (3, 3))].matched
        assert by_grid[((1, 1), (1, 1))].origin == "candidate"


class TestJudgePrompt:
    def test_prompt_contains_task_traces_and_instructions(self):
        entries = [
            {"content": "reasoning A", "grid": GA},
            {"content": "def transform(g): return g", "grid": GB},
        ]
        prompt = build_judge_prompt(
            [[[0, 0], [0, 0]]], [[[1, 1], [1, 1]]], [[5, 5], [5, 5]],
            entries, total_attempts=7,
        )
        assert "run 7 times" in prompt
        assert "<PROMPT START>" in prompt and "<PROMPT STOP>" in prompt
        assert "<SOLUTION 1 START>" in prompt and "<SOLUTION 2 STOP>" in prompt
        assert "reasoning A" in prompt
        assert "1,1\n1,1" in prompt  # predicted grid in CSV (Land's winning format)
        assert "meta-conclusion" in prompt  # anti-consistency closing preserved
        assert "synthesize" in prompt.lower()

    def test_trace_truncation(self):
        entries = [{"content": "x" * 10_000, "grid": GA}]
        prompt = build_judge_prompt([[[0]]], [[[1]]], [[2]], entries, 1, trace_cap_chars=500)
        assert "[truncated]" in prompt
        assert len(prompt) < 9000


class TestCouncil:
    def test_run_council_with_scripted_mock(self):
        mock = MockLLM(responses={
            "judge": ["```\n1,1\n1,1\n```\n```\n2,2\n2,2\n```"],
        })
        entries = [{"content": "t", "grid": GA}, {"content": "t2", "grid": GB}]
        picks = asyncio.run(run_council(
            "taskid", [[[0]]], [[[1]]], [[7]], entries, mock, n_judges=3,
        ))
        assert len(picks) == 3
        assert all(p.grids == [GA, GB] for p in picks)
        board = aggregate_picks(picks)
        assert board[0].points == 6  # unanimous first choice

    def test_council_survives_errors(self):
        mock = MockLLM(responses={"judge": [RuntimeError("boom"), "```\n1,1\n1,1\n```"]})
        entries = [{"content": "t", "grid": GA}]
        picks = asyncio.run(run_council("t", [[[0]]], [[[1]]], [[7]], entries, mock, n_judges=2))
        assert picks[0].error is not None
        assert picks[1].grids == [GA]
