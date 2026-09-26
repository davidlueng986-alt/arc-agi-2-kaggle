import asyncio
import json

import pytest

from harness.llm import MockLLM, OpenAICompatClient, _check_effort
from harness.oracle import evaluate_code


class TestEffortValidation:
    def test_high_rejected(self):
        # QwenLM/Qwen3.8#217: 'high' returns HTTP 500 upstream — fail fast locally
        with pytest.raises(ValueError):
            _check_effort("high")

    def test_valid_efforts(self):
        for e in ("xhigh", "medium", "low"):
            assert _check_effort(e) == e
        assert _check_effort(None) is None


class TestScriptedMock:
    def test_cycles_responses(self):
        mock = MockLLM(responses=["one", "two"])
        r1 = asyncio.run(mock.complete("p", tag="x"))
        r2 = asyncio.run(mock.complete("p", tag="x"))
        r3 = asyncio.run(mock.complete("p", tag="x"))
        assert (r1.text, r2.text, r3.text) == ("one", "two", "one")

    def test_per_tag_dict(self):
        mock = MockLLM(responses={"judge": ["J"], "*": ["OTHER"]})
        assert asyncio.run(mock.complete("p", tag="judge")).text == "J"
        assert asyncio.run(mock.complete("p", tag="solver_a")).text == "OTHER"

    def test_exception_item_becomes_error_response(self):
        mock = MockLLM(responses=[RuntimeError("boom")])
        r = asyncio.run(mock.complete("p"))
        assert not r.ok
        assert "boom" in r.error

    def test_callable_item(self):
        mock = MockLLM(responses=[lambda prompt, tag: f"echo:{tag}:{len(prompt)}"])
        r = asyncio.run(mock.complete("abcd", tag="t1"))
        assert r.text == "echo:t1:4"

    def test_records_calls(self):
        mock = MockLLM(responses=["x"])
        asyncio.run(mock.complete("p", tag="solver_a", cache_key="ck"))
        assert mock.calls[0]["tag"] == "solver_a"
        assert mock.calls[0]["cache_key"] == "ck"


class TestLookupMock:
    def test_solver_response_verifies_on_real_tasks(self, eval_challenges, eval_solutions):
        ids = sorted(k for k in eval_challenges if k in eval_solutions)[:3]
        subset = {k: eval_challenges[k] for k in ids}
        sols = {k: eval_solutions[k] for k in ids}
        mock = MockLLM.from_tasks(subset, sols)
        for tid in ids:
            task = subset[tid]
            from harness.prompts import build_solver_a_prompt

            prompt = build_solver_a_prompt(
                [ex["input"] for ex in task["train"]],
                [ex["output"] for ex in task["train"]],
                [ex["input"] for ex in task["test"]],
            )
            resp = asyncio.run(mock.complete(prompt, tag="solver_expert_a"))
            from harness.parsing import pick_transform_code

            picked = pick_transform_code(resp.text)
            assert picked is not None, tid
            code, entry = picked
            train_res, test_res = asyncio.run(
                evaluate_code(
                    code,
                    [ex["input"] for ex in task["train"]],
                    [ex["output"] for ex in task["train"]],
                    [ex["input"] for ex in task["test"]],
                    timeout_s=60,
                    entry=entry,
                )
            )
            assert all(r.success for r in train_res), tid
            for j, r in enumerate(test_res):
                assert r.grid == sols[tid][j], (tid, j)

    def test_fail_every_injects_malformed(self, eval_challenges, eval_solutions):
        tid = sorted(eval_challenges)[0]
        subset = {tid: eval_challenges[tid]}
        mock = MockLLM.from_tasks(subset, eval_solutions, fail_every=1)
        r = asyncio.run(mock.complete("anything", tag="solver_expert_a"))
        assert "```" not in r.text  # malformed: no code block

    def test_judge_response_has_two_csv_blocks(self):
        mock = MockLLM(responses=None)
        prompt = (
            "<PREDICTED_GRID>\n1,1\n</PREDICTED_GRID>\n"
            "<PREDICTED_GRID>\n2,2\n</PREDICTED_GRID>\n"
            "<PREDICTED_GRID>\n1,1\n</PREDICTED_GRID>\n"
        )
        r = asyncio.run(mock.complete(prompt, tag="judge"))
        assert r.text.count("```") == 4
        assert "1,1" in r.text and "2,2" in r.text

    def test_text_response_parses(self):
        from harness.grids import parse_grid_from_text

        tid_tasks = {
            "t1": {
                "train_in": [[[0]]], "train_out": [[[1]]],
                "test_in": [[[7, 8]]], "outputs": [[[9, 9]]],
            }
        }
        mock = MockLLM(lookup=tid_tasks)
        r = asyncio.run(mock.complete("7,8 test input", tag="text"))
        grid = parse_grid_from_text(r.text)
        assert grid == [[7, 8]]  # mock echoes the test input (plausible wrong answer)


class TestOpenAICompatClient:
    def test_effort_high_rejected_before_network(self):
        client = OpenAICompatClient(base_url="http://127.0.0.1:1/v1")
        with pytest.raises(ValueError):
            asyncio.run(client.complete("p", reasoning_effort="high"))

    def test_connection_error_becomes_error_response(self):
        client = OpenAICompatClient(base_url="http://127.0.0.1:1/v1", retries=0)
        r = asyncio.run(client.complete("p", reasoning_effort="low", timeout=2))
        assert not r.ok
        assert r.error
