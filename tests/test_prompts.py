from __future__ import annotations

import pytest

from harness.prompts import SYSTEM_PROMPT, build_messages, build_prompt, prompt_token_estimate
from harness.tasks import ARCTask


def make_task(task_id="p1", n_train=3, n_test=1):
    train = tuple(([[i % 10, 1], [2, 3]], [[3, 2], [1, i % 10]]) for i in range(n_train))
    test = tuple([[4, 4], [4, 4]] for _ in range(n_test))
    return ARCTask(task_id, train, test)


def test_prompt_is_csv_encoded():
    prompt = build_prompt(make_task())
    assert "0,1" in prompt.user
    assert "3,2" in prompt.user
    assert "[[" not in prompt.user


def test_system_prompt_states_format_and_rules():
    assert "ANSWER 1:" in SYSTEM_PROMPT
    assert "comma-separated" in SYSTEM_PROMPT
    assert "0-9" in SYSTEM_PROMPT


def test_prompt_contains_every_train_pair_and_test_input():
    task = make_task(n_train=4, n_test=1)
    prompt = build_prompt(task)
    for i in range(1, 5):
        assert f"Training pair {i} input:" in prompt.user
        assert f"Training pair {i} output:" in prompt.user
    assert "Test input 1:" in prompt.user


def test_multi_test_prompt_lists_all_answers():
    prompt = build_prompt(make_task(n_test=3))
    assert "Test input 1:" in prompt.user
    assert "Test input 3:" in prompt.user
    assert "ANSWER 2:" in prompt.user
    assert "ANSWER 3:" in prompt.user


def test_single_test_prompt_does_not_mention_answer_two():
    prompt = build_prompt(make_task(n_test=1))
    assert "ANSWER 2:" not in prompt.user


def test_messages_shape():
    messages = build_messages(make_task())
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"]


def test_max_train_pairs_truncates():
    prompt = build_prompt(make_task(n_train=6), max_train_pairs=2)
    assert "Training pair 2 input:" in prompt.user
    assert "Training pair 3 input:" not in prompt.user


def test_shuffle_is_deterministic_per_seed_and_affects_order():
    task = make_task(n_train=5)
    a = build_prompt(task, shuffle_train=True, seed=1).user
    b = build_prompt(task, shuffle_train=True, seed=1).user
    c = build_prompt(task, shuffle_train=True, seed=2).user
    assert a == b
    assert a != c


def test_no_shuffle_is_canonical_order():
    task = make_task(n_train=3)
    assert build_prompt(task).user == build_prompt(task, seed=99).user


def test_approx_tokens_positive_and_monotone_in_size():
    small = build_prompt(make_task(n_train=1))
    big = build_prompt(make_task(n_train=6))
    assert small.approx_tokens > 0
    assert big.approx_tokens > small.approx_tokens


def test_prompt_token_estimate_summary():
    tasks = [make_task(task_id=f"t{i}", n_train=i + 1) for i in range(5)]
    stats = prompt_token_estimate(tasks)
    assert stats["n"] == 5
    assert stats["mean"] > 0
    assert stats["max"] >= stats["p50"] >= stats["mean"] * 0.5
    assert stats["total"] == sum(t.approx_prompt_tokens() for t in tasks)


def test_prompt_token_estimate_empty():
    assert prompt_token_estimate([])["n"] == 0


def test_real_eval_prompts_fit_context(data_dir, eval_tasks):
    stats = prompt_token_estimate(eval_tasks[:20])
    assert stats["max"] < 12000, "eval prompts must fit inside max-model-len with a 16K completion"


@pytest.mark.parametrize("n_test", [1, 2, 3])
def test_footer_matches_test_count(n_test):
    prompt = build_prompt(make_task(n_test=n_test))
    assert f"each of the {n_test} test" in prompt.user
