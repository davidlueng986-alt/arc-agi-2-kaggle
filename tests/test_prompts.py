import numpy as np

from harness.grid import to_grid
from harness.prompts import (
    build_adjudication_messages,
    build_loo_messages,
    build_solver_messages,
    build_verified_nudge_messages,
    render_task,
)
from harness.types import Task


def _task():
    x1 = to_grid([[1, 2], [3, 4]])
    y1 = to_grid([[0, 0], [0, 0]])
    x2 = to_grid([[5, 5], [5, 5]])
    y2 = to_grid([[1, 1], [1, 1]])
    return Task("t", [(x1, y1), (x2, y2)], [to_grid([[9, 9], [9, 9]])])


def test_render_task_csv_and_analysis():
    text = render_task(_task(), shuffle=False)
    assert "1,2" in text and "3,4" in text
    assert "Test input #1" in text
    assert "Structural analysis" in text


def test_render_task_omit_pair():
    text = render_task(_task(), shuffle=False, omit_pair=0)
    assert "1,2" not in text.split("Structural")[0]
    assert "5,5" in text


def test_render_task_shuffle_changes_order_sometimes():
    t = _task()
    a = render_task(t, shuffle=True, seed=1)
    b = render_task(t, shuffle=True, seed=2)
    # both contain both pairs, order may differ; assert determinism per seed
    assert a == render_task(t, shuffle=True, seed=1)
    assert b == render_task(t, shuffle=True, seed=2)


def test_solver_messages_include_feedback_block():
    msgs = build_solver_messages(
        _task(),
        iteration=2,
        variant="A",
        partials=[("def transform(g):\n    return g\n", "Example #1: WRONG.", 0.5)],
    )
    assert len(msgs) == 1 and msgs[0]["role"] == "user"
    assert "PREVIOUS ATTEMPTS" in msgs[0]["content"]
    assert "def transform" in msgs[0]["content"]


def test_solver_prompt_variants():
    a = build_solver_messages(_task(), 0, variant="A")[0]["content"]
    b = build_solver_messages(_task(), 0, variant="B")[0]["content"]
    assert len(a) > len(b)  # variant B is the minimal-diversity prompt
    assert "GENERALIZATION CONTRACT" in a


def test_verified_nudge_mentions_both_options():
    msgs = build_verified_nudge_messages(_task(), "def transform(g):\n    return g\n", 1)
    c = msgs[0]["content"]
    assert "PASSES ALL TRAINING PAIRS" in c
    assert "DIFFERENT hypothesis" in c


def test_adjudication_message_contains_programs_and_outputs():
    out = to_grid([[7, 7], [7, 7]])
    msgs = build_adjudication_messages(
        _task(), [("def transform(g):\n    return g\n", out), ("def transform(g):\n    return g*0\n", None)]
    )
    c = msgs[0]["content"]
    assert "program_1" in c and "program_2" in c
    assert "crashed" in c


def test_loo_message_omits_pair_and_test():
    msgs = build_loo_messages(_task(), omit_pair=1)
    c = msgs[0]["content"]
    assert "Test input" not in c
    assert "withheld" in c or "SOME training pairs" in c
