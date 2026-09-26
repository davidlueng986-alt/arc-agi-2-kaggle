from __future__ import annotations

import pytest

from harness.diagnostics import CoverageStats, CoverageTracker
from harness.tasks import ARCTask

TRAIN = (([[0]], [[1]]),)


def tasks_with_truth():
    return [
        ARCTask("a", TRAIN, ([[2]],), solutions=([[9]],)),
        ARCTask("b", TRAIN, ([[3]], [[4]]), solutions=([[9]], [[8]])),
    ]


def test_tracker_disabled_without_ground_truth():
    tracker = CoverageTracker([ARCTask("x", TRAIN, ([[2]],))])
    assert tracker.enabled is False
    stats = tracker.finalize(0.0)
    assert stats.outputs == 0
    assert stats.verdict() == "no ground truth"


def test_verdict_reports_no_data_when_zero_samples():
    """Regression: a run with no LLM must not claim to have refuted H1."""
    tracker = CoverageTracker(tasks_with_truth())
    stats = tracker.finalize(0.0)
    assert stats.samples == 0
    assert stats.outputs == 3
    assert stats.verdict().startswith("NO DATA")
    assert "REFUTED" not in stats.verdict()


def test_verdict_alive_at_high_coverage():
    tracker = CoverageTracker(tasks_with_truth())
    tracker.observe("a", 0, [[9]], "xhigh")
    tracker.observe("b", 0, [[9]], "xhigh")
    tracker.observe("b", 1, [[8]], "xhigh")
    stats = tracker.finalize(1.0)
    assert stats.coverage == pytest.approx(1.0)
    assert stats.verdict().startswith("H1 ALIVE")
    assert stats.selection_efficiency == pytest.approx(1.0)


def test_verdict_marginal_band():
    tasks = [ARCTask(f"t{i}", TRAIN, ([[i]],), solutions=([[9]],)) for i in range(10)]
    tracker = CoverageTracker(tasks)
    for i in range(4):
        tracker.observe(f"t{i}", 0, [[9]], "xhigh")
    tracker.observe("tx", 0, [[0]], "xhigh")
    stats = tracker.finalize(0.4)
    assert stats.coverage == pytest.approx(0.4)
    assert stats.verdict().startswith("H1 MARGINAL")


def test_verdict_refuted_at_low_coverage():
    tasks = [ARCTask(f"t{i}", TRAIN, ([[i]],), solutions=([[9]],)) for i in range(10)]
    tracker = CoverageTracker(tasks)
    tracker.observe("t0", 0, [[9]], "xhigh")
    for i in range(1, 5):
        tracker.observe(f"t{i}", 0, [[0]], "xhigh")
    stats = tracker.finalize(0.1)
    assert stats.coverage == pytest.approx(0.1)
    assert stats.verdict().startswith("H1 REFUTED")


def test_pass_at_1_counts_every_sample():
    tracker = CoverageTracker(tasks_with_truth())
    assert tracker.observe("a", 0, [[9]], "xhigh") is True
    assert tracker.observe("a", 0, [[0]], "xhigh") is False
    assert tracker.observe("a", 0, [[9]], "xhigh") is True
    assert tracker.observe("b", 1, [[8]], "low") is True
    stats = tracker.finalize(None)
    assert stats.samples == 4
    assert stats.correct_samples == 3
    assert stats.pass_at_1 == pytest.approx(0.75)


def test_by_effort_breakdown():
    tracker = CoverageTracker(tasks_with_truth())
    tracker.observe("a", 0, [[9]], "xhigh")
    tracker.observe("a", 0, [[0]], "xhigh")
    tracker.observe("b", 0, [[9]], "low")
    stats = tracker.finalize(None)
    assert stats.by_effort["xhigh"] == {"samples": 2, "correct": 1}
    assert stats.by_effort["low"] == {"samples": 1, "correct": 1}


def test_coverage_counts_each_output_once_despite_repeats():
    tracker = CoverageTracker(tasks_with_truth())
    for _ in range(5):
        tracker.observe("b", 0, [[9]], "xhigh")
    stats = tracker.finalize(None)
    assert stats.outputs_covered == 1
    assert stats.coverage == pytest.approx(1 / 3)
    assert stats.tasks_fully_covered == 0


def test_tasks_fully_covered_requires_every_index():
    tracker = CoverageTracker(tasks_with_truth())
    tracker.observe("a", 0, [[9]], "xhigh")
    tracker.observe("b", 0, [[9]], "xhigh")
    tracker.observe("b", 1, [[8]], "xhigh")
    stats = tracker.finalize(None)
    assert stats.tasks_fully_covered == 2


def test_uncovered_outputs_and_unknown_task():
    tracker = CoverageTracker(tasks_with_truth())
    assert tracker.observe("nonexistent", 0, [[9]], "xhigh") is False
    assert tracker.observe("b", 99, [[9]], "xhigh") is False
    stats = tracker.finalize(None)
    assert stats.uncovered_outputs == 3


def test_selection_efficiency_zero_coverage_is_safe():
    stats = CoverageStats(outputs=5, samples=3, correct_samples=0)
    assert stats.coverage == 0.0
    assert stats.selection_efficiency == 0.0


def test_as_dict_is_json_safe():
    import json

    tracker = CoverageTracker(tasks_with_truth())
    tracker.observe("a", 0, [[9]], "xhigh")
    blob = tracker.finalize(0.5).as_dict()
    assert json.loads(json.dumps(blob))["pass_at_1"] == 1.0
    assert "verdict" in blob
