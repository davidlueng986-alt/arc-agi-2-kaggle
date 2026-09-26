from __future__ import annotations

from harness.scoring import score_predictions, score_submission
from harness.submission import empty_submission
from harness.tasks import ARCTask


TRAIN = (([[0]], [[1]]),)


def make_tasks():
    return [
        ARCTask("t1", TRAIN, ([[2]],), solutions=([[3]],)),
        ARCTask("t2", TRAIN, ([[4]], [[5]]), solutions=([[6]], [[7]])),
        ARCTask("t3", TRAIN, ([[8]],), solutions=None),
    ]


def test_score_predictions_exact_match_only():
    preds = [{"attempt_1": [[3]], "attempt_2": [[0]]}]
    correct, a1, a2 = score_predictions(preds, [[[3]]])
    assert (correct, a1, a2) == (1, 1, 0)


def test_score_predictions_second_attempt_counts_once():
    preds = [{"attempt_1": [[0]], "attempt_2": [[3]]}]
    correct, a1, a2 = score_predictions(preds, [[[3]]])
    assert (correct, a1, a2) == (1, 0, 1)


def test_score_predictions_shape_mismatch_is_wrong():
    preds = [{"attempt_1": [[3, 3]], "attempt_2": [[3, 3]]}]
    correct, _, _ = score_predictions(preds, [[[3]]])
    assert correct == 0


def test_score_predictions_both_attempts_same_correct_grid_counts_once():
    preds = [{"attempt_1": [[3]], "attempt_2": [[3]]}]
    correct, a1, a2 = score_predictions(preds, [[[3]]])
    assert (correct, a1, a2) == (1, 1, 0)


def test_score_submission_averages_over_outputs_not_tasks():
    tasks = make_tasks()
    sub = empty_submission(tasks)
    sub["t1"] = [{"attempt_1": [[3]], "attempt_2": [[3]]}]
    sub["t2"] = [{"attempt_1": [[6]], "attempt_2": [[0]]}, {"attempt_1": [[0]], "attempt_2": [[0]]}]
    report = score_submission(sub, tasks)
    assert report.total_outputs == 3
    assert report.total_correct == 2
    assert abs(report.score - 2 / 3) < 1e-9
    assert report.unscored == 1


def test_attempt_one_score_and_second_attempt_gain():
    tasks = make_tasks()
    sub = empty_submission(tasks)
    sub["t1"] = [{"attempt_1": [[3]], "attempt_2": [[3]]}]
    sub["t2"] = [{"attempt_1": [[0]], "attempt_2": [[6]]}, {"attempt_1": [[0]], "attempt_2": [[0]]}]
    report = score_submission(sub, tasks)
    assert report.attempt_1_correct == 1
    assert report.attempt_2_only_correct == 1
    assert abs(report.attempt_1_score - 1 / 3) < 1e-9
    assert abs(report.second_attempt_gain - 1 / 3) < 1e-9


def test_solved_and_failed_lists():
    tasks = make_tasks()
    sub = empty_submission(tasks)
    sub["t1"] = [{"attempt_1": [[3]], "attempt_2": [[3]]}]
    report = score_submission(sub, tasks)
    assert report.solved_task_ids() == ["t1"]
    assert report.failed_task_ids() == ["t2"]


def test_report_as_dict_roundtrip():
    tasks = make_tasks()
    report = score_submission(empty_submission(tasks), tasks)
    blob = report.as_dict()
    assert blob["score"] == 0.0
    assert blob["total_outputs"] == 3
    assert blob["unscored_tasks"] == 1


def test_missing_predictions_score_zero():
    tasks = [ARCTask("x", TRAIN, ([[2]],), solutions=([[3]],))]
    report = score_submission({}, tasks)
    assert report.unscored == 1
    assert report.total_outputs == 0
    assert report.score == 0.0
