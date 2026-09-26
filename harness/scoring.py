"""Exact-match scoring against ground truth, mirroring the competition metric.

Score per test output is 1.0 if either attempt matches the solution exactly
(dimensions and every cell), else 0.0. Final score is the mean over all test
outputs, not over tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .grids import Grid, grids_equal
from .submission import SubmissionWriter, validate_submission_obj
from .tasks import ARCTask


@dataclass
class TaskScore:
    task_id: str
    n_outputs: int
    correct: int
    attempt_1_correct: int = 0
    attempt_2_only_correct: int = 0

    @property
    def score(self) -> float:
        return self.correct / self.n_outputs if self.n_outputs else 0.0


@dataclass
class ScoreReport:
    tasks: List[TaskScore] = field(default_factory=list)
    total_outputs: int = 0
    total_correct: int = 0
    attempt_1_correct: int = 0
    attempt_2_only_correct: int = 0
    unscored: int = 0

    @property
    def score(self) -> float:
        return self.total_correct / self.total_outputs if self.total_outputs else 0.0

    @property
    def attempt_1_score(self) -> float:
        return self.attempt_1_correct / self.total_outputs if self.total_outputs else 0.0

    @property
    def second_attempt_gain(self) -> float:
        return self.score - self.attempt_1_score

    def solved_task_ids(self) -> List[str]:
        return [t.task_id for t in self.tasks if t.correct == t.n_outputs]

    def failed_task_ids(self) -> List[str]:
        return [t.task_id for t in self.tasks if t.correct == 0]

    def as_dict(self) -> Dict[str, object]:
        return {
            "score": round(self.score, 6),
            "attempt_1_score": round(self.attempt_1_score, 6),
            "second_attempt_gain": round(self.second_attempt_gain, 6),
            "total_outputs": self.total_outputs,
            "total_correct": self.total_correct,
            "attempt_1_correct": self.attempt_1_correct,
            "attempt_2_only_correct": self.attempt_2_only_correct,
            "unscored_tasks": self.unscored,
            "n_tasks_scored": len(self.tasks),
        }


def score_predictions(
    predictions: Sequence[Dict[str, Grid]], ground_truth: Sequence[Grid]
) -> Tuple[int, int, int]:
    correct = a1 = a2_only = 0
    for i, truth in enumerate(ground_truth):
        if i >= len(predictions):
            break
        entry = predictions[i]
        hit_1 = grids_equal(entry.get("attempt_1", []), truth)
        hit_2 = grids_equal(entry.get("attempt_2", []), truth)
        if hit_1:
            a1 += 1
            correct += 1
        elif hit_2:
            a2_only += 1
            correct += 1
    return correct, a1, a2_only


def score_submission(
    submission: Dict[str, List[Dict[str, Grid]]], tasks: Sequence[ARCTask]
) -> ScoreReport:
    report = ScoreReport()
    for task in tasks:
        if task.solutions is None:
            report.unscored += 1
            continue
        preds = submission.get(task.task_id)
        if preds is None:
            report.unscored += 1
            continue
        correct, a1, a2_only = score_predictions(preds, task.solutions)
        report.tasks.append(
            TaskScore(
                task_id=task.task_id,
                n_outputs=task.n_test,
                correct=correct,
                attempt_1_correct=a1,
                attempt_2_only_correct=a2_only,
            )
        )
        report.total_outputs += task.n_test
        report.total_correct += correct
        report.attempt_1_correct += a1
        report.attempt_2_only_correct += a2_only
    return report


def validate_and_score(path: str, tasks: Sequence[ARCTask]) -> Tuple[List[str], Optional[ScoreReport]]:
    submission = SubmissionWriter.load(path)
    errors = validate_submission_obj(submission, tasks)
    report = score_submission(submission, tasks) if not errors else None
    return errors, report
