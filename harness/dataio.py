"""Data loading and Kaggle path detection.

At rerun the real 240 unseen tasks live under the competition mount; in an
interactive "Save & Run" the same directory holds the 120-task evaluation split
plus its solutions, which is what ``scripts/eval_local.py`` scores against.
``KAGGLE_IS_COMPETITION_RERUN`` distinguishes them -- the same flag the Duck's
Kaggle notebook and the lb33.89 baseline both key on.
"""

from __future__ import annotations

import json
import os
from typing import Any, Iterable, Sequence

from .grids import Grid, Pair, Task, task_from_dict
from .oracle import normalize_grid

COMPETITION_SLUG = "arc-prize-2026-arc-agi-2"
KAGGLE_COMPETITION_DIR = f"/kaggle/input/competitions/{COMPETITION_SLUG}"
KAGGLE_WORKING_DIR = "/kaggle/working"

CHALLENGE_FILES = (
    "arc-agi_test_challenges.json",
    "arc-agi_evaluation_challenges.json",
    "arc-agi_training_challenges.json",
)
SOLUTION_FILES = {
    "arc-agi_test_challenges.json": "arc-agi_test_solutions.json",
    "arc-agi_evaluation_challenges.json": "arc-agi_evaluation_solutions.json",
    "arc-agi_training_challenges.json": "arc-agi_training_solutions.json",
}


def is_competition_rerun() -> bool:
    return os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}


def on_kaggle() -> bool:
    return os.path.isdir("/kaggle/input") or is_competition_rerun()


def find_data_dir(explicit: str | None = None) -> str:
    """Locate a directory holding ARC challenge JSON files.

    An explicitly supplied directory is *strict*: if it does not contain a
    challenge file we raise rather than silently falling back to some other
    location, because a wrong ``--data-dir`` must not quietly score a different
    split.
    """
    if explicit:
        if os.path.isdir(explicit) and any(
            os.path.isfile(os.path.join(explicit, name)) for name in CHALLENGE_FILES
        ):
            return explicit
        raise FileNotFoundError(f"no ARC-AGI-2 challenge file in the explicit data dir: {explicit}")

    candidates: list[str] = []
    env_dir = os.environ.get("ARC_DATA_DIR")
    if env_dir:
        candidates.append(env_dir)
    candidates.extend(
        [
            KAGGLE_COMPETITION_DIR,
            "/kaggle/input/arc-prize-2026-arc-agi-2",
            os.path.join(os.getcwd(), "data"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data")),
            os.path.join(os.path.expanduser("~"), "Downloads", "arc-agi-2", "data"),
        ]
    )
    for directory in candidates:
        if directory and os.path.isdir(directory):
            if any(os.path.isfile(os.path.join(directory, name)) for name in CHALLENGE_FILES):
                return directory
    raise FileNotFoundError(
        "no ARC-AGI-2 data directory found; looked in: " + ", ".join(c for c in candidates if c)
        + " (set ARC_DATA_DIR or pass --data-dir)"
    )


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_challenges(data_dir: str, filename: str) -> dict[str, dict]:
    path = os.path.join(data_dir, filename)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return load_json(path)


def load_solutions(data_dir: str, challenge_filename: str) -> dict[str, list[Grid]] | None:
    name = SOLUTION_FILES.get(challenge_filename)
    if not name:
        return None
    path = os.path.join(data_dir, name)
    if not os.path.isfile(path):
        return None
    raw = load_json(path)
    out: dict[str, list[Grid]] = {}
    for task_id, grids in raw.items():
        normalized = [normalize_grid(g) for g in grids]
        out[task_id] = [g for g in normalized if g is not None]
    return out


def pick_challenge_file(data_dir: str, *, prefer_test: bool | None = None) -> str:
    """Choose which challenge file to solve.

    Defaults to the test file at rerun and the evaluation file otherwise, which
    is what makes the same notebook produce a real submission on rerun and a
    locally scoreable run interactively.
    """
    if prefer_test is None:
        prefer_test = is_competition_rerun()
    order = (
        ["arc-agi_test_challenges.json", "arc-agi_evaluation_challenges.json"]
        if prefer_test
        else ["arc-agi_evaluation_challenges.json", "arc-agi_test_challenges.json"]
    )
    for name in order:
        if os.path.isfile(os.path.join(data_dir, name)):
            return name
    for name in CHALLENGE_FILES:
        if os.path.isfile(os.path.join(data_dir, name)):
            return name
    raise FileNotFoundError(f"no challenge file under {data_dir}")


def iter_tasks(
    challenges: dict[str, dict],
    *,
    limit: int | None = None,
    task_ids: Iterable[str] | None = None,
    skip_invalid: bool = True,
) -> list[Task]:
    wanted = set(task_ids) if task_ids else None
    tasks: list[Task] = []
    for task_id, blob in challenges.items():
        if wanted is not None and task_id not in wanted:
            continue
        try:
            tasks.append(task_from_dict(task_id, blob))
        except Exception:
            if not skip_invalid:
                raise
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def task_shapes(challenges: dict[str, dict]) -> dict[str, int]:
    """``{task_id: n_test_inputs}`` -- what the submission schema must cover."""
    shapes: dict[str, int] = {}
    for task_id, blob in challenges.items():
        shapes[task_id] = len(blob.get("test") or [])
    return shapes


def sample_submission_path(data_dir: str) -> str | None:
    path = os.path.join(data_dir, "sample_submission.json")
    return path if os.path.isfile(path) else None


def load_sample_submission(data_dir: str) -> dict[str, Any] | None:
    path = sample_submission_path(data_dir)
    return load_json(path) if path else None


def default_submission_path() -> str:
    if on_kaggle():
        return os.path.join(KAGGLE_WORKING_DIR, "submission.json")
    return os.path.abspath(os.path.join(os.getcwd(), "submission.json"))
