"""The generated Kaggle notebook: metadata, embedded payload, and a real dry run."""

from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebook" / "arc26_h2_duck_repl.ipynb"
METADATA = ROOT / "notebook" / "kernel-metadata.json"
CELLS_DIR = ROOT / "notebook" / "cells"
HARNESS_DIR = ROOT / "harness"
DRIVER = ROOT / "scripts" / "run_notebook_locally.py"

CELL_NAMES = [
    "c01_env.py",
    "c02_install.py",
    "c03_model.py",
    "c04_serve.py",
    "c05_run.py",
    "c06_validate.py",
]


@pytest.fixture(scope="module")
def notebook():
    if not NOTEBOOK.is_file():
        pytest.skip("notebook not built; run scripts/build_notebook.py")
    with open(NOTEBOOK, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def code_cells(notebook):
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


@pytest.fixture(scope="module")
def metadata():
    with open(METADATA, encoding="utf-8") as handle:
        return json.load(handle)


class TestMetadata:
    def test_gpu_is_disabled(self, metadata):
        # Hard constraint: this branch must not consume GPU quota on push.
        assert metadata["enable_gpu"] is False

    def test_internet_is_disabled(self, metadata):
        assert metadata["enable_internet"] is False

    def test_slugs(self, metadata):
        assert metadata["id"] == "liangwanyiudavid/arc26-h2-duck-repl"
        assert metadata["code_file"] == "arc26_h2_duck_repl.ipynb"
        assert metadata["kernel_type"] == "notebook"
        assert metadata["language"] == "python"
        assert metadata["is_private"] is True

    def test_data_sources(self, metadata):
        assert metadata["competition_data_sources"] == ["arc-prize-2026-arc-agi-2"]
        assert "saltb0x/qwen3-8-27b-fp8" in metadata["dataset_data_sources"]
        # NVFP4 is Blackwell-native; L4 is Ada sm_89
        assert not any("nvfp4" in s.lower() for s in metadata["dataset_data_sources"])

    def test_notebook_kaggle_metadata_agrees(self, notebook):
        kaggle = notebook["metadata"]["kaggle"]
        assert kaggle["isGpuEnabled"] is False
        assert kaggle["isInternetEnabled"] is False
        assert kaggle["accelerator"] == "none"
        # Kaggle needs NUMERIC ids here. A slug is silently dropped and the
        # kernel boots with an empty /kaggle/input -- which is what the first two
        # pushes did, so this is pinned to the numbers the working vendored
        # arc-prize-2026 notebooks use.
        sources = kaggle["dataSources"]
        competition = [s for s in sources if s.get("sourceType") == "competition"]
        assert len(competition) == 1
        assert competition[0]["sourceId"] == 133469
        assert competition[0]["databundleVersionId"] == 16010536
        assert all(isinstance(s.get("sourceId"), int) for s in sources)

    def test_nbformat(self, notebook):
        assert notebook["nbformat"] == 4
        assert notebook["cells"]


class TestStructure:
    def test_six_code_cells(self, code_cells):
        assert len(code_cells) == 6

    def test_every_code_cell_compiles(self, code_cells):
        for index, source in enumerate(code_cells):
            compile(source, f"<cell {index}>", "exec")

    def test_cells_match_their_source_files(self, code_cells):
        """The notebook is generated; hand-editing it must not silently drift."""
        payload = _payload_from_cell(code_cells[0])
        for index, name in enumerate(CELL_NAMES):
            expected = (CELLS_DIR / name).read_text(encoding="utf-8")
            actual = code_cells[index]
            if name == "c01_env.py":
                expected = expected.replace("__HARNESS_ZIP_B64__", payload)
            assert actual == expected, f"cell {index} ({name}) is out of sync with notebook/cells/"

    def test_markdown_precedes_each_code_cell(self, notebook):
        kinds = [cell["cell_type"] for cell in notebook["cells"]]
        assert kinds[0] == "markdown"
        for index, kind in enumerate(kinds):
            if kind == "code":
                assert kinds[index - 1] == "markdown"


class TestEmbeddedPayload:
    def test_payload_is_a_zip_of_the_harness(self, code_cells):
        archive = _zip_from_cell(code_cells[0])
        names = set(archive.namelist())
        for module in ("oracle.py", "segmentation.py", "grids.py", "render.py", "sandbox.py",
                       "llm.py", "prompts.py", "context.py", "budget.py", "config.py",
                       "submission.py", "dataio.py", "agent.py", "runner.py", "mock_agent.py",
                       "__init__.py"):
            assert f"harness/{module}" in names, module

    def test_payload_bytes_are_identical_to_the_repo(self, code_cells):
        archive = _zip_from_cell(code_cells[0])
        checked = 0
        for name in archive.namelist():
            if not name.startswith("harness/") or not name.endswith(".py"):
                continue
            local = HARNESS_DIR / name.split("/", 1)[1]
            assert local.is_file(), f"{name} is embedded but missing locally"
            assert archive.read(name) == local.read_bytes(), f"{name} differs from the repo copy"
            checked += 1
        assert checked >= 16

    def test_every_embedded_module_compiles(self, code_cells):
        archive = _zip_from_cell(code_cells[0])
        for name in archive.namelist():
            if name.endswith(".py"):
                compile(archive.read(name).decode("utf-8"), name, "exec")

    def test_payload_is_not_absurdly_large(self, code_cells):
        # It goes into the notebook body, so it must stay reviewable.
        assert _payload_from_cell(code_cells[0]).__len__() < 400_000


def _payload_from_cell(cell_source: str) -> str:
    marker = 'HARNESS_ZIP_B64 = "'
    start = cell_source.index(marker) + len(marker)
    end = cell_source.index('"', start)
    return cell_source[start:end]


def _strip_comments(source: str) -> str:
    """Drop whole-line and trailing comments so assertions test code, not prose."""
    out = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        in_string = False
        quote = ""
        kept = []
        for char in line:
            if in_string:
                kept.append(char)
                if char == quote:
                    in_string = False
                continue
            if char in "\"'":
                in_string = True
                quote = char
                kept.append(char)
                continue
            if char == "#":
                break
            kept.append(char)
        out.append("".join(kept))
    return "\n".join(out)


def _zip_from_cell(cell_source: str) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(base64.b64decode(_payload_from_cell(cell_source))))


class TestKeyFactsEncodedInTheNotebook:
    def test_max_model_len_and_tp(self, code_cells):
        env = code_cells[0]
        assert "MAX_MODEL_LEN = 32768" in env
        assert "TENSOR_PARALLEL = 4" in env
        assert "WORKERS = 16" in env

    def test_serve_flags(self, code_cells):
        serve = _strip_comments(code_cells[3])
        for flag in ("--reasoning-parser", "--tool-call-parser", "--enable-auto-tool-choice",
                     "--enable-prefix-caching", "--mamba-cache-mode", "--trust-remote-code"):
            assert flag in serve, flag
        # MTP / speculative decoding breaks constrained decoding on this model.
        # Checked on comment-stripped source: the flag is named in a comment that
        # explains why it is absent.
        assert "--speculative-config" not in serve
        assert "--speculative" not in serve

    def test_reasoning_effort_is_never_high(self, code_cells):
        for source in map(_strip_comments, code_cells):
            assert '"high"' not in source
            assert "'high'" not in source

    def test_nvfp4_is_rejected(self, code_cells):
        assert "nvfp4" in code_cells[2].lower()
        assert "blocked" in code_cells[2]

    def test_deadline_math(self, code_cells):
        env = code_cells[0]
        assert "TOTAL_SECONDS = 12 * 3600.0" in env
        assert "SAFETY_BUFFER = 600.0" in env
        assert "HARD_DEADLINE = NOTEBOOK_START + TOTAL_SECONDS - SAFETY_BUFFER" in env

    def test_xhigh_is_never_used(self, code_cells):
        # xhigh is the server default and would emit 8-16K thinking tokens per
        # turn: 28h of decode alone at 240 tasks x 14 turns.
        for source in map(_strip_comments, code_cells):
            assert "xhigh" not in source


@pytest.mark.skipif(
    not (DRIVER.is_file() and os.environ.get("ARC_DATA_DIR") or True),
    reason="driver missing",
)
class TestLocalDryRun:
    """Execute the real notebook in a subprocess and check what lands on disk."""

    def _run(self, tmp_path, data_dir, limit=3, max_turns=3):
        env = dict(os.environ)
        env["ARC_DATA_DIR"] = str(data_dir)
        env["ARC26_LIMIT"] = str(limit)
        env["ARC26_MAX_TURNS"] = str(max_turns)
        env["ARC26_CHALLENGE_FILE"] = "arc-agi_evaluation_challenges.json"
        env["PYTHONIOENCODING"] = "utf-8"
        return subprocess.run(
            [
                sys.executable, str(DRIVER),
                "--data-dir", str(data_dir),
                "--limit", str(limit),
                "--max-turns", str(max_turns),
                "--workers", "3",
                "--workdir", str(tmp_path),
                "--keep",
            ],
            cwd=str(tmp_path),
            env=env,
            text=True,
            capture_output=True,
            timeout=1200,
        )

    def test_all_cells_execute_and_the_submission_is_valid(self, tmp_path, real_data_dir):
        completed = self._run(tmp_path, real_data_dir)
        assert "ALL CELLS OK" in completed.stdout, completed.stdout[-4000:] + completed.stderr[-2000:]
        assert completed.returncode == 0

        path = tmp_path / "submission.json"
        assert path.is_file()
        submission = json.loads(path.read_text(encoding="utf-8"))

        sys.path.insert(0, str(ROOT))
        from harness.dataio import load_challenges, task_shapes
        from harness.submission import validate_submission

        shapes = task_shapes(load_challenges(real_data_dir, "arc-agi_evaluation_challenges.json"))
        assert validate_submission(submission, shapes) == []
        # Only 3 tasks were attempted, yet all 120 ids are present: the pre-fill
        # is what makes a kill at any instant safe.
        assert set(submission) == set(shapes)
        assert sum(len(v) for v in submission.values()) == sum(shapes.values())
        assert "schema problems : 0" in completed.stdout

    def test_degrades_to_mock_without_gpu_or_weights(self, tmp_path, real_data_dir):
        completed = self._run(tmp_path, real_data_dir, limit=1, max_turns=2)
        assert "SERVER_READY = False" in completed.stdout
        assert "no usable Qwen3 weights found" in completed.stdout
        assert "MockLLM" in completed.stdout
        assert completed.returncode == 0

    def test_embedded_harness_is_the_one_that_runs(self, tmp_path, real_data_dir):
        completed = self._run(tmp_path, real_data_dir, limit=1, max_turns=2)
        assert "harness written to" in completed.stdout
        extracted = tmp_path / "h2" / "harness"
        assert (extracted / "sandbox.py").is_file()
        assert (extracted / "oracle.py").read_bytes() == (HARNESS_DIR / "oracle.py").read_bytes()

    def test_no_mounted_data_degrades_loudly_instead_of_raising(self, tmp_path):
        """What happens on Kaggle when the data sources fail to attach.

        Observed on the real kernel: an interactive push booted with an empty
        /kaggle/input. The notebook must not die with a papermill traceback --
        it has to say what is missing and how to fix it, and exit cleanly.
        """
        empty = tmp_path / "empty_data"
        empty.mkdir()
        env = dict(os.environ)
        env["ARC_DATA_DIR"] = str(empty)
        env["ARC26_LIMIT"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        completed = subprocess.run(
            [
                sys.executable, str(DRIVER),
                "--data-dir", str(empty),
                "--limit", "1",
                "--workdir", str(tmp_path / "wd"),
                "--keep",
            ],
            cwd=str(tmp_path), env=env, text=True, capture_output=True, timeout=600,
        )
        assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-2000:]
        assert "ALL CELLS OK" in completed.stdout
        assert "no ARC challenge file is reachable" in completed.stdout
        assert "Add input" in completed.stdout
        assert "Traceback" not in completed.stdout
        # An override pointing at an empty directory must not be trusted.
        assert "ignoring it" in completed.stdout

    def test_bad_override_falls_through_to_the_search(self, tmp_path):
        """An ARC_DATA_DIR pointing somewhere empty must be ignored, not trusted.

        The notebook then falls through its own search, which includes
        ``Path.cwd()/data`` -- so we plant a minimal challenge file there and
        check the run proceeds against it rather than crashing.
        """
        workdir = tmp_path / "wd"
        (workdir / "data").mkdir(parents=True)
        challenge = {
            "tiny0001": {
                "train": [{"input": [[0, 1], [2, 3]], "output": [[2, 3], [0, 1]]}],
                "test": [{"input": [[4, 5], [6, 7]]}],
            }
        }
        (workdir / "data" / "arc-agi_evaluation_challenges.json").write_text(
            json.dumps(challenge), encoding="utf-8"
        )
        (workdir / "data" / "arc-agi_evaluation_solutions.json").write_text(
            json.dumps({"tiny0001": [[[6, 7], [4, 5]]]}), encoding="utf-8"
        )

        empty = tmp_path / "empty"
        empty.mkdir()
        env = dict(os.environ)
        env["ARC_DATA_DIR"] = str(empty)
        env["ARC26_LIMIT"] = "1"
        env["ARC26_MAX_TURNS"] = "2"
        env["ARC26_CHALLENGE_FILE"] = "arc-agi_evaluation_challenges.json"
        env["PYTHONIOENCODING"] = "utf-8"

        completed = subprocess.run(
            [
                sys.executable, str(DRIVER),
                "--data-dir", str(workdir / "data"),
                "--keep-arc-data-dir-env",
                "--limit", "1", "--max-turns", "2",
                "--workdir", str(workdir),
                "--keep",
            ],
            cwd=str(workdir), env=env, text=True, capture_output=True, timeout=900,
        )
        assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-2000:]
        assert "ALL CELLS OK" in completed.stdout
        assert "ignoring it" in completed.stdout
        assert "Traceback" not in completed.stdout
        # it found the planted copy via cwd/data and actually solved it
        assert "tasks          : 1" in completed.stdout
        assert "tiny0001" in completed.stdout
        submission = json.loads((workdir / "submission.json").read_text(encoding="utf-8"))
        assert set(submission) == {"tiny0001"}
        assert set(submission["tiny0001"][0]) == {"attempt_1", "attempt_2"}
