"""Regression guard for the Kaggle notebook packaging.

Reproduces the exact failure the first CPU push hit: IPython's `%%writefile` does not
create parent directories, so if `harness/` is not made first every writefile cell
raises FileNotFoundError and the kernel dies in cell 1 -- on a GPU worker too.

These tests simulate the notebook's file-writing cells in order, the way IPython
would, in a clean temp directory.
"""

from __future__ import annotations

import ast
import json
import os

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_PATH = os.path.join(ROOT, "notebook", "arc26_h1_pure_xhigh.ipynb")
MAGIC = "%" + "%writefile"

EXPECTED_MODULES = {
    "__init__.py", "budget.py", "config.py", "diagnostics.py", "grids.py", "llm.py",
    "parse.py", "prompts.py", "runner.py", "scoring.py", "select.py", "submission.py",
    "tasks.py", "watchdog.py", "waves.py",
}


@pytest.fixture(scope="module")
def notebook():
    if not os.path.exists(NB_PATH):
        pytest.skip("notebook not generated yet; run scripts/make_notebook.py")
    with open(NB_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


def code_sources(notebook):
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def test_every_harness_module_is_embedded(notebook):
    found = set()
    for src in code_sources(notebook):
        if src.startswith(MAGIC):
            found.add(os.path.basename(src.partition("\n")[0].split()[1]))
    assert found == EXPECTED_MODULES


def test_writefile_cells_simulate_cleanly_without_parent_creation(notebook, tmp_path):
    """Walk the cells in order; writefile must not need to invent directories."""
    written = {}
    for index, src in enumerate(code_sources(notebook)):
        if src.startswith(MAGIC):
            head, _, body = src.partition("\n")
            target = head.split()[1]
            destination = tmp_path / target
            if not destination.parent.is_dir():
                pytest.fail(
                    f"code cell #{index} writes {target} but {destination.parent.name}/ "
                    f"does not exist yet; IPython %%writefile would raise FileNotFoundError"
                )
            destination.write_text(body, encoding="utf-8")
            written[target] = body
        elif "os.makedirs('harness'" in src:
            (tmp_path / "harness").mkdir(parents=True, exist_ok=True)

    assert set(written) == {f"harness/{m}" for m in EXPECTED_MODULES}
    assert (tmp_path / "harness" / "__init__.py").exists()


def test_embedded_sources_are_byte_identical_to_the_tested_package(notebook, tmp_path):
    for src in code_sources(notebook):
        if not src.startswith(MAGIC):
            continue
        head, _, body = src.partition("\n")
        target = head.split()[1]
        with open(os.path.join(ROOT, target.replace("/", os.sep)), "r", encoding="utf-8") as fh:
            real = fh.read().replace("\r\n", "\n")
        got = body.replace("\r\n", "\n")
        if not got.endswith("\n"):
            got += "\n"
        assert got == real, f"{target} drifted from the tested source"


def test_all_embedded_modules_parse(notebook):
    for src in code_sources(notebook):
        if src.startswith(MAGIC):
            target = src.partition("\n")[0].split()[1]
            ast.parse(src.partition("\n")[2])  # noqa: AST target verified in validate_notebook
            assert target.startswith("harness/")


def test_runtime_cells_parse_and_are_ordered(notebook):
    runtime = [s for s in code_sources(notebook) if not s.startswith(MAGIC)]
    for index, src in enumerate(runtime):
        try:
            ast.parse(src)
        except SyntaxError as exc:
            pytest.fail(f"runtime cell #{index} does not parse: {exc}")

    markers = [
        "RUN_START_MONO = time.monotonic()",
        "def ensure_vllm()",
        "def find_checkpoint()",
        "RUNGS = [",
        "PROBE_EFFORTS =",
        "outcome = run_harness(",
        "H1 pure-xhigh run summary",
    ]
    joined = "\n".join(runtime)
    positions = []
    for marker in markers:
        assert marker in joined, f"missing runtime marker {marker!r}"
        positions.append(joined.index(marker))
    assert positions == sorted(positions), "runtime cells are out of order"


def test_metadata_disables_gpu(notebook):
    meta_path = os.path.join(ROOT, "notebook", "kernel-metadata.json")
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)
    assert meta["enable_gpu"] is False, "enable_gpu must be false; pushing a GPU kernel burns paid quota"
    assert meta["code_file"] == os.path.basename(NB_PATH)
    assert "arc-prize-2026-arc-agi-2" in meta["competition_sources"]
    assert meta["dataset_sources"], "the model checkpoint dataset must be attached"


def test_runtime_avoids_known_vllm_traps(notebook):
    runtime = "\n".join(s for s in code_sources(notebook) if not s.startswith(MAGIC))
    for forbidden, why in (
        ("guided_json", "structured output is unreliable in thinking mode (vLLM#35700)"),
        ("response_format", "doubles braces with reasoning (vLLM#48228)"),
        ('"high"', "reasoning_effort='high' raises HTTP 500 (QwenLM/Qwen3.8#217)"),
        ("speculative-config", "MTP/speculative decoding breaks constrained decoding"),
        ("nvfp4", "NVFP4 is Blackwell-native; L4 is Ada sm_89"),
        ("mamba-cache-mode\", \"all", "prefix caching needs 'align', not 'all'"),
    ):
        assert forbidden not in runtime, f"runtime contains {forbidden!r}: {why}"


def test_effort_mechanism_probe_covers_all_three_paths(notebook):
    runtime = "\n".join(s for s in code_sources(notebook) if not s.startswith(MAGIC))
    assert 'MECHANISMS = ("chat_template_kwargs", "top_level", "none")' in runtime


def test_launcher_has_a_fallback_ladder(notebook):
    runtime = "\n".join(s for s in code_sources(notebook) if not s.startswith(MAGIC))
    for rung in ("tp4-full", "tp4-noalign", "tp4-nocache", "tp4-bare", "tp2-small", "tp2-bare"):
        assert rung in runtime, f"missing vLLM fallback rung {rung}"
    assert '"--mamba-cache-mode", "align"' in runtime


DRIVER = """
import json, os, sys
sys.path.insert(0, os.getcwd())
from harness.budget import BudgetConfig, BudgetGovernor, ManualClock
from harness.config import HarnessConfig, SamplingConfig
from harness.llm import MockLLM
from harness.runner import mock_answers, run_harness
from harness.tasks import load_challenges, resolve_solutions_path

data_dir = sys.argv[1]
tasks = load_challenges(
    os.path.join(data_dir, "arc-agi_evaluation_challenges.json"),
    resolve_solutions_path(data_dir, "evaluation"),
)[:6]
config = HarnessConfig(split="evaluation", output_dir=os.getcwd())
config.sampling = SamplingConfig(phase_a_samples=1, wave_samples=(2,))
config.budget = BudgetConfig(startup_seconds=0.0, planned_tok_per_s=1e9,
                             pessimistic_tok_per_s=1e9)
client = MockLLM(answers=mock_answers(tasks), correct_prob=1.0, tok_per_sec=0)
outcome = run_harness(config, client=client, clock=ManualClock(0.0),
                      logger=lambda _m: None, tasks=tasks)
assert outcome.report.submission_errors == [], outcome.report.submission_errors
assert outcome.score.score == 1.0, outcome.score.score
blob = json.load(open(outcome.submission_path, encoding="utf-8"))
assert len(blob) == 6
print("PACKAGED_HARNESS_OK", outcome.score.score, len(blob))
"""


def test_harness_extracted_from_the_notebook_actually_runs(notebook, tmp_path, data_dir):
    """End-to-end proof that the *packaged* copy works, not just the repo copy."""
    import subprocess
    import sys

    for src in code_sources(notebook):
        if src.startswith(MAGIC):
            head, _, body = src.partition("\n")
            destination = tmp_path / head.split()[1]
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(body, encoding="utf-8")
        elif "os.makedirs('harness'" in src:
            (tmp_path / "harness").mkdir(parents=True, exist_ok=True)

    assert (tmp_path / "harness" / "__init__.py").exists()
    driver = tmp_path / "_driver.py"
    driver.write_text(DRIVER, encoding="utf-8")

    proc = subprocess.run(
        [sys.executable, str(driver), data_dir],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"packaged harness failed:\n{proc.stdout}\n{proc.stderr}"
    assert "PACKAGED_HARNESS_OK" in proc.stdout
