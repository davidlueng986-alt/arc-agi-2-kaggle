"""Notebook artifact integrity: every harness module is embedded verbatim,
compiles, and the Kaggle metadata matches the competition contract."""

import json
import os
import py_compile
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import build_notebook  # noqa: E402


def test_every_module_embedded_verbatim(tmp_path):
    nb = build_notebook.build()
    embedded = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if src.startswith("%%writefile"):
            first, rest = src.split("\n", 1)
            name = first.split("/")[-1].strip()
            embedded[name] = rest

    for module in build_notebook.MODULE_ORDER:
        assert module in embedded, f"{module} missing from notebook"
        with open(os.path.join(build_notebook.HARNESS_DIR, module), encoding="utf-8") as f:
            on_disk = f.read()
        assert embedded[module] == on_disk, f"{module} content differs"

        # the embedded copy must compile
        p = tmp_path / module
        p.write_text(embedded[module], encoding="utf-8")
        py_compile.compile(str(p), doraise=True)


def test_notebook_has_run_and_validate_cells():
    nb = build_notebook.build()
    code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    joined = "\n".join(code)
    assert "GLOBAL_END_TIME = NOTEBOOK_START + 12 * 3600 - 600" in joined
    assert "kaggle_main.main" in joined
    assert "validate_submission" in joined
    assert "/kaggle/working/submission.json" in joined
    assert "KAGGLE_IS_COMPETITION_RERUN" in joined


def test_no_foreign_magics():
    nb = build_notebook.build()
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = cell["source"]
        first = (lines[0].strip() if lines else "")
        if first.startswith("%%writefile"):
            for line in lines[1:]:
                assert not line.startswith(("%%", "%pip", "!pip"))
        else:
            assert not first.startswith("%%")


def test_kernel_metadata_contract():
    path = os.path.join(ROOT, "notebook", "kernel-metadata.json")
    with open(path, encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["enable_gpu"] is False  # hard constraint: do not enable GPU
    assert meta["id"] == "liangwanyiudavid/arc26-h4-poetiq-judge"
    assert meta["kernel_type"] == "notebook"
    assert "arc-prize-2026-arc-agi-2" in meta["competition_sources"]
    assert "saltb0x/qwen3-8-27b-fp8" in meta["dataset_sources"]
    assert meta["code_file"] == "arc26_h4_poetiq_judge.ipynb"


def test_committed_notebook_matches_builder():
    """The checked-in .ipynb must be the one the builder produces."""
    path = os.path.join(ROOT, "notebook", "arc26_h4_poetiq_judge.ipynb")
    assert os.path.exists(path), "run `python scripts/build_notebook.py` first"
    with open(path, encoding="utf-8") as f:
        committed = json.load(f)
    fresh = build_notebook.build()
    assert committed["cells"] == fresh["cells"]
