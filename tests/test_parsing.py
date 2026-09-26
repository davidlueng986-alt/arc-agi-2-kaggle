from harness.parsing import (
    extract_code_blocks,
    looks_like_refusal,
    pick_transform_code,
    truncate,
)

CODE_T = "import numpy as np\n\ndef transform(grid):\n    return grid\n"
CODE_S = "def solver(grid):\n    return grid\n"


class TestCodeBlockExtraction:
    def test_python_fence(self):
        text = f"Explanation here.\n```python\n{CODE_T}\n```\nDone."
        got = pick_transform_code(text)
        assert got is not None
        code, entry = got
        assert entry == "transform"
        assert "def transform" in code

    def test_py_fence_variant(self):
        text = f"```py\n{CODE_T}```"
        code, entry = pick_transform_code(text)
        assert entry == "transform"

    def test_bare_fence(self):
        text = f"```\n{CODE_T}\n```"
        code, entry = pick_transform_code(text)
        assert entry == "transform"

    def test_solver_entry_accepted(self):
        text = f"```python\n{CODE_S}\n```"
        code, entry = pick_transform_code(text)
        assert entry == "solver"

    def test_last_python_block_wins(self):
        draft = "def transform(grid):\n    return None\n"
        text = f"```python\n{draft}\n```\nActually, corrected:\n```python\n{CODE_T}\n```"
        code, _ = pick_transform_code(text)
        assert "return grid" in code
        assert "return None" not in code

    def test_multiple_blocks_prefers_transform_over_solver(self):
        text = f"```python\n{CODE_S}\n```\nthen\n```python\n{CODE_T}\n```"
        code, entry = pick_transform_code(text)
        assert entry == "transform"

    def test_unfenced_code(self):
        text = (
            "Here is the solution.\n\n"
            "import numpy as np\n"
            "def transform(grid):\n"
            "    return np.flip(grid)\n"
        )
        got = pick_transform_code(text)
        assert got is not None
        code, entry = got
        assert entry == "transform"
        assert "np.flip" in code

    def test_no_code_at_all(self):
        assert pick_transform_code("The rule is unclear, sorry.") is None
        assert pick_transform_code("") is None
        assert pick_transform_code(None) is None

    def test_non_python_block_ignored_when_python_present(self):
        text = f"```\n1,2,3\n```\n```python\n{CODE_T}\n```"
        code, entry = pick_transform_code(text)
        assert entry == "transform"

    def test_extract_code_blocks_langs(self):
        text = "```python\nA\n```\n```json\n{}\n```\n```\nB\n```"
        blocks = extract_code_blocks(text)
        assert [lang for lang, _ in blocks] == ["python", "json", ""]


class TestRefusals:
    def test_refusals_detected(self):
        assert looks_like_refusal("I cannot assist with that request.")
        assert looks_like_refusal("I'm sorry, but I can't help.")
        assert looks_like_refusal("As an AI language model, ...")
        assert looks_like_refusal("")
        assert looks_like_refusal("  ")

    def test_normal_text_not_refusal(self):
        assert not looks_like_refusal("The grid is rotated 90 degrees. ```python\ndef transform(g): return g\n```")


class TestTruncate:
    def test_short_unchanged(self):
        assert truncate("abc", 100) == "abc"

    def test_long_truncated(self):
        s = "x" * 5000
        out = truncate(s, 300)
        assert len(out) < 400
        assert "[truncated]" in out

    def test_none(self):
        assert truncate(None, 10) == ""
