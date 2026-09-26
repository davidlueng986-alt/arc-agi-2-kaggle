from harness.extraction import compiles, extract_code_blocks, pick_program


def test_single_python_block():
    text = "Explanation here.\n```python\nimport numpy as np\n\ndef transform(grid):\n    return grid\n```\nbye"
    code = pick_program(text)
    assert code is not None
    assert "def transform" in code
    assert compiles(code)


def test_multiple_blocks_prefers_transform_last():
    text = (
        "First draft:\n```python\ndef helper():\n    pass\n```\n"
        "Final:\n```python\ndef transform(grid):\n    return grid * 2\n```\n"
    )
    code = pick_program(text)
    assert "grid * 2" in code


def test_unfenced_fence_tilde():
    text = "~~~python\ndef transform(grid):\n    return grid\n~~~"
    assert pick_program(text) is not None


def test_unclosed_fence():
    text = "```python\ndef transform(grid):\n    return grid\n"
    code = pick_program(text)
    assert code is not None and "def transform" in code


def test_raw_code_no_fence():
    text = "import numpy as np\ndef transform(grid):\n    return grid\n"
    assert pick_program(text) is not None


def test_no_code_returns_none():
    assert pick_program("I cannot solve this task.") is None
    assert pick_program("") is None


def test_block_without_language_tag():
    text = "```\ndef transform(grid):\n    return grid\n```"
    assert pick_program(text) is not None


def test_extract_code_blocks_counts():
    text = "```python\na=1\n```\ntext\n```\nb=2\n```"
    blocks = extract_code_blocks(text)
    assert len(blocks) == 2
    assert "a=1" in blocks[0] and "b=2" in blocks[1]


def test_compiles():
    assert compiles("x = 1")
    assert not compiles("def f(:")
