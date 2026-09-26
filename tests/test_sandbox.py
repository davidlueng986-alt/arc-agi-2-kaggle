import asyncio
import json
import os
import time

from harness.sandbox import run_code


def _run(*args, **kwargs):
    return asyncio.run(run_code(*args, **kwargs))


class TestSandbox:
    def test_correct_code(self):
        code = (
            "import numpy as np\n"
            "def transform(grid):\n"
            "    return np.where(grid == 0, 1, grid)\n"
        )
        ok, out = _run(code, [[0, 2], [3, 0]], timeout_s=30)
        assert ok
        assert json.loads(out) == [[1, 2], [3, 1]]

    def test_solver_entry_bound(self):
        code = "def solver(grid):\n    return grid * 0\n"
        ok, out = _run(code, [[5, 5], [5, 5]], timeout_s=30)
        assert ok
        assert json.loads(out) == [[0, 0], [0, 0]]

    def test_timeout_kills_runaway(self):
        code = "def transform(grid):\n    while True:\n        pass\n"
        t0 = time.time()
        ok, out = _run(code, [[0]], timeout_s=1.0)
        dt = time.time() - t0
        assert not ok
        assert out == "timeout"
        assert dt < 5.0  # returns promptly after the hard timeout

    def test_exception_reported(self):
        code = "def transform(grid):\n    return 1 / 0\n"
        ok, out = _run(code, [[0]], timeout_s=30)
        assert not ok
        assert "ZeroDivisionError" in out

    def test_no_transform_defined(self):
        code = "def other(grid):\n    return grid\n"
        ok, out = _run(code, [[0]], timeout_s=30)
        assert not ok
        assert "no transform" in out.lower()

    def test_non_2d_return_rejected(self):
        code = "def transform(grid):\n    return [1, 2, 3]\n"
        ok, out = _run(code, [[0]], timeout_s=30)
        assert not ok
        assert "2D" in out

    def test_clean_env_hides_parent_secrets(self):
        os.environ["H4_SECRET_TOKEN_XYZ"] = "s3cret"
        try:
            code = (
                "import os\n"
                "def transform(grid):\n"
                "    raise RuntimeError('envcheck=' + str(os.environ.get('H4_SECRET_TOKEN_XYZ', 'absent')))\n"
            )
            ok, out = _run(code, [[0]], timeout_s=30)
            assert not ok
            assert "envcheck=absent" in out
        finally:
            del os.environ["H4_SECRET_TOKEN_XYZ"]

    def test_output_cap_bounds_garbage(self):
        code = (
            "def transform(grid):\n"
            "    print('A' * 5_000_000)\n"
            "    return grid\n"
        )
        ok, out = _run(code, [[1]], timeout_s=30, output_cap=1024)
        # the huge print corrupts the JSON payload; harness must not hang or crash
        assert isinstance(ok, bool)
        assert isinstance(out, str)
        assert len(out) <= 4096

    def test_syntax_error_reported(self):
        code = "def transform(grid)\n    return grid\n"
        ok, out = _run(code, [[0]], timeout_s=30)
        assert not ok
        assert "SyntaxError" in out

    def test_stdin_input_isolation(self):
        # cwd is a temp dir: the script cannot see harness files
        code = (
            "import os\n"
            "def transform(grid):\n"
            "    files = sorted(os.listdir('.'))\n"
            "    raise RuntimeError(str(files))\n"
        )
        ok, out = _run(code, [[0]], timeout_s=30)
        assert not ok
        assert "u.py" in out  # only the generated script lives there
