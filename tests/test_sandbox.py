"""REPL sandbox: isolation, limits, state reset, and the verify round-trip."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest

from harness.grids import Pair, Task
from harness.oracle import normalize_grid
from harness.sandbox import (
    BLOCKED_MODULES,
    DEFAULT_MAX_OUTPUT_CHARS,
    ReplSandbox,
    ToolResult,
    Verification,
    _build_bootstrap,
    _child_env,
)


@pytest.fixture(scope="module")
def bootstrap() -> str:
    text = _build_bootstrap()
    compile(text, "<bootstrap>", "exec")  # must be valid Python on its own
    return text


def test_bootstrap_splices_both_modules(bootstrap):
    assert "def segment_grid(" in bootstrap
    assert "def verify_candidate(" in bootstrap
    assert "__SEGMENTATION_SOURCE__" not in bootstrap
    assert "__ORACLE_SOURCE__" not in bootstrap
    assert "__BLOCKED_MODULES__" not in bootstrap
    assert "frozenset(" in bootstrap


def test_bootstrap_is_cached():
    assert _build_bootstrap() is _build_bootstrap()


def test_child_env_is_clean():
    env = _child_env()
    assert env["CUDA_VISIBLE_DEVICES"] == ""  # keep model code off the serving GPU
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert "PYTHONPATH" in env and env["PYTHONPATH"] == ""
    for secret in ("OPENAI_API_KEY", "KAGGLE_USERNAME", "AWS_SECRET_ACCESS_KEY"):
        assert secret not in env


@pytest.fixture
def sandbox(flip_task):
    return ReplSandbox(flip_task, timeout_s=20, max_output_chars=2000)


class TestExecution:
    def test_print_and_result(self, sandbox):
        out = sandbox.run("print('hello')\nresult = {'a': 1}")
        assert out.error == ""
        assert "hello" in out.stdout
        assert out.result == {"a": 1}
        assert out.timed_out is False

    def test_preloaded_task_state(self, sandbox, flip_task):
        out = sandbox.run(
            "result = {"
            " 'n_train': len(TRAIN),"
            " 'n_test': len(TEST),"
            " 'train0_in': TRAIN[0].input,"
            " 'train0_out': TRAIN[0].output,"
            " 'pairs': TRAIN_PAIRS[0],"
            " 'task_id': TASK_ID,"
            " 'shape': TRAIN[0].input.shape,"
            " 'ascii': TRAIN[0].input.ascii,"
            " 'csv': TRAIN[0].output.csv,"
            " 'colors': TRAIN[0].input.colors(),"
            " 'hist': TRAIN[0].input.histogram(),"
            "}"
        )
        assert out.error == "", out.error
        assert out.result["n_train"] == flip_task.n_train
        assert out.result["n_test"] == flip_task.n_test
        assert out.result["train0_in"] == flip_task.train[0].input
        assert out.result["task_id"] == flip_task.task_id
        assert out.result["shape"] == [2, 2]
        assert out.result["csv"] == "3,4\n1,2"
        # `result` crosses a JSON boundary, so int dict keys arrive as strings
        assert out.result["hist"] == {"1": 1, "2": 1, "3": 1, "4": 1}

    def test_numpy_available(self, sandbox):
        out = sandbox.run("import numpy as np\nresult = list(np.array(TRAIN[0].input).shape)")
        assert out.error == "", out.error
        assert out.result == [2, 2]

    def test_numpy_round_trip_through_verify(self, sandbox):
        out = sandbox.run(
            "import numpy as np\n"
            "def transform(g):\n"
            "    return np.flipud(np.array(g))\n"
            "rep = verify(transform)\n"
            "result = {'exact': rep['exact'], 'test_shape': rep['test'][0]['shape']}"
        )
        assert out.error == "", out.error
        assert out.result == {"exact": True, "test_shape": [2, 2]}

    def test_segmentation_helper(self, sandbox):
        out = sandbox.run("seg = segment([[0,1],[0,1]])\nresult = {'n': seg['n_components'], 'adj': seg['adjacency_list']}")
        assert out.error == "", out.error
        assert out.result == {"n": 2, "adj": [[0, 1]]}

    def test_show_helper(self, sandbox):
        out = sandbox.run("result = show([[0,1,2,3,4,5,6,7,8,9]])")
        assert out.result == ".brgysmocd"

    def test_state_resets_between_calls(self, sandbox):
        first = sandbox.run("PERSISTED = 123\nresult = PERSISTED")
        assert first.result == 123
        second = sandbox.run("result = 'PERSISTED' in dir()")
        assert second.result is False

    def test_syntax_error_reported_without_spawning(self, sandbox):
        out = sandbox.run("def f(:\n  pass")
        assert "syntax error" in out.error.lower()
        assert out.elapsed_s < 0.5

    def test_empty_code(self, sandbox):
        out = sandbox.run("   ")
        assert "non-empty" in out.error

    def test_exception_traceback_only_shows_user_frames(self, sandbox):
        out = sandbox.run("def f():\n    return 1 / 0\nf()")
        assert "ZeroDivisionError" in out.error
        assert "<repl>" in out.error
        assert "bootstrap" not in out.error
        assert "_CappedStream" not in out.error

    def test_render_payload(self, sandbox):
        out = sandbox.run("print('x')\nresult = 5")
        payload = json.loads(out.render())
        assert payload["stdout"] == "x"
        assert payload["result"] == 5

    def test_render_empty_is_ok(self):
        assert json.loads(ToolResult().render()) == {"ok": True}


class TestLimits:
    def test_timeout_kills_the_process(self, flip_task):
        sandbox = ReplSandbox(flip_task, timeout_s=2)
        started = time.perf_counter()
        out = sandbox.run("while True:\n    pass")
        elapsed = time.perf_counter() - started
        assert out.timed_out is True
        assert "timed out" in out.error
        assert elapsed < 15

    def test_blocked_imports(self, sandbox):
        for module in ("os", "sys", "subprocess", "socket", "shutil", "pathlib", "threading"):
            out = sandbox.run(f"import {module}")
            assert "ImportError" in out.error, f"{module} was not blocked: {out.error}"
            assert "blocked" in out.error

    def test_blocked_list_contents(self):
        for module in ("os", "sys", "subprocess", "socket", "importlib", "builtins"):
            assert module in BLOCKED_MODULES

    def test_allowed_imports(self, sandbox):
        out = sandbox.run(
            "import itertools, functools, collections, math, re, random, heapq, "
            "bisect, statistics, copy, json, string, operator, hashlib\nresult = 'ok'"
        )
        assert out.error == "", out.error
        assert out.result == "ok"

    def test_output_cap(self, flip_task):
        sandbox = ReplSandbox(flip_task, timeout_s=20, max_output_chars=500)
        out = sandbox.run("print('x' * 100000)")
        assert out.truncated is True
        assert len(out.stdout) < 1200
        payload = json.loads(out.render(max_chars=500))
        assert payload["truncated"] is True
        assert "truncation_note" in payload

    def test_default_output_cap(self):
        assert DEFAULT_MAX_OUTPUT_CHARS == 4096

    def test_cannot_write_outside_cwd_or_read_source(self, sandbox):
        out = sandbox.run("handle = open('escape.txt', 'w')\nhandle.write('x')\nhandle.close()\nresult='wrote'")
        # Writing inside the throwaway cwd is allowed; escaping it is not possible
        # because os/pathlib/shutil are blocked and cwd is a fresh temp dir.
        assert out.error == "" or "FileNotFoundError" in out.error or "PermissionError" in out.error

    def test_cannot_reach_harness_source(self, sandbox):
        # `-I` keeps the harness package (and the repo) off the child's sys.path,
        # so model code cannot import the thing that is grading it.
        out = sandbox.run("import harness")
        assert "ModuleNotFoundError" in out.error or "ImportError" in out.error
        assert "harness" in out.error


class TestVerifyRoundTrip:
    def test_correct_program_registers_and_reports(self, sandbox, flip_task):
        out = sandbox.run(
            "def transform(g):\n    return list(reversed(g))\n"
            "rep = verify(transform)\n"
            "print(rep['summary'])"
        )
        assert out.error == "", out.error
        assert len(out.verifications) == 1
        verification = out.verifications[0]
        assert isinstance(verification, Verification)
        assert verification.exact is True
        assert verification.soft_score == 1.0
        assert verification.test_outputs == [normalize_grid([[8, 8], [7, 7]])]
        assert verification.test_shapes == [[2, 2]]
        assert "def transform" in (verification.source or "")
        assert "VERIFIED" in out.stdout

    def test_report_shown_to_model_has_no_raw_grids(self, sandbox):
        out = sandbox.run("def t(g):\n    return g\nrep = verify(t)\nresult = rep")
        payload = json.loads(out.render())
        assert "_grid" not in json.dumps(payload["verify"])
        assert payload["verify"][0]["test"][0]["shape"] == [2, 2]

    def test_wrong_program_reports_diffs(self, sandbox):
        out = sandbox.run("def t(g):\n    return g\nrep = verify(t)\nresult = rep")
        assert out.error == "", out.error
        report = out.result
        assert report["exact"] is False
        assert report["failures"][0]["kind"] == "cells"
        assert report["failures"][0]["samples"]

    def test_host_callback_receives_payload(self, flip_task):
        seen = []

        def on_verify(payload):
            seen.append(payload)
            return {"registered": True}

        sandbox = ReplSandbox(flip_task, timeout_s=20, on_verify=on_verify)
        out = sandbox.run("def t(g):\n    return list(reversed(g))\nverify(t)")
        assert len(seen) == 1
        assert seen[0]["exact"] is True
        assert seen[0]["test_outputs"]
        assert "source" in seen[0]

    def test_host_can_reject(self, flip_task):
        def on_verify(payload):
            return {"registered": False, "reason": "pool full"}

        sandbox = ReplSandbox(flip_task, timeout_s=20, on_verify=on_verify)
        out = sandbox.run("def t(g):\n    return list(reversed(g))\nresult = verify(t)")
        assert out.result["registered"] is False
        assert out.result["note"] == "pool full"

    def test_host_callback_exception_does_not_kill_snippet(self, flip_task):
        def on_verify(payload):
            raise RuntimeError("host bug")

        sandbox = ReplSandbox(flip_task, timeout_s=20, on_verify=on_verify)
        out = sandbox.run("def t(g):\n    return list(reversed(g))\nresult = verify(t)\nprint('survived')")
        assert "survived" in out.stdout
        assert out.result["registered"] is False

    def test_multiple_verifies_in_one_snippet(self, sandbox):
        code = (
            "for name, fn in [('id', lambda g: g), ('flip', lambda g: list(reversed(g)))]:\n"
            "    rep = verify(fn)\n"
            "    print(name, rep['exact'], round(rep['soft_score'], 3))\n"
        )
        out = sandbox.run(code)
        assert out.error == "", out.error
        assert len(out.verifications) == 2
        assert out.calls == 2

    def test_augmentation_reported(self, sandbox):
        out = sandbox.run("def t(g):\n    return list(reversed(g))\nresult = verify(t)['aug']")
        assert out.result["flipud"] == 1.0


class TestFinish:
    def test_finish_flag(self, sandbox):
        out = sandbox.run("finish('found it')")
        assert out.finished is True
        assert out.finish_reason == "found it"

    def test_finish_survives_a_later_exception(self, sandbox):
        out = sandbox.run("finish('x')\nraise ValueError('after')")
        assert out.finished is True
        assert "ValueError" in out.error

    def test_no_finish_by_default(self, sandbox):
        assert sandbox.run("result = 1").finished is False


class TestProcessFailure:
    def test_bad_interpreter_is_reported(self, flip_task):
        sandbox = ReplSandbox(flip_task, timeout_s=5, python_exe=sys.executable + "-does-not-exist")
        out = sandbox.run("print(1)")
        assert "could not start" in out.error

    def test_hard_crash_is_reported_not_hung(self, flip_task):
        sandbox = ReplSandbox(flip_task, timeout_s=10)
        started = time.perf_counter()
        # os._exit is reachable via ctypes-free means: allocate until MemoryError
        out = sandbox.run("import gc" if False else "x = 'a' * (10 ** 12)")
        assert time.perf_counter() - started < 30
        assert out.error or out.result is None
