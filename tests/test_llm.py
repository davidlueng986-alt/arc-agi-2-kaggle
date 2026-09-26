import numpy as np
import pytest

from harness.extraction import compiles, pick_program
from harness.llm import MockLLM, VLLMClient, TEMPLATE_PROGRAMS, library_responder, scripted_responder


def test_mock_llm_token_accounting():
    llm = MockLLM(scripted_responder(["abcd" * 10]))
    resp = llm.chat([{"role": "user", "content": "xy" * 100}])
    assert resp.prompt_tokens == 50
    assert resp.completion_tokens == 10
    assert llm.total_prompt_tokens == 50
    assert llm.calls == 1


def test_scripted_responder_repeats_last():
    r = scripted_responder(["a", "b"])
    assert r([], 0) == "a"
    assert r([], 1) == "b"
    assert r([], 5) == "b"


def test_vllm_client_rejects_invalid_reasoning_effort():
    client = VLLMClient(base_url="http://127.0.0.1:1/v1")
    with pytest.raises(AssertionError):
        client.chat([{"role": "user", "content": "x"}], reasoning_effort="high")


def test_vllm_client_retries_then_raises():
    client = VLLMClient(base_url="http://127.0.0.1:1/v1", max_retries=1, request_timeout=0.2)
    with pytest.raises(RuntimeError):
        client.chat([{"role": "user", "content": "x"}], reasoning_effort="low")


def test_all_template_programs_extract_and_compile():
    llm = MockLLM(library_responder())
    for i in range(len(TEMPLATE_PROGRAMS)):
        resp = llm.chat([{"role": "user", "content": "task"}])
        code = pick_program(resp.text)
        assert code is not None, f"template {i} not extractable"
        assert compiles(code), f"template {i} does not compile"
        assert "def transform" in code


def test_templates_execute_on_random_grids():
    from harness.sandbox import LocalExecutor

    ex = LocalExecutor()
    rng = np.random.default_rng(0)
    for name, code in TEMPLATE_PROGRAMS:
        g = rng.integers(0, 6, size=(5, 7)).astype(np.int64)
        res = ex.run(code, g)
        assert res.ok, f"template {name} failed: {res.error}"
        assert res.output is not None and res.output.ndim == 2
