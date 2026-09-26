"""Sandboxed execution of model-generated code.

Port of Poetiq's arc_agi/sandbox.py (pure asyncio subprocess, clean env, hard
timeout) with additions for local-model robustness:
  - output cap (a runaway print loop cannot OOM the harness),
  - `solver` entry auto-bound to `transform`,
  - optional scipy (Kaggle has it; local dev boxes may not),
  - minimal clean env (no API keys, no HOME) + the few vars Windows needs.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile

OUTPUT_CAP = 64 * 1024  # bytes
DEFAULT_TIMEOUT_S = 5.0  # Poetiq config: timeout_s = 5


def _clean_env() -> dict[str, str]:
    env = {"PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1"}
    if sys.platform == "win32":
        # numpy/stdlib need these on Windows; still no secrets, no HOME.
        for k in ("SYSTEMROOT", "WINDIR", "TEMP", "TMP", "PATH"):
            v = os.environ.get(k)
            if v:
                env[k] = v
    return env


def _build_script(code: str, entry: str = "transform") -> str:
    # NOTE: template lines are at column 0 on purpose — model code is interpolated
    # verbatim, so any shared-indent dedent would corrupt it.
    return f"""
# generated file
{code}
if __name__ == '__main__':
    import json
    import sys
    import numpy as np
    try:
        import scipy  # optional; present on Kaggle
    except ImportError:
        pass
    _fn = globals().get('transform') or globals().get('solver')
    if _fn is None:
        raise NameError("no transform()/solver() function defined")
    data = json.load(sys.stdin)
    res = _fn(np.array(data['input']))
    import numpy as _np
    arr = _np.asarray(res)
    if arr.ndim != 2:
        raise ValueError("transform must return a 2D grid, got ndim=%d" % arr.ndim)
    print(json.dumps({{"ok": True, "result": arr.astype(int).tolist()}}))
"""


async def run_code(
    code: str,
    input_grid: list,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    output_cap: int = OUTPUT_CAP,
    entry: str = "transform",
) -> tuple[bool, str]:
    """Run `code` in a subprocess on `input_grid`. Returns (ok, json_result_or_error)."""
    script = _build_script(code, entry)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "u.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            path,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=td,
            env=_clean_env(),
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(
                proc.communicate(input=json.dumps({"input": input_grid}).encode()),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            try:
                await proc.wait()
            except Exception:
                pass
            return False, "timeout"

        stdout = (stdout_b or b"")[:output_cap].decode("utf-8", errors="replace")
        stderr = (stderr_b or b"")[:output_cap].decode("utf-8", errors="replace")

        if proc.returncode != 0:
            return False, (stderr.strip() or stdout.strip() or "execution failed")
        try:
            payload = json.loads(stdout)
            return bool(payload.get("ok")), json.dumps(payload.get("result"))
        except Exception as e:
            return False, f"bad-json: {e}"


def run_code_sync(
    code: str,
    input_grid: list,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    output_cap: int = OUTPUT_CAP,
    entry: str = "transform",
) -> tuple[bool, str]:
    """Synchronous convenience wrapper (used by tests and single-threaded paths)."""
    return asyncio.run(run_code(code, input_grid, timeout_s, output_cap, entry))
