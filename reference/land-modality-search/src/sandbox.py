import os
import sys
import json
import signal
import subprocess
import tempfile
import time
import traceback
import numpy as np
from typing import Any, Tuple

# This driver script runs INSIDE the subprocess
_SANDBOX_DRIVER = r"""
import json
import sys
import traceback
import math
import itertools
from collections import Counter, deque, defaultdict
from typing import List, Optional, Tuple, Any, Dict, Set
import copy

# Attempt to import common libs
try:
    import numpy as np
except ImportError:
    np = None

try:
    import scipy
    import scipy.ndimage
except ImportError:
    scipy = None

try:
    import cv2
except ImportError:
    cv2 = None

def convert_to_numpy(obj):
    if np is None: return obj
    if isinstance(obj, list):
        return np.array(obj)
    return obj

def sanitize_output(obj):
    if isinstance(obj, list):
        return [sanitize_output(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_output(x) for x in obj)
    if isinstance(obj, dict):
        return {k: sanitize_output(v) for k, v in obj.items()}
    if np and isinstance(obj, (np.integer, int)):
        return int(obj)
    if np and isinstance(obj, (np.floating, float)):
        return float(obj)
    if np and isinstance(obj, np.ndarray):
        return sanitize_output(obj.tolist())
    return obj

def secure_runtime():
    import sys
    import os

    # --- Layer 1: Socket Monkey Patching (Patch FIRST) ---
    # We import and cripple the socket module immediately. 
    # Even if Layer 2 (Poisoning) is bypassed or undone, the cached module object 
    # remains broken ("booby-trapped").
    try:
        import socket
        def blocked_op(*args, **kwargs):
            raise RuntimeError("Sandbox Violation: Network operation blocked by monkey-patch.")
        
        socket.socket = blocked_op
        socket.create_connection = blocked_op
        socket.getaddrinfo = blocked_op
        socket.gethostbyname = blocked_op
    except (ImportError, AttributeError, TypeError):
        pass 

    # --- Layer 2: Network Poisoning (Poison SECOND) ---
    # Disable specific libraries by injecting None into sys.modules.
    # This prevents them from being imported in the first place.
    banned_modules = [
        # Fundamentals
        'socket', 'ssl', 'asyncio',
        
        # Standard Library Clients
        'requests', 'urllib3', 'ftplib', 'poplib', 'imaplib', 'nntplib', 'smtplib', 'telnetlib', 
        
        # Modern Async & WebSockets
        'httpx', 'aiohttp', 'websockets',
        
        # SSH / Cloud / System
        'paramiko', 'boto3', 'botocore', 'google', 'azure', 'subprocess'
    ]
    
    for mod in banned_modules:
        sys.modules[mod] = None

    # --- Layer 3: Audit Hooks ---
    # The final line of defense for events that might bypass python layers.
    def audit_hook(event, args):
        # 1. Block Low-Level Network Access
        if event in ["socket.bind", "socket.connect", "http.client.connect", "urllib.request"]:
            raise RuntimeError(f"Sandbox Violation: Network attempt detected ({event})")
        
        # 2. Block Subprocesses
        if event in ["os.system", "subprocess.Popen", "os.posix_spawn"]:
            raise RuntimeError("Sandbox Violation: Subprocess detected")

    if hasattr(sys, 'addaudithook'):
        sys.addaudithook(audit_hook)

def main():
    try:
        # Secure the runtime environment immediately
        secure_runtime()

        # Read payload from stdin
        input_data = sys.stdin.read()
        if not input_data:
            raise ValueError("No input received on stdin")
            
        payload = json.loads(input_data)
        code = payload["code"]
        inp_raw = payload["input"]
        
        # Convert input list to numpy array if available
        inp = convert_to_numpy(inp_raw)

        # Build execution scope
        local_scope = {
            "np": np,
            "cv2": cv2,
            "scipy": scipy,
            "Counter": Counter,
            "deque": deque,
            "defaultdict": defaultdict,
            "List": List,
            "Optional": Optional,
            "Tuple": Tuple,
            "Any": Any,
            "Dict": Dict,
            "Set": Set,
            "copy": copy.copy,
            "deepcopy": copy.deepcopy,
            "gcd": math.gcd,
            "math": math,
            "itertools": itertools,
            "Grid": List[List[int]]
        }

        # Execute the definition
        exec(code, local_scope)
        
        if "solver" not in local_scope:
            raise RuntimeError("No 'solver' function defined in code.")

        solver = local_scope["solver"]
        if not callable(solver):
            raise RuntimeError("'solver' is not callable.")

        # Run the solver
        raw_out = solver(inp)
        
        # Serialize output
        out = sanitize_output(raw_out)

        json.dump({"ok": True, "output": out}, sys.stdout)
        
    except Exception as e:
        # Print error details to stdout as JSON
        json.dump(
            {
                "ok": False, 
                "error": f"{type(e).__name__}: {str(e)}", 
                "traceback": traceback.format_exc()
            },
            sys.stdout
        )
        # Also print to stderr for debugging logs
        print(f"Sandbox Error: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)

if __name__ == "__main__":
    main()
"""

def _preexec_new_pgrp():
    """
    Sets the process group ID to the PID of the new process.
    This allows killing the entire process group later.
    """
    os.setsid()

def run_untrusted_code(code: str, input_data: Any, timeout_s: float = 10.0) -> Tuple[bool, Any, str]:
    """
    Runs untrusted code in a separate subprocess.
    Returns: (success, result_or_error, logs)
    
    success: bool
    result_or_error: result data if success, else error message
    logs: captured stderr
    """
    
    # Convert numpy inputs to list for JSON serialization
    if isinstance(input_data, np.ndarray):
        input_data = input_data.tolist()
        
    payload = {"code": code, "input": input_data}
    
    # Create a temporary file for the driver
    # We use a temp file instead of passing code via -c to avoid shell escaping hell
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as driver_file:
        driver_file.write(_SANDBOX_DRIVER)
        driver_path = driver_file.name

    try:
        p = subprocess.Popen(
            [sys.executable, "-u", driver_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=_preexec_new_pgrp, # Unix only, crucial for killpg
        )

        try:
            stdout_data, stderr_data = p.communicate(input=json.dumps(payload), timeout=timeout_s)
        except subprocess.TimeoutExpired:
            # Kill the process group
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass # Already dead
            
            return False, "TIMEOUT_EXPIRED", f"Execution timed out after {timeout_s}s"

        if p.returncode != 0:
            # Crashed without JSON output (e.g., segfault or syntax error in driver)
            return False, f"Subprocess crashed (Exit Code: {p.returncode})", stderr_data

        if not stdout_data.strip():
             return False, "Empty output from subprocess", stderr_data

        try:
            result = json.loads(stdout_data)
        except json.JSONDecodeError:
             return False, "Invalid JSON output from subprocess", f"Stdout: {stdout_data}\nStderr: {stderr_data}"

        if result.get("ok"):
            return True, result["output"], stderr_data
        else:
            return False, result.get("error", "Unknown error"), result.get("traceback", stderr_data)

    except Exception as e:
        return False, f"System Error in Sandbox: {e}", str(e)
    finally:
        if os.path.exists(driver_path):
            try:
                os.remove(driver_path)
            except OSError:
                pass
