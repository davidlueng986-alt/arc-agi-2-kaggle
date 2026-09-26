import re
import sys
import copy
import traceback
import time
from src.augmentation import get_augmented_pairs
from src.sandbox import run_untrusted_code

def sanitize_output(obj):
    """Recursively converts numpy types to standard Python types."""
    if isinstance(obj, list):
        return [sanitize_output(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_output(x) for x in obj)
    if isinstance(obj, dict):
        return {k: sanitize_output(v) for k, v in obj.items()}
    
    # Numpy checks (using string matching if numpy not imported, but here it is imported)
    try:
        import numpy as np
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return sanitize_output(obj.tolist())
    except ImportError:
        pass
        
    return obj

def extract_and_run_solver(llm_code: str, test_input_grid: list, train_examples: list = None, task_id: str = None, test_index: int = None) -> tuple[list | None, dict | None]:
    """
    Extracts Python code from LLM response, executes it using a robust sandbox (subprocess), 
    and returns the predicted grid.
    
    If train_examples is provided, it verifies the solver against all training pairs first.
    Returns: (predicted_grid, verification_log)
    """
    verification_log = {"train_results": [], "status": "UNKNOWN"}
    
    # Prefix for logs
    log_prefix = f"[{task_id}:{test_index}]" if task_id else "[Unknown Task]"

    try:
        code = llm_code
        
        # Multi-stage extraction for robustness
        code_search_area = None
        
        # Stage 1: Explicit marker search (Preferred for v4)
        if "### FINAL SOLUTION ###" in llm_code:
            parts = llm_code.split("### FINAL SOLUTION ###")
            # Take the part after the marker
            code_search_area = parts[-1]
        
        # Stage 2: Fallback - Search all markdown blocks in reverse for 'def solver'
        pattern = r"```python(.*?)```"
        if not code_search_area:
            blocks = re.findall(pattern, llm_code, re.DOTALL)
            for block in reversed(blocks):
                if "def solver" in block:
                    code = block.strip()
                    code_search_area = "FOUND_IN_BLOCK"
                    break
        
        # Stage 3: If we have a search area (from marker or default), extract the block
        if code_search_area and code_search_area != "FOUND_IN_BLOCK":
            match = re.search(pattern, code_search_area, re.DOTALL)
            if match:
                code = match.group(1).strip()
            else:
                # Heuristic: If marker exists but no markdown after it, or no marker and no markdown found
                if "def solver" in code_search_area:
                    lines = code_search_area.splitlines()
                    start_idx = -1
                    for i, line in enumerate(lines):
                        if "def solver" in line:
                            start_idx = i
                            break
                    if start_idx != -1:
                        code = "\n".join(lines[start_idx:])
        
        # Stage 4: Ultimate fallback - search entire raw response if nothing found yet
        if not code_search_area:
            if "def solver" in llm_code:
                lines = llm_code.splitlines()
                start_idx = -1
                for i, line in enumerate(lines):
                    if "def solver" in line:
                        start_idx = i
                        break
                if start_idx != -1:
                    code = "\n".join(lines[start_idx:])

        # Verification Step
        if train_examples:
            all_passed = True
            first_fail_status = None
            first_fail_index = -1

            for i, ex in enumerate(train_examples):
                entry = {
                    "index": i, 
                    "status": "UNKNOWN",
                    "input": ex.input, 
                    "expected": ex.output,
                    "actual": None
                }
                
                # Execute in Sandbox
                exec_id = f"{task_id}:{test_index}:TRAIN:{i}:{time.time():.6f}"
                # print(f"DEBUG_EXEC: START {exec_id}", file=sys.stderr)
                try:
                    success, result, logs = run_untrusted_code(code, ex.input, timeout_s=10.0)
                finally:
                    # print(f"DEBUG_EXEC: FINISH {exec_id}", file=sys.stderr)
                    pass
                
                if not success:
                    print(f"DEBUG {log_prefix}: Solver FAILED on Train Example {i+1}: {result}\nDetails:\n{logs}", file=sys.stderr)
                    if result == "TIMEOUT_EXPIRED":
                        entry["status"] = "TIMEOUT"
                        entry["error"] = logs
                    else:
                        entry["status"] = "CRASH"
                        entry["error"] = str(result) + "\n" + str(logs)
                    
                    all_passed = False
                    if first_fail_status is None:
                        first_fail_status = "FAIL_CRASH"
                        first_fail_index = i
                else:
                    # Check Accuracy
                    # result is already sanitized by sandbox driver
                    entry["actual"] = result
                    if result != ex.output:
                        entry["status"] = "FAIL"
                        all_passed = False
                        if first_fail_status is None:
                            first_fail_status = "FAIL_VERIFICATION"
                            first_fail_index = i
                    else:
                        entry["status"] = "PASS"
                
                verification_log["train_results"].append(entry)
            
            if not all_passed:
                verification_log["status"] = first_fail_status
                verification_log["failed_example_index"] = first_fail_index
                return None, verification_log
            
            verification_log["status"] = "PASS"

            # --- Augmentation Verification (Soft Check) ---
            # Skipping implementation details for augmentation in this snippet to keep it focused on core replacement
            # but in production you'd loop run_untrusted_code similarly.
            
        # Test Execution
        exec_id_test = f"{task_id}:{test_index}:TEST:{time.time():.6f}"
        # print(f"DEBUG_EXEC: START {exec_id_test}", file=sys.stderr)
        try:
            success, result, logs = run_untrusted_code(code, test_input_grid, timeout_s=10.0)
        finally:
            # print(f"DEBUG_EXEC: FINISH {exec_id_test}", file=sys.stderr)
            pass
        
        if success:
            if isinstance(result, list):
                 if len(result) > 0 and isinstance(result[0], list):
                     return result, verification_log
                 if len(result) == 0:
                     return result, verification_log
            
            print(f"DEBUG {log_prefix}: Solver returned invalid type (not list of lists).", file=sys.stderr)
            verification_log["test_run_error"] = "Result validation failed (not list of lists)"
            return None, verification_log
        else:
            print(f"DEBUG {log_prefix}: Solver CRASHED on Test Input: {result}", file=sys.stderr)
            verification_log["test_run_error"] = f"Test execution failed: {str(result)}"
            return None, verification_log

    except Exception as e:
        print(f"DEBUG {log_prefix}: Extractor System Crash: {e}", file=sys.stderr)
        verification_log["status"] = "FAIL_EXTRACTOR_CRASH"
        verification_log["error"] = f"{type(e).__name__}: {str(e)}"
        verification_log["traceback"] = traceback.format_exc()
        return None, verification_log