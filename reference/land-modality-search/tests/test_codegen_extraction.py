import sys
import json
import pytest
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.tasks import load_task
from src.parallel.codegen import extract_and_run_solver

TEST_LOGS_DIR = Path(__file__).parent / "codegen_test_logs"

def find_task_path_robust(task_id: str) -> Path:
    # Try different data folders
    search_dirs = [
        "data/evaluation-arc-agi-2",
        "data/training-arc-agi-2",
        "data/evaluation-arc-agi-1"
    ]
    for d in search_dirs:
        candidate = Path(d) / f"{task_id}.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Task {task_id} not found in {search_dirs}")

def get_all_test_runs():
    """
    Returns a list of (log_file_name, run_id, task_id, test_index, raw_response)
    """
    runs = []
    if not TEST_LOGS_DIR.exists():
        return runs
        
    for log_file in TEST_LOGS_DIR.glob("*.json"):
        # Filename format: {timestamp}_{task_id}_{test_index}_{step_suffix}.json
        # Example: 2025-12-27_12-00-11_20a9e565_2_step_1.json
        parts = log_file.stem.split("_")
        if len(parts) < 5:
            continue
            
        task_id = parts[-4]
        test_index = int(parts[-3])
        
        with open(log_file, "r") as f:
            data = json.load(f)
            
        for run_id, run_val in data.items():
            if not isinstance(run_val, dict): continue
            # We want the LLM response, not the call/prompt.
            response_text = run_val.get("Full raw LLM response")
            
            if response_text:
                runs.append((log_file.name, run_id, task_id, test_index, response_text))
    return runs

@pytest.mark.parametrize("log_name, run_id, task_id, test_index, response", get_all_test_runs())
def test_codegen_extraction_and_syntax(log_name, run_id, task_id, test_index, response):
    """
    Tests if the production extraction logic can successfully extract and 
    compile the code from the given response.
    """
    task_path = find_task_path_robust(task_id)
    task = load_task(task_path)
    
    # We want to use the specific test example index
    test_idx = test_index - 1
    if test_idx < 0 or test_idx >= len(task.test):
        pytest.fail(f"Test index {test_index} out of range for task {task_id}")
    
    test_input = task.test[test_idx].input
    train_examples = task.train
    
    # Call the actual production extraction/execution engine
    predicted_grid, verification_log = extract_and_run_solver(
        response, 
        test_input, 
        train_examples=train_examples, 
        task_id=task_id, 
        test_index=test_index
    )
    
    # Assertions
    status = verification_log.get("status")
    error = verification_log.get("error", "No error message")
    
    # We fail if the extraction couldn't even define the code or crashed the system
    assert status != "FAIL_DEFINE", f"Extraction failed with syntax error in {log_name} [{run_id}]: {error}"
    assert status != "FAIL_EXTRACTOR_CRASH", f"Extraction logic crashed in {log_name} [{run_id}]: {error}"
    assert status != "FAIL_NO_SOLVER", f"No 'solver' function found in {log_name} [{run_id}]"

if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
