import json
import sys

def inspect_run(log_file_path, run_key):
    try:
        with open(log_file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    run_data = data.get(run_key)
    if not run_data:
        print(f"Run key '{run_key}' not found.")
        return

    response_text = run_data.get("Full raw LLM response", "")
    print(f"--- Response Length: {len(response_text)} characters ---")
    print("--- Last 2000 characters of response ---")
    print(response_text[-2000:])
    print("----------------------------------------")

if __name__ == "__main__":
    log_file = "logs_backup_2/2025-12-07_17-21-38_cb2d8a2c_2_step_1.json"
    target_key = "gpt-5.1-high_2_step_1_1765158154.383655"
    inspect_run(log_file, target_key)