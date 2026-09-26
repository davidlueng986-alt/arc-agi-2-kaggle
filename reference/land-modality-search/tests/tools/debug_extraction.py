import json
import sys
import re
from pathlib import Path

# Add project root to sys.path to import src modules
sys.path.append(str(Path(__file__).parent))

from src.grid import parse_grid_from_text

def test_extraction(log_file_path):
    try:
        with open(log_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found: {log_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file: {log_file_path}")
        return

    print(f"Testing extraction for log file: {log_file_path}\n")

    for run_key, run_data in data.items():
        print(f"--- Model Run: {run_key} ---")
        
        response_text = run_data.get("Full raw LLM response")
        if not response_text:
            print("  [WARNING] No 'Full raw LLM response' found.")
            continue

        print(f"  Original 'Extracted grid': {run_data.get('Extracted grid') is not None}")
        
        try:
            grid = parse_grid_from_text(response_text)
            if grid:
                print(f"  [SUCCESS] Successfully extracted grid (Size: {len(grid)}x{len(grid[0])})")
                # Optional: Print a small preview
                # print(f"  Preview: {grid[:2]}...") 
            else:
                print("  [FAILURE] Could not extract grid from text.")
                
                # Debugging: check for common tags
                tags = ["grid", "answer", "json"]
                found_tags = []
                for tag in tags:
                    if f"<{tag}>" in response_text:
                        found_tags.append(f"<{tag}>")
                    if f"```{tag}" in response_text:
                         found_tags.append(f"```{tag}")
                
                if found_tags:
                    print(f"  [DEBUG] Found potential tags in text: {found_tags}")
                else:
                    print("  [DEBUG] No standard tags found in text.")

        except Exception as e:
            print(f"  [ERROR] Exception during extraction: {e}")
        
        print("")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_grid_extraction.py <path_to_log_json>")
        sys.exit(1)
    
    log_file = sys.argv[1]
    test_extraction(log_file)
