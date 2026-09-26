import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.grid import parse_grid_from_text

def debug(txt_file):
    with open(txt_file, 'r') as f:
        text = f.read()
    
    print(f"--- Parsing {txt_file} ---")
    try:
        grid = parse_grid_from_text(text)
        print(f"Grid found. Size: {len(grid)}x{len(grid[0])}")
        print("Row 0:", grid[0])
        
        truth_file = Path(txt_file).with_suffix(".truth.json")
        if truth_file.exists():
            with open(truth_file, 'r') as f:
                truth = json.load(f)
            print("Truth Row 0:", truth[0])
            
            if grid[0] != truth[0]:
                print("MISMATCH at Row 0")
            else:
                print("MATCH at Row 0")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    debug("tests/grid_parsing_cases/cases/gpt-5.1-high_2_step_1_1765155396.197504.txt")
