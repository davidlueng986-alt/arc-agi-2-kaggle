import sys
import json
import glob
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.grid import parse_grid_from_text

def generate_truth(limit=None):
    files = sorted(glob.glob("tests/grid_parsing_cases/cases/*.txt"))
    if limit:
        files = files[:limit]
        
    for f in files:
        with open(f, 'r') as fh:
            text = fh.read()
        
        try:
            grid = parse_grid_from_text(text)
            truth_path = Path(f).with_suffix(".truth.json")
            with open(truth_path, 'w') as out:
                json.dump(grid, out)
            print(f"Generated {truth_path}")
        except Exception as e:
            print(f"Failed to parse {f}: {e}")

if __name__ == "__main__":
    # Default to 5 as requested, can be changed to None for all
    generate_truth()
