import sys
import os
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
import httpx

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models import call_model
from anthropic import Anthropic

def load_keys():
    """Load keys from config/api_keys.env if present."""
    env_path = Path(__file__).parent.parent.parent / "config" / "api_keys.env"
    if env_path.exists():
        load_dotenv(env_path)

def generate_truth(cases_dir, force=False):
    load_keys()
    claude_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    
    if not claude_key:
        print("Error: ANTHROPIC_API_KEY not found in environment or config/api_keys.env")
        return

    http_client = httpx.Client(verify=False)
    client = Anthropic(api_key=claude_key, http_client=http_client)
    
    case_files = sorted(list(Path(cases_dir).glob("*.txt")))
    print(f"Found {len(case_files)} test cases in {cases_dir}")
    
    # Process all files
    # case_files = case_files[:5]

    for case_file in case_files:
        truth_file = case_file.with_suffix(".truth.json")
        
        if truth_file.exists() and not force:
            continue

        try:
            with open(case_file, "r") as f:
                content = f.read()

            prompt = f"""You are a data extraction assistant.
Your task is to extract the FINAL output grid from the text provided below.
The text contains the output of an LLM solving an ARC task. It may contain reasoning, multiple grids, or conversational text.
Your must identify the FINAL solution grid intended by the model.

Return ONLY the grid as a raw JSON list of lists of integers. 
Example format: [[0, 1], [2, 3]]
DO NOT wrap in markdown blocks. DO NOT add any other text.

--- BEGIN TEXT ---
{content}
--- END TEXT ---
"""
            
            response = client.messages.create(
                model="claude-opus-4-5-20251101",
                max_tokens=4096,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # validation
            try:
                # Clean up markdown if model ignores instruction
                if response_text.startswith("```json"):
                    response_text = response_text.replace("```json", "").replace("```", "").strip()
                elif response_text.startswith("```"):
                    response_text = response_text.replace("```", "").strip()

                grid = json.loads(response_text)
                if not isinstance(grid, list) or not grid or not isinstance(grid[0], list):
                    raise ValueError("Not a valid grid structure")
                
                with open(truth_file, "w") as f:
                    json.dump(grid, f)
                    
            except json.JSONDecodeError:
                print(f"\n[ERROR] Failed to parse JSON for {case_file.name}")
                print(f"Response: {response_text[:100]}...")
            except ValueError as e:
                print(f"\n[ERROR] Invalid grid structure for {case_file.name}: {e}")

        except Exception as e:
            print(f"\n[ERROR] processing {case_file.name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ground truth grids for test cases using an LLM.")
    parser.add_argument("--cases-dir", default="tests/grid_parsing_cases/cases", help="Directory containing .txt test cases")
    parser.add_argument("--force", action="store_true", help="Overwrite existing .truth.json files")
    
    args = parser.parse_args()
    generate_truth(args.cases_dir, force=args.force)
