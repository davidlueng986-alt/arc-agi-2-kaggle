import sys
import os
import json
import argparse
from pathlib import Path
import httpx
from dotenv import load_dotenv

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent.parent))

from anthropic import Anthropic

def load_keys():
    """Load keys from config/api_keys.env if present."""
    env_path = Path(__file__).parent.parent.parent / "config" / "api_keys.env"
    if env_path.exists():
        load_dotenv(env_path)

def regenerate_truth(case_path):
    load_keys()
    claude_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    
    if not claude_key:
        print("Error: ANTHROPIC_API_KEY not found.")
        return

    http_client = httpx.Client(verify=False)
    client = Anthropic(api_key=claude_key, http_client=http_client)
    
    case_file = Path(case_path)
    if not case_file.exists():
        print(f"File not found: {case_file}")
        return

    truth_file = case_file.with_suffix(".truth.json")
    
    print(f"Regenerating truth for {case_file.name}...")

    with open(case_file, "r") as f:
        content = f.read()

    prompt = f"""You are a data extraction assistant.
Your task is to extract the FINAL output grid from the text provided below.
The text contains the output of an LLM solving an ARC task. It may contain reasoning, multiple grids, or conversational text.
You must identify the FINAL solution grid intended by the model.

Return ONLY the grid as a raw JSON list of lists of integers. 
Example format: [[0, 1], [2, 3]]
DO NOT wrap in markdown blocks. DO NOT add any other text.

--- BEGIN TEXT ---
{content}
--- END TEXT ---
"""
    
    try:
        response = client.messages.create(
            model="claude-opus-4-5-20251101",
            max_tokens=8192,
            thinking={"type": "enabled", "budget_tokens": 4000},
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Content block 0 is thought, block 1 is text (usually)
        # But SDK might return it differently. We iterate to find text.
        response_text = ""
        for block in response.content:
            if block.type == "text":
                response_text += block.text
        
        response_text = response_text.strip()
        
        # validation
        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "").replace("```", "").strip()
        elif response_text.startswith("```"):
            response_text = response_text.replace("```", "").strip()

        grid = json.loads(response_text)
        
        with open(truth_file, "w") as f:
            json.dump(grid, f)
        print(f"Success for {case_file.name}.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    files = [
        "tests/grid_parsing_cases/cases/gpt-5.1-high_2_step_1_1765155396.197504.txt",
        "tests/grid_parsing_cases/cases/objects_pipeline_gemini-3-high_12_step_5_opus_gen_sol_1765142275.8652031.txt"
    ]
    for f in files:
        regenerate_truth(f)
