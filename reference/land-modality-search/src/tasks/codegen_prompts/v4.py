from typing import List
from src.types import Example
from .common import format_grid

def build_prompt_codegen_v4(train_examples: List[Example], test_examples: List[Example], model_name: str = None) -> str:
    lines = [
        "[ARC TASK DATA START]",
        ""
    ]
    
    lines.append("Solved examples:")
    for idx, ex in enumerate(train_examples, start=1):
        lines.append(f"Example {idx}:")
        lines.append("input:")
        lines.append(format_grid(ex.input))
        lines.append("output:")
        lines.append(format_grid(ex.output))
        lines.append("")

    lines.append("Input-only training data (Probe Inputs):")
    for idx, ex in enumerate(test_examples, start=1):
        lines.append(f"Probe {idx}:")
        lines.append("input:")
        lines.append(format_grid(ex.input))
        lines.append("")
    
    lines.append("[ARC TASK DATA END]")
    lines.append("")

    # OpenAI / GPT Branch (Constraint-Oriented to avoid "Invalid Prompt" / Hidden CoT violation)
    if model_name and "gpt" in model_name.lower():
        lines.extend([
            "*** ANALYSIS & VERIFICATION PROCESS ***",
            "",
            "You are an expert ARC-AGI Solver Architect equipped with a **python tool** (Code Interpreter).",
            "Your goal is to write a final, robust `solver(input_grid)` function.",
            "The `input_grid` provided to `solver` will be a **2D NumPy array**.",
            "",
            "GOAL:",
            "Synthesize a robust Python function `solver(input_grid)` that correctly transforms input grids to output grids.",
            "",
            "VERIFICATION GUIDELINES:",
            "1. **Verification:** Please verify your solution with the tool before submitting the final answer. The Python code you output should be executed in the tool and confirmed to match all `train_inputs` to `train_outputs` exactly.",
            "2. **Generalization Check:** Use the Probe Inputs to ensure your logic is robust and does not crash on different grid sizes or color distributions.",
            "3. **Error Handling:** If verification fails or the code crashes, please correct the logic and re-verify.",
            "4. **Final Output:** Your response should conclude with the final, standalone, and verified `solver` function. Please precede this block with exactly this label: `### FINAL SOLUTION ###`",
            "",
            "Format:",
            "### FINAL SOLUTION ###",
            "```python",
            "import numpy as np",
            "import cv2",
            "",
            "def solver(input_grid):",
            "    # input_grid is a 2D numpy array",
            "    # ...",
            "```"
        ])
    # Default Branch (Process-Oriented for Gemini/Claude etc.)
    else:
        lines.extend([
            "*** INSTRUCTIONS: SCIENTIFIC VERIFICATION PROTOCOL ***",
            "",
            "You are an expert ARC-AGI Solver Architect equipped with a **python tool**.",
            "Your goal is to write a final, robust `solver(input_grid)` function.",
            "The `input_grid` provided to `solver` will be a **2D NumPy array**.",
            "You are provided with **Solved Training Pairs** (to derive the rule) and **Unlabeled Probe Inputs** (to test generalizability).",
            "",
            "**CRITICAL RULE:** Do NOT guess. Do NOT rely on visual intuition alone. You must PROVE your solution works using the tool.",
            "",
            "### PHASE 1: EXPERIMENTATION (Mandatory Tool Usage)",
            "Before answering, you must use the python tool to perform this cycle:",
            "",
            "1.  **Load Data**: The python environment starts empty. Write a script to define:",
            "    *   `train_inputs` and `train_outputs` (from the Solved Examples, converted to **numpy arrays**).",
            "    *   `probe_inputs` (from the Input-Only Data, converted to **numpy arrays**).",
            "",
            "2.  **Analyze**: Use numpy and cv2 to inspect ALL grids (training + probes). ",
            "    *   *Check the Probes*: Do they have different sizes? Do they introduce new colors? ",
            "    *   Ensure your rule does not rely on assumptions that the Probes violate.",
            "",
            "3.  **Prototype & Verify (The Loop)**:",
            "    *   Draft a candidate `solver` function inside the tool.",
            "    *   **Test 1 (Accuracy)**: Run the function on ALL `train_inputs` and assert that the result matches `train_outputs` EXACTLY.",
            "    *   **Test 2 (Generalization)**: Run the function on ALL `probe_inputs`. ",
            "        *   Since you don't have the answers, you cannot check accuracy. ",
            "        *   INSTEAD, CHECK FOR CRASHES. Ensure the function runs without error (e.g., no `IndexError`, `ValueError`) and returns a valid grid.",
            "    *   **Refine**: If any test fails, use `print()` to see the error/diff, rewrite the function, and run the cycle again.",
            '    *   Repeat until your python script prints "ALL TRAINING PASSED & PROBES VALID".',
            "",
            "### PHASE 2: FINAL OUTPUT",
            "Once (and ONLY once) you have a valid python script that has passed the verification loop in the logs:",
            "",
            "1.  Output the final, standalone `solver(input_grid)` function.",
            "2.  The code must be self-contained (import numpy as np, cv2 inside).",
            "3.  **Do not include the test harness or data loading** in the final block. Just the solver function.",
            "4.  **CRITICAL**: You MUST copy your verified solver into a final block preceded by exactly this label: `### FINAL SOLUTION ###`",
            "5.  Your response must conclude with this final block. Do not end with tool output or partial analysis.",
            "",
            "Format:",
            "### FINAL SOLUTION ###",
            "```python",
            "import numpy as np",
            "import cv2",
            "",
            "def solver(input_grid):",
            "    # input_grid is a 2D numpy array",
            "    # ...",
            "```"
        ])
    
    return "\n".join(lines)
