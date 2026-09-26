from typing import List
from src.types import Example
from .common import format_grid

def build_prompt_codegen_v1(train_examples: List[Example]) -> str:
    lines = [
        "Below is an ARC AGI task. You're given the training input/output pairs in python. Your task is to write a python function solver(input_grid) that returns the output grid. The input_grid is a 2D numpy array. The solver() function must solve all the input/output pairs",
        "",
        "You may use numpy, scipy, and cv2 (OpenCV) for grid manipulation.",
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

    lines.append("Only output the python code for the solver() function")
    return "\n".join(lines)

def build_prompt_codegen_v1b(train_examples: List[Example], test_examples: List[Example], model_name: str = None) -> str:
    lines = [
        "Below is an ARC AGI task. You're given the training input/output pairs. Your task is to write a python function solver(input_grid) that returns the output grid. The input_grid is a 2D numpy array. The solver() function must solve all the input/output pairs. You're also given some input-only training data to help you ensure your solution is generalizable.",
        "",
        "You may use numpy, scipy, and cv2 (OpenCV) for grid manipulation.",
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

    lines.append("Input-only training data:")
    for idx, ex in enumerate(test_examples, start=1):
        lines.append(f"Probe {idx}:")
        lines.append("input:")
        lines.append(format_grid(ex.input))
        lines.append("")

    if model_name and "gemini" in model_name.lower():
        lines.append("*** MANDATORY REASONING PROTOCOL (High Reasoning Mode) ***")
        lines.append("You are running with a HIGH thinking budget. Do NOT output the code immediately.")
        lines.append("You must first output a verbose \"Thinking Process\" that follows these steps:")
        lines.append("")
        lines.append("1.  **Object Analysis**: Explicitly list the objects, colors, and geometric patterns you see in the Training Examples.")
        lines.append("2.  **Hypothesis Generation**: Propose a logic rule that transforms Input -> Output.")
        lines.append("3.  **Falsification (Crucial)**: Pick the *last* Training Example and the first Probe. Mentally \"dry run\" your proposed rule pixel-by-pixel. Write out the trace (e.g., \"Pixel (0,0) is 8, rule says change to...\").")
        lines.append("4.  **Refinement**: If the dry run fails or is ambiguous on the Probe, refine the rule.")
        lines.append("")
        lines.append("Only after this exhaustive analysis, output the final python code wrapped in a markdown block:")
        lines.append("```python")
        lines.append("def solver(input):")
        lines.append("    # ...")
    else:
        lines.append("Only output the python code for the solver() function")

    return "\n".join(lines)
