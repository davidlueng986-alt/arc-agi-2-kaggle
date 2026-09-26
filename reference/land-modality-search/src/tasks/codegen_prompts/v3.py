from typing import List
from src.types import Example
from .common import format_grid

def _format_v3_data(train_examples: List[Example], test_examples: List[Example]) -> str:
    lines = [
        "Below is an ARC AGI task. You're given the training input/output pairs. You're also given some input-only training data to help you ensure your solution is generalizable.",
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
    return "\n".join(lines)

def build_prompt_codegen_v3_stage1(train_examples: List[Example], test_examples: List[Example]) -> str:
    data_part = _format_v3_data(train_examples, test_examples)
    instructions = [
        "",
        "**Goal:** Analyze the input/output pairs to identify the underlying transformation logic.",
        "**Task:** Do not narrow down to a single definitive rule immediately if there is ambiguity. Instead, output a **Prioritized Plan** containing multiple potential transformation hypotheses or edge-case handling strategies.",
        "**Output Constraint:** Output ONLY the list of hypotheses/strategies in natural language. DO NOT write any Python code.",
        ""
    ]
    return data_part + "\n".join(instructions)

def build_prompt_codegen_v3_stage2(train_examples: List[Example], test_examples: List[Example], hypothesis_plan: str) -> str:
    data_part = _format_v3_data(train_examples, test_examples)
    lines = [
        data_part,
        "",
        "Here are the potential transformation hypotheses and strategies identified by the Analyst:",
        "",
        hypothesis_plan,
        "",
        "**Your Task:**",
        "1. Your task is to write a python function solver(input_grid) that returns the output grid. The input_grid is a 2D numpy array. The solver() function must solve all the input/output pairs. You're also given some input-only training data to help you ensure your solution is generalizable.",
        "2. Implement the correct logic into a Python function named `solver(input_grid)`.",
        "3. Return only the Python code.",
        "",
        "ALLOWED / DISALLOWED TOOLS:",
        "- You may use `numpy` (as `np`), `scipy`, and `cv2` (OpenCV) for efficient grid manipulation and structural analysis.",
        "- If you need other standard library modules, import ONLY inside `solver()` and keep it minimal.",
        "- For your convenience, `numpy` (as `np`), `scipy`, `cv2`, and common utilities from `collections`, `typing`, `copy`, `math`, and `itertools` are already pre-imported (e.g., `Counter`, `deque`, `defaultdict`, `deepcopy`, `List`, `gcd`).",
        "- No file/network access, no reading/writing, no debugging output."
    ]
    return "\n".join(lines)
