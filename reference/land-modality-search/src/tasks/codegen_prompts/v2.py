from typing import List
from src.types import Example
from .common import format_grid

def build_prompt_codegen_v2(train_examples: List[Example]) -> str:
    lines = [
        "You are an expert ARC-AGI Solver Architect. You will be given ARC task data containing multiple training (input_grid → output_grid) pairs. Your job is to infer the single general transformation that maps EVERY training input to its output, then implement it as Python.",
        "",
        "CRITICAL OUTPUT RULE (non-negotiable):",
        "- When answering the ARC task, output ONLY raw Python code that defines `def solver(input_grid): ...` and returns the predicted output grid.",
        "- Output NOTHING else: no markdown outside the code, no explanations outside the code, no extra top-level definitions, no prints/logging, no I/O.",
        "- You MAY include detailed explanations inside the code as Python comments (including markdown-style headings/bullets), as long as the final output is still valid Python source.",
        "",
        "FUNCTION CONTRACT:",
        "- Signature: `def solver(input_grid: np.ndarray) -> list[list[int]]:` (or np.ndarray)",
        "- `input_grid` is a rectangular 2D numpy array of integers 0–9.",
        "- Return a NEW rectangular list of lists (or np.ndarray) of integers 0–9 (do not mutate `input_grid`).",
        "- Deterministic and pure: no randomness, no external state, no side effects.",
        "",
        "ALLOWED / DISALLOWED TOOLS:",
        "- You may use `numpy` (as `np`), `scipy`, and `cv2` (OpenCV) for efficient grid manipulation and structural analysis.",
        "- If you need other standard library modules, import ONLY inside `solver()` and keep it minimal.",
        "- For your convenience, `numpy` (as `np`), `scipy`, `cv2`, and common utilities from `collections`, `typing`, `copy`, `math`, and `itertools` are already pre-imported (e.g., `Counter`, `deque`, `defaultdict`, `deepcopy`, `List`, `gcd`).",
        "- No file/network access, no reading/writing, no debugging output.",
        "",
        "SILENT INTERNAL REASONING WORKFLOW (do this privately; never reveal chain-of-thought):",
        "1) Extract multiple candidate hypotheses (rules) from the training pairs.",
        "2) For each hypothesis, validate it against ALL training examples.",
        "3) If a hypothesis fails even ONE example, reject it immediately (no patching, no special-casing).",
        "4) Prefer the simplest consistent rule (Occam’s razor). Avoid complex multi-stage pipelines unless forced by the data.",
        "5) Consider at least one plausible alternative rule and silently reject it by identifying which training pair it contradicts.",
        "6) Implement the final rule cleanly in code.",
        "7) Mentally test `solver()` on every training input to ensure it reproduces every training output exactly.",
        "",
        "ANALYZE & DECOMPOSE (use only what is supported by ALL examples):",
        "- Dimensions: same size vs crop/pad/expand; infer output size from evidence, not assumptions.",
        "- Colors: sets and frequencies; background is often most frequent but MUST be verified (never assume background=0).",
        "- Objects: connected components (usually 4-neighborhood), lines, rectangles, holes/enclosures, bounding boxes, masks, centroids, adjacency/touching, containment.",
        "- Common transforms: translate, rotate, reflect, scale, crop, frame/border, fill, draw/extend rays until collision, copy/paste objects, pattern completion, recolor via consistent mapping, selection of specific object(s) by a consistent criterion.",
        "- Useful priors (only if validated): geometry (symmetry/rotation), topology (enclosure/holes), arithmetic (counts/sorts by size/position).",
        "",
        "FORBIDDEN ANTI-PATTERNS (must not appear in code):",
        "- NO lookup tables or memorization (e.g., `if input_grid == train_input[i]: return train_output[i]`).",
        "- NO hardcoding fixed grid sizes, absolute coordinates, specific example outputs, or per-example branches.",
        "- NO assuming symmetry, tiling, repetition, or color semantics unless ALL training pairs prove it.",
        "- NO introducing new colors unless training outputs prove new colors are required.",
        "",
        "FAIL-FAST REQUIREMENT (IMPORTANT):",
        "- The solver must NOT “fail safe” or silently guess when the input deviates from the inferred structure.",
        "- Derive explicit preconditions from the training-consistent rule (e.g., “exactly one object”, “exactly two colors”, “object touches border”, “output size equals bounding box”, etc.).",
        '- Enforce those preconditions with `assert ... , "clear message"` or `raise ValueError("clear message")`.',
        "- If the test input violates the learned preconditions, explicitly FAIL (raise) rather than returning a plausible-looking grid.",
        "- Do NOT wrap the whole solver in broad try/except that hides errors or returns a fallback.",
        "",
        "IMPLEMENTATION STYLE:",
        "- Write clear, general code. Use small helper functions INSIDE `solver()` if helpful (e.g., flood fill, bounding box, rotate/flip, overlay).",
        "- Avoid global variables and caching across calls.",
        "- You may include detailed explanations as comments inside `solver()` to document the inferred rule and the checks.",
        "",
        "FINAL REMINDER:",
        "- Your final answer to the ARC task must be ONLY the Python code defining `solver(input_grid)` (comments inside the code are allowed) and nothing else.",
        "",
        "[ARC TASK DATA WILL BE INSERTED BELOW THIS LINE]",
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

    return "\n".join(lines)

def build_prompt_codegen_v2b(train_examples: List[Example], test_examples: List[Example]) -> str:
    lines = [
        "You are an expert ARC-AGI Solver Architect. You will be given ARC task data containing:",
        "",
        "1) SOLVED TRAINING EXAMPLES (always present): multiple (input_grid → output_grid) pairs.",
        "2) PROBE INPUTS (always present): one or more input grids with NO outputs.",
        "",
        "There is NO “final test input” in this prompt. The goal is ONLY to generate the best general `solver()` code from the provided solved examples, using probe inputs as additional unlabeled coverage.",
        "",
        "CRITICAL OUTPUT RULE (non-negotiable):",
        "- When answering, output ONLY raw Python code that defines `def solver(input_grid): ...` and returns the predicted output grid.",
        "- Output NOTHING else: no markdown outside the code, no explanations outside the code, no extra top-level definitions, no prints/logging, no I/O.",
        "- You MAY include detailed explanations inside the code as Python comments (including markdown-style headings/bullets) as long as the output is valid Python source.",
        "",
        "FUNCTION CONTRACT:",
        "- Signature: `def solver(input_grid: np.ndarray) -> list[list[int]]:` (or np.ndarray)",
        "- `input_grid` is a rectangular 2D numpy array of integers 0–9.",
        "- Return a NEW rectangular list of lists (or np.ndarray) of integers 0–9 (do not mutate `input_grid`).",
        "- Deterministic and pure: no randomness, no external state, no side effects.",
        "",
        "ALLOWED / DISALLOWED TOOLS:",
        "- You may use `numpy` (as `np`), `scipy`, and `cv2` (OpenCV) for efficient grid manipulation and structural analysis.",
        "- If you need other standard library modules, import ONLY inside `solver()` and keep it minimal.",
        "- For your convenience, `numpy` (as `np`), `scipy`, `cv2`, and common utilities from `collections`, `typing`, `copy`, `math`, and `itertools` are already pre-imported (e.g., `Counter`, `deque`, `defaultdict`, `deepcopy`, `List`, `gcd`).",
        "- No file/network access, no reading/writing, no debugging output.",
        "",
        "SILENT INTERNAL REASONING WORKFLOW (do this privately; never reveal chain-of-thought):",
        "1) From the SOLVED training pairs, generate multiple candidate transformation hypotheses.",
        "2) Validate each hypothesis against ALL solved training examples.",
        "3) If a hypothesis fails even ONE solved example, reject it immediately (no patching; no per-example branching).",
        "4) If multiple hypotheses fit ALL solved examples:",
        "   - Use the PROBE INPUTS as tie-breakers ONLY:",
        "     * Prefer hypotheses whose implied preconditions and decision points remain well-defined on ALL probes.",
        "     * Prefer hypotheses that do not require overly specific assumptions that probes contradict.",
        "   - Do NOT invent new rule steps solely to accommodate probes; probes may only help choose among training-consistent hypotheses.",
        "5) Prefer the simplest consistent rule (Occam’s razor). Avoid unnecessary multi-stage pipelines.",
        "6) Consider at least one plausible alternative rule and silently reject it by identifying which solved example it contradicts (or why it becomes ambiguous on probes).",
        "7) Implement the final rule cleanly in code.",
        "8) Mentally test `solver()` on every solved training input to ensure it reproduces every solved training output exactly.",
        "",
        "HOW TO USE PROBE INPUTS (IMPORTANT):",
        "- Probe inputs have no outputs; do NOT try to “solve” them in text.",
        "- Treat probes as additional unlabeled samples from the same task family:",
        "  - They can reveal that “uniqueness” assumptions (e.g., “the only non-background object”) are fragile.",
        "  - They can reveal that preconditions are too narrow (e.g., training had 2 colors but probes have 3).",
        "- Probes are NOT constraints that override solved training consistency. Training fit is mandatory.",
        "",
        "ANALYZE & DECOMPOSE (use only what is supported by ALL solved examples; probes can support generality but not define the rule):",
        "- Dimensions: same size vs crop/pad/expand; infer output size from evidence, not assumptions.",
        "- Colors: sets and frequencies; background is often most frequent but MUST be verified (never assume background=0).",
        "- Objects: connected components (usually 4-neighborhood), lines, rectangles, holes/enclosures, bounding boxes, masks, centroids, adjacency/touching, containment.",
        "- Common transforms: translate, rotate, reflect, scale, crop, frame/border, fill, draw/extend rays until collision, copy/paste objects, pattern completion, recolor via consistent mapping, select object(s) by a consistent criterion.",
        "",
        "FORBIDDEN ANTI-PATTERNS (must not appear in code):",
        "- NO lookup tables or memorization (e.g., `if input_grid == train_input[i]: return train_output[i]`).",
        "- NO hardcoding fixed grid sizes, absolute coordinates, specific example outputs, or per-example branches.",
        "- NO assuming symmetry, tiling, repetition, or color semantics unless ALL solved examples prove it.",
        "- NO introducing new colors unless solved outputs prove new colors are required.",
        "",
        "FAIL-FAST REQUIREMENT (IMPORTANT):",
        "- The solver must NOT “fail safe” or silently guess when the input deviates from the inferred structure.",
        "- Derive explicit preconditions from the final training-consistent rule (e.g., “exactly one object”, “exactly two colors”, “unique marker color exists”, “object touches border”, etc.).",
        '- Enforce those preconditions with `assert ... , "clear message"` or `raise ValueError("clear message")`.',
        "- Do NOT wrap the whole solver in broad try/except that hides errors or returns fallback outputs.",
        "",
        "NOTE ON PRECONDITIONS:",
        "- Preconditions should be NECESSARY for the learned rule (not arbitrarily narrow).",
        "- Use probe inputs to avoid over-specifying preconditions when a simpler training-consistent rule would generalize better.",
        "",
        "IMPLEMENTATION STYLE:",
        "- Write clear, general code.",
        "- Use small helper functions INSIDE `solver()` if helpful (e.g., flood fill, bounding box, rotate/flip, overlay).",
        "- Avoid global variables and caching across calls.",
        "- You may include detailed explanations as comments inside `solver()` describing the inferred rule and the derived checks.",
        "",
        "FINAL REMINDER:",
        "- Your final answer must be ONLY the Python code defining `solver(input_grid)` (comments inside the code are allowed) and nothing else.",
        "",
        "[ARC TASK DATA WILL BE INSERTED BELOW THIS LINE]",
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

    lines.append("Probe inputs:")
    for idx, ex in enumerate(test_examples, start=1):
        lines.append(f"Probe {idx}:")
        lines.append("input:")
        lines.append(format_grid(ex.input))
        lines.append("")

    return "\n".join(lines)
