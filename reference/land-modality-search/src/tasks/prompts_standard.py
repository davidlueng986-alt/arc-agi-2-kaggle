import random
from typing import List
from src.grid import format_grid
from src.types import Example

def build_objects_extraction_prompt(
    train_examples: List[Example],
    test_example: Example
) -> str:
    lines = [
        "Describe the types objects involved in the grids below. Do not infer rules, transformations, or relationships between grids. Focus solely on the types of objects that are involved in the various grids and the objects attributes, and describe them generally - not for each grid.",
        "When describing colors, use ONLY the numeric values (0-9), e.g., 'color 0', 'color 5'. Do not use color names like 'black' or 'gray'.",
        "After describing the objects, please provide a concise summary of the object types and attributes within <objects_summary>...</objects_summary> tags.",
        ""
    ]
    
    all_grids = []
    for ex in train_examples:
        all_grids.append(ex.input)
        if ex.output:
            all_grids.append(ex.output)
    all_grids.append(test_example.input)
    random.shuffle(all_grids)
    
    for grid in all_grids:
        lines.append(format_grid(grid))
        lines.append("")
    
    return "\n".join(lines)

def build_objects_transformation_prompt(
    train_examples: List[Example],
    test_example: Example,
    transformation_text: str
) -> str:
    lines = [
        "Below is a set of input / output grids and a description of the objects in each of these grids. Your task is to describe all potential transformations that are involved in changing the objects in the input to the objects in the output. Please ensure that you list ALL possibly transformations that you have identified.",
        "Use strictly numeric values (0-9) for colors in your description and summary.",
        "After describing the transformations, please provide a concise summary of the rules and changes within <transformation_summary>...</transformation_summary> tags.",
        ""
    ]
    for idx, ex in enumerate(train_examples, start=1):
        lines.append(f"Example {idx}:")
        lines.append("input:")
        lines.append(format_grid(ex.input))
        lines.append("output:")
        lines.append(format_grid(ex.output))
        lines.append("")
    lines.append("Test input:")
    lines.append(format_grid(test_example.input))
    lines.append("")
    lines.append("## Objects Description")
    lines.append(transformation_text)
    
    return "\n".join(lines)

def build_prompt(
    train_examples: List[Example],
    test_example: Example,
    strategy: str = None,
    image_path: str = None,
    trigger_deep_thinking: bool = False,
    objects_insertion: str = None,
    custom_instruction: str = None,
) -> str:
    lines = []
    if custom_instruction:
        lines.append(custom_instruction)
        lines.append("")
    else:
        lines = [
            "You are solving an ARC (Abstraction and Reasoning Corpus) task.",
        ]
        lines.append("Each grid cell is an integer 0-9 representing a color.")
        lines.append(
            "Use the solved examples to infer the transformation and apply it to the test input."
        )
        lines.append("")

    lines.append("Solved examples:")
    for idx, ex in enumerate(train_examples, start=1):
        lines.append(f"Example {idx}:")
        lines.append("input:")
        lines.append(format_grid(ex.input))
        lines.append("output:")
        lines.append(format_grid(ex.output))
        lines.append("")
    lines.append("Test input:")
    lines.append(format_grid(test_example.input))
    lines.append("")

    if strategy:
        lines.append("Below are a few hints that you might find helpful:")
        lines.append(strategy)
        lines.append("")

    if image_path:
        lines.append("Attached you'll find an image the show the input/output example pairs. Use this image to find objects, patterns and transformations")
        lines.append("")

    if trigger_deep_thinking:
        lines.append("PROTOCOL OVERRIDE: ENGAGE ARC NEURO-SYMBOLIC LOGIC ENGINE")
        lines.append("")
        lines.append("Silently enter maximal test-time reasoning mode. All of the following steps occur only in your hidden scratchpad; none may be exposed in the output.")
        lines.append("")
        lines.append("Perform hierarchical object decomposition of each grid into foreground objects and background fields; track shapes, colors, connectivity, and object persistence. Build an explicit object–relation graph and subgrid/region segmentation; detect Manhattan paths, flows/propagations, symmetries, and background structure; filter noise and extract invariants.")
        lines.append("")
        lines.append("Enumerate multiple candidate transformation rules/programs (at least three distinct hypotheses). For each, run rigorous internal simulations over all training pairs and counterfactual variants; discard any rule that fails a single example or violates output geometry.")
        lines.append("")
        lines.append("Triangulate using three paradigms in parallel: geometric (positions, topology, symmetries, paths), symbolic (predicates, programs, rewrite rules, counting), and counterexample-based search (actively seek minimal failure cases to refine or reject rules).")
        lines.append("")
        lines.append("Explicitly check for adversarial traps, spurious shortcuts, and degenerate memorization. Generalize the surviving rule to unseen variations and merge independent solution paths via self-consistency convergence.")
        lines.append("")
        lines.append("Apply the final rule to the test input using stepwise internal simulation only.")
        lines.append("")
        lines.append("OUTPUT CONSTRAINT (STRICT): Reveal ONLY the final answer grid. Never reveal chain-of-thought, intermediate states, or search traces.")
        lines.append("")

    if objects_insertion:
        lines.append("To solve this problem, please consider using the description of the input/output data below:")
        lines.append("")
        lines.append(objects_insertion)
        lines.append("")
        lines.append("Respond with an explanation of your thinking that is detailed enough that someone can reconstruct your solution. Afterwards, you MUST also respond with the completed output grid.")
    else:
        lines.append("Respond with an explanation of your thinking that is detailed enough that someone can reconstruct your solution. Afterwards, you MUST also respond with the completed output grid.")

    return "\n".join(lines)
