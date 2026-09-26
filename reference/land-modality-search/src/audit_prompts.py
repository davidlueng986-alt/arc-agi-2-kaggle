from src.grid import grid_to_string, grid_to_csv_rows, format_grid
import re
from src.audit_templates_logic import (
    PROMPT_LOGIC_SYSTEM_ROLE,
    PROMPT_LOGIC_INSTRUCTIONS
)
from src.audit_templates_consistency import (
    PROMPT_CONSISTENCY_SYSTEM_ROLE,
    PROMPT_CONSISTENCY_TASK_CONTEXT,
    PROMPT_CONSISTENCY_INSTRUCTIONS_FORMAT,
    PROMPT_CONSISTENCY_OUTPUT_FORMAT
)

def build_duo_pick_prompt(train_examples, test_input, candidates_list, reasoning_store, total_attempts):
    """
    Constructs the prompt for the "Duo Pick Judge" (Meta-Conclusion).
    """
    parts = []
    
    # 1. Header
    parts.append(f"Below is a prompt that was run {total_attempts} times:")
    
    # 2. Base Prompt
    # We build a standard prompt (non-codegen) to provide context
    from src.tasks import build_prompt
    from types import SimpleNamespace
    
    test_example_wrapper = SimpleNamespace(input=test_input) if test_input is not None else None
    base_prompt_text = build_prompt(train_examples, test_example_wrapper)
    parts.append("\n<PROMPT START>")
    parts.append(base_prompt_text)
    parts.append("<PROMPT STOP>\n")
    
    # 3. Solution Introduction
    total_solutions = sum(len(cand['models']) for cand in candidates_list)
    parts.append(f"Solutions were generated {total_solutions} times, using different types of solvers. All solutions are represented below:\n")
    
    # 4. The Solutions
    solution_index = 1
    for cand in candidates_list:
        grid_csv = format_grid(cand['grid'])
        
        for model_id in cand['models']:
            parts.append(f"<SOLUTION {solution_index} START>")
            
            raw_response = reasoning_store.get(model_id, "(Reasoning not found)")
            
            # Identify if it was codegen by checking for 'solver'
            content = raw_response
            if "def solver" in raw_response:
                # Try to extract just the solver function
                match = re.search(r"(def solver\(.*?\):.*?\n\s+return\s+.*)", raw_response, re.DOTALL)
                if match:
                    content = match.group(1)
            
            parts.append("<CONTENT>")
            parts.append(content)
            parts.append("</CONTENT>")
            
            parts.append("<PREDICTED_GRID>")
            parts.append(grid_csv)
            parts.append("</PREDICTED_GRID>")
            
            parts.append(f"<SOLUTION {solution_index} STOP>\n")
            solution_index += 1
            
    # 5. Closing Instructions
    closing = [
        "Your task is to understand these solutions, and assess how well they've understood the problem, and how likely their solutions are to provide the correct solution to the test input.",
        "Often, new mechanics are introduced in the test example for which the solutions do not generalize well. Please output two solutions that you think represent the right mechanic for solving the problem.",
        "Output your two solutions as grids (in code blocks). Format the grids as comma-separated values (CSV) with each row on a new line, like this:",
        "```",
        "7,0,0,7",
        "0,7,7,0",
        "```",
        "Explain how you came to these two solutions being the two most likely. In coming up with your two solutions, study all the provided solutions and their reasoning to come up with a meta-conclusion about how to solve the problem."
    ]
    parts.append("\n".join(closing))
    
    return "\n".join(parts)

def build_logic_prompt(train_examples, test_input, candidates_list):
    logic_parts = []
    logic_parts.append(PROMPT_LOGIC_SYSTEM_ROLE)
    logic_parts.append("\n<INPUT_DATA>")
    
    logic_parts.append("1. {SOLVED_EXAMPLES}:")
    for i, example in enumerate(train_examples):
        logic_parts.append(f"<EXAMPLE_{i+1}>")
        logic_parts.append("<INPUT>")
        logic_parts.append(grid_to_string(example.input))
        logic_parts.append("</INPUT>")
        logic_parts.append("<OUTPUT>")
        logic_parts.append(grid_to_string(example.output))
        logic_parts.append("</OUTPUT>")
        logic_parts.append(f"</EXAMPLE_{i+1}>")
    
    logic_parts.append("\n2. {TEST_INPUT}:")
    if test_input:
        logic_parts.append(grid_to_string(test_input))
    else:
        logic_parts.append("(No Test Input)")

    logic_parts.append("\n3. {CANDIDATES}:")
    for cand in candidates_list:
        c_id = cand['id']
        logic_parts.append(f"<CANDIDATE {c_id}>")
        logic_parts.append("<PROPOSED_SOLUTION>")
        logic_parts.append(grid_to_string(cand['grid']))
        logic_parts.append("</PROPOSED_SOLUTION>")
        for j, model_id in enumerate(cand['models']):
            alias = chr(65 + j)
            logic_parts.append(f'<REASONING_MODEL_{alias} model_id="{model_id}">')
            reasoning = cand['reasoning'].get(model_id, "(Reasoning not found)")
            logic_parts.append(reasoning)
            logic_parts.append(f"</REASONING_MODEL_{alias}>")
        logic_parts.append(f"</CANDIDATE {c_id}>")

    logic_parts.append("</INPUT_DATA>\n")
    logic_parts.append(PROMPT_LOGIC_INSTRUCTIONS)
    return "\n".join(logic_parts)

def build_consistency_prompt(train_examples, test_input, candidates_list):
    cons_parts = []
    cons_parts.append(PROMPT_CONSISTENCY_SYSTEM_ROLE)
    cons_parts.append(PROMPT_CONSISTENCY_TASK_CONTEXT)
    cons_parts.append("\n<PROBLEM>")
    
    for i, ex in enumerate(train_examples):
        cons_parts.append(f'  <TRAIN_EXAMPLE index="{i+1}">')
        cons_parts.append("    <INPUT_GRID>")
        cons_parts.append(grid_to_csv_rows(ex.input))
        cons_parts.append("    </INPUT_GRID>")
        cons_parts.append("    <OUTPUT_GRID>")
        cons_parts.append(grid_to_csv_rows(ex.output))
        cons_parts.append("    </OUTPUT_GRID>")
        cons_parts.append("  </TRAIN_EXAMPLE>")
        
    if test_input:
        cons_parts.append("  <TEST_INPUT>")
        cons_parts.append("    <INPUT_GRID>")
        cons_parts.append(grid_to_csv_rows(test_input))
        cons_parts.append("    </INPUT_GRID>")
        cons_parts.append("  </TEST_INPUT>")
        
    cons_parts.append("</PROBLEM>\n")
    
    cons_parts.append("<CANDIDATES>")
    for cand in candidates_list:
        c_id = cand['id']
        cons_parts.append(f'  <CANDIDATE id="{c_id}">')
        for j, model_id in enumerate(cand['models']):
            alias = chr(65 + j)
            cons_parts.append(f'    <ANSWER id="{alias}" model_id="{model_id}">')
            cons_parts.append(f'      <EXPLANATION>')
            reasoning = cand['reasoning'].get(model_id, "(Reasoning not found)")
            cons_parts.append(reasoning)
            cons_parts.append(f'      </EXPLANATION>')
            cons_parts.append(f'      <OUTPUT_GRID>')
            cons_parts.append(grid_to_csv_rows(cand['grid']))
            cons_parts.append(f'      </OUTPUT_GRID>')
            cons_parts.append(f'    </ANSWER>')
        cons_parts.append(f'  </CANDIDATE>')
    cons_parts.append("</CANDIDATES>\n")
    
    cons_parts.append(PROMPT_CONSISTENCY_INSTRUCTIONS_FORMAT)
    cons_parts.append(PROMPT_CONSISTENCY_OUTPUT_FORMAT)
    return "\n".join(cons_parts)