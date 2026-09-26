import json
import re
import time
import sys
from src.models import call_model, calculate_cost, parse_model_arg

def extract_json(text):
    """
    Robustly extract a JSON object from text.
    Prioritizes objects containing 'candidates' key.
    """
    if not text:
        return None
    text = text.strip()
    
    # 1. Try to find JSON block within markdown fences
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(1))
            if isinstance(obj, dict) and "candidates" in obj:
                return obj
        except json.JSONDecodeError:
            pass

    # 2. Scan for any '{' and try to decode a valid JSON object
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        start_idx = match.start()
        try:
            # raw_decode parsing stops at the end of the valid object
            obj, _ = decoder.raw_decode(text, idx=start_idx)
            if isinstance(obj, dict) and "candidates" in obj:
                return obj
        except json.JSONDecodeError:
            continue
            
    return None

def run_judge(judge_name, prompt, judge_model, openai_client, anthropic_client, google_keys, result_container, verbose: int = 0, use_background: bool = False):
    timings = []
    try:
        start_ts = time.perf_counter()
        response_obj = call_model(openai_client, anthropic_client, google_keys, prompt, judge_model, use_background=use_background, timing_tracker=timings)
        duration = time.perf_counter() - start_ts
    
        
        result_container["response"] = response_obj.text
        
        actual_model = getattr(response_obj, "model_name", None) or judge_model
        
        # Calculate cost
        cost = 0.0
        try:
            model_to_price = actual_model if actual_model != judge_model else judge_model
            model_config = parse_model_arg(model_to_price)
            cost = calculate_cost(model_config, response_obj)
        except Exception:
            pass
        
        # Capture metrics
        result_container["model"] = actual_model
        result_container["requested_model"] = judge_model
        result_container["actual_model"] = actual_model
        result_container["duration_seconds"] = round(duration, 2)
        result_container["total_cost"] = cost
        result_container["input_tokens"] = response_obj.prompt_tokens
        result_container["output_tokens"] = response_obj.completion_tokens
        result_container["cached_tokens"] = response_obj.cached_tokens
        result_container["timing_breakdown"] = timings
        
        parsed_json = extract_json(response_obj.text)
        
        if parsed_json:
            result_container["parsed"] = parsed_json
            return parsed_json
        else:
            print(f"[pick_solution_v2] {judge_name} Judge: Could not parse JSON. Response start: {response_obj.text[:500]}")
            
    except Exception as e:
        print(f"[pick_solution_v2] {judge_name} Judge Error: {e}")
        result_container["error"] = str(e)
    return None

def extract_all_grids(text):
    """
    Robustly extract all CSV-like grid blocks from text.
    Replicates logic from src.grid.parse_grid_from_text but returns ALL blocks.
    """
    if not text:
        return []
        
    text = text.strip()
    lines = text.splitlines()
    
    candidate_rows = []
    hard_separators = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("```"):
            candidate_rows.append(None)
            hard_separators.append(i)
            continue
        if not stripped:
            candidate_rows.append(None)
            continue
        if re.match(r'^Row\s+\d+:?$', stripped, re.IGNORECASE):
            candidate_rows.append(None)
            continue
        if stripped.startswith(("-", "*", "+")):
            candidate_rows.append(None)
            continue
            
        row = None
        try:
            clean_line = stripped.replace("`", " ").replace("[", " ").replace("]", " ").strip()
            numbered_list_match = re.match(r'^\d+[\.\)]\s+', clean_line)
            if numbered_list_match:
                clean_line = clean_line[numbered_list_match.end():]

            tokens = clean_line.split(",")
            if len(tokens) > 0 and all(t.strip().isdigit() for t in tokens):
                row = [int(t.strip()) for t in tokens]
            else:
                if ":" in clean_line:
                    clean_line = clean_line.split(":")[-1].strip()
                match = re.search(r'\d', clean_line)
                if match:
                    last_digit_idx = -1
                    for idx, char in enumerate(clean_line):
                        if char.isdigit():
                            last_digit_idx = idx
                    if last_digit_idx != -1 and last_digit_idx >= match.start():
                        candidate_sub = clean_line[match.start() : last_digit_idx + 1]
                        sub_tokens = candidate_sub.split(",")
                        if len(sub_tokens) > 1 and all(t.strip().isdigit() for t in sub_tokens):
                             remainder = clean_line[last_digit_idx + 1:].strip()
                             if not any(c.isalpha() for c in remainder):
                                 row = [int(t.strip()) for t in sub_tokens]
        except ValueError:
            pass
        candidate_rows.append(row)

    blocks = []
    current_block = []
    last_row_index = -1
    MAX_GAP = 2 
    
    for i, row in enumerate(candidate_rows):
        if row is not None:
            if not current_block:
                current_block = [row]
                last_row_index = i
            else:
                has_hard_sep = any(last_row_index < sep_idx < i for sep_idx in hard_separators)
                gap_size = i - last_row_index - 1
                width_diff = abs(len(row) - len(current_block[0]))
                width_match = width_diff <= 5
                
                if has_hard_sep or gap_size > MAX_GAP or not width_match:
                    blocks.append(current_block)
                    current_block = [row]
                    last_row_index = i
                else:
                    current_block.append(row)
                    last_row_index = i                    
    if current_block: blocks.append(current_block)
    return blocks

def run_duo_pick_judge(prompt, judge_model, openai_client, anthropic_client, google_keys, result_container, verbose: int = 0, use_background: bool = False):
    timings = []
    try:
        start_ts = time.perf_counter()
        response_obj = call_model(openai_client, anthropic_client, google_keys, prompt, judge_model, use_background=use_background, timing_tracker=timings)
        duration = time.perf_counter() - start_ts
        
        result_container["response"] = response_obj.text
        actual_model = getattr(response_obj, "model_name", None) or judge_model
        
        # Calculate cost
        cost = 0.0
        try:
            model_to_price = actual_model if actual_model != judge_model else judge_model
            model_config = parse_model_arg(model_to_price)
            cost = calculate_cost(model_config, response_obj)
        except Exception:
            pass
            
        result_container["model"] = actual_model
        result_container["duration_seconds"] = round(duration, 2)
        result_container["total_cost"] = cost
        result_container["input_tokens"] = response_obj.prompt_tokens
        result_container["output_tokens"] = response_obj.completion_tokens
        
        grids = extract_all_grids(response_obj.text)
        
        # We need unique grids for the final picks
        unique_grids = []
        for g in grids:
            if g not in unique_grids:
                unique_grids.append(g)
        
        if len(unique_grids) >= 2:
            # Take the last two distinct grids
            result_container["picked_grids"] = unique_grids[-2:]
            return unique_grids[-2:]
        elif len(unique_grids) == 1:
            result_container["picked_grids"] = unique_grids
            return unique_grids
            
    except Exception as e:
        print(f"[pick_solution_v2] Duo Pick Judge Error: {e}")
        result_container["error"] = str(e)
    return None