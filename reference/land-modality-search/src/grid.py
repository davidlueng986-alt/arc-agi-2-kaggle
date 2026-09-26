from typing import List, Optional
import re

Grid = List[List[int]]

def format_grid(grid: Grid) -> str:
    """Formats a grid as CSV."""
    if grid is None:
        return ""
    return "\n".join(",".join(str(c) for c in row) for row in grid)

def parse_grid_from_text(text: str) -> Grid:
    """Parses a CSV grid from text, handling noise and labels.
    
    Strategies:
    1.  Identify all 'candidate rows' (lines containing only comma-separated numbers).
    2.  Group consecutive rows into 'blocks'.
    3.  Treat markdown code fences (```) as HARD separators that break blocks.
    4.  Allow small gaps (blank lines/text) within a block ONLY if no hard separators are encountered.
    5.  Return the LAST valid block found, assuming it is the final answer.
    """
    text = text.strip()
    lines = text.splitlines()
    
    candidate_rows = []
    hard_separators = [] # Track indices of lines that are hard separators
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Hard Separator Detection
        if stripped.startswith("```"):
            candidate_rows.append(None)
            hard_separators.append(i)
            continue
            
        if not stripped: 
            candidate_rows.append(None)
            continue

        # Ignore explicit row labels which confuse the parser (e.g. "Row 1:", "Row 10")
        # if they stand alone on a line.
        if re.match(r'^Row\s+\d+:?$', stripped, re.IGNORECASE):
            candidate_rows.append(None)
            continue

        # Ignore markdown list items (bullet points) as they are usually descriptions, not raw data
        if stripped.startswith(("-", "*", "+")):
            candidate_rows.append(None)
            continue
            
        row = None
        # CSV parsing
        try:
            # Pre-clean: remove ` [ ] and spaces to handle conversational formatting
            # Replace with space to prevent merging (e.g. "10.`8" -> "10. 8")
            clean_line = stripped.replace("`", " ").replace("[", " ").replace("]", " ").strip()
            
            # Handle numbered lists (e.g. "1. 8,8,8" or "1) 8,8,8")
            numbered_list_match = re.match(r'^\d+[\.\)]\s+', clean_line)
            if numbered_list_match:
                clean_line = clean_line[numbered_list_match.end():]

            tokens = clean_line.split(",")
            # Allow for spaces around numbers just in case
            if len(tokens) > 0 and all(t.strip().isdigit() for t in tokens):
                row = [int(t.strip()) for t in tokens]
            else:
                # Fallback: Parsing lines like "Row 1: 8,8,8..." or "Output: 1,2,3"
                # 1. Try splitting by colon and taking the last part
                if ":" in clean_line:
                    clean_line = clean_line.split(":")[-1].strip()
                
                # 2. Fallback: Try to find the first digit and parse from there
                match = re.search(r'\d', clean_line)
                if match:
                    sub = clean_line[match.start():]
                    
                    # Simple approach: Slice from first digit to last digit
                    last_digit_idx = -1
                    for idx, char in enumerate(clean_line):
                        if char.isdigit():
                            last_digit_idx = idx
                    
                    if last_digit_idx != -1 and last_digit_idx >= match.start():
                        candidate_sub = clean_line[match.start() : last_digit_idx + 1]
                        sub_tokens = candidate_sub.split(",")
                        if len(sub_tokens) > 1 and all(t.strip().isdigit() for t in sub_tokens):
                             # Ensure the rest of the line doesn't contain alphabetic chars (noise)
                             remainder = clean_line[last_digit_idx + 1:].strip()
                             if not any(c.isalpha() for c in remainder):
                                 row = [int(t.strip()) for t in sub_tokens]

        except ValueError:
            pass

        candidate_rows.append(row)

    # Block reconstruction
    blocks = []
    current_block = []
    
    # State tracking for the current block
    current_block_start_index = -1
    last_row_index = -1
    
    MAX_GAP = 2 # Allow a small gap of text/newlines within a grid (e.g. noise) 
    
    for i, row in enumerate(candidate_rows):
        if row is not None:
            # We have a grid row
            if not current_block:
                # Start new block
                current_block = [row]
                current_block_start_index = i
                last_row_index = i
            else:
                # Existing block. Check compatibility.
                
                # 1. Check for Hard Separators between last row and this row
                has_hard_sep = any(last_row_index < sep_idx < i for sep_idx in hard_separators)
                
                # 2. Check for Gap Size
                gap_size = i - last_row_index - 1
                
                # 3. Check Width Compatibility
                # Allow significant variation (up to 5 columns) to handle model typos (extra commas, ragged rows).
                # We rely on gaps/separators to distinguish distinct grids.
                width_diff = abs(len(row) - len(current_block[0]))
                width_match = width_diff <= 5
                
                if has_hard_sep or gap_size > MAX_GAP or not width_match:
                    # Check for Exact Duplication before finalizing
                    # If the NEW block we are about to start is identical to the CURRENT block,
                    # we might be seeing a repetition.
                    # However, we can't see the future rows yet.
                    # Instead, let's look at the gap logic:
                    # If we are here, we are SPLITTING.
                    # So blocks.append(current_block) happens.
                    blocks.append(current_block)
                    current_block = [row]
                    current_block_start_index = i
                    last_row_index = i
                else:
                    # Extend current block
                    current_block.append(row)
                    last_row_index = i                    
    if current_block: blocks.append(current_block)
    
    if not blocks:
        raise ValueError("Could not parse grid")
    
    # Return the last block found
    return blocks[-1]

def verify_prediction(predicted: Grid, expected: Optional[Grid]) -> Optional[bool]:
    if expected is None:
        return None
    return predicted == expected

def grid_to_string(grid: Optional[Grid]) -> str:
    """Formats grid for Prompt Logic (visual style)."""
    if not grid:
        return "(Empty Grid)"
    
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    
    lines = [f"Size: {rows}x{cols}"]
    for row in grid:
        lines.append("".join(str(c) for c in row))
    return "\n".join(lines)

def grid_to_csv_rows(grid: Optional[Grid], padding: str = "      ") -> str:
    """Formats grid for Prompt Consistency (comma-separated rows with padding)."""
    if not grid:
        return ""
    lines = []
    for row in grid:
        lines.append(padding + ",".join(map(str, row)))
    return "\n".join(lines)
