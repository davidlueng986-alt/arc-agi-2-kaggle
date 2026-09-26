from src.selection_legacy import pick_solution
from src.selection_advanced import pick_solution_v2

def is_solved(candidates_object) -> bool:
    if not candidates_object:
        return False

    total_model_runs = sum(group['count'] for group in candidates_object.values())
    if total_model_runs == 0:
        return False

    sorted_groups = sorted(candidates_object.values(), key=lambda g: g['count'], reverse=True)
    top_group = sorted_groups[0]
    max_count = top_group['count']
    
    percentage = (max_count / total_model_runs)
    
    # Condition 1: count > 25%
    # Condition 2: count >= 11
    if not (percentage > 0.25 and max_count >= 11):
        return False
        
    # Condition 3: all other groups have exactly count=1
    for group in sorted_groups[1:]:
        if group['count'] != 1:
            return False
            
    return True