def pick_solution(candidates_object, verbose: int = 0):
    # Model priority mapping (higher number = higher priority)
    MODEL_PRIORITY = {
        "claude-opus-4.5-thinking-60000": 4,
        "gemini-3-high": 3,
        "gpt-5.1-medium": 2,
        "claude-sonnet-4.5-thinking-60000": 1
    }

    def get_group_priority(group):
        max_priority = 0
        for run_id in group['models']:
            # run_id format is typically "model-name_count_step"
            # We try to match the longest possible prefix in our priority map
            for model_name, priority in MODEL_PRIORITY.items():
                if run_id.startswith(model_name):
                    if priority > max_priority:
                        max_priority = priority
        return max_priority

    # Sort by count (descending) and then by priority (descending)
    sorted_groups = sorted(
        candidates_object.values(), 
        key=lambda g: (g['count'], get_group_priority(g)), 
        reverse=True
    )
    
    if verbose >= 1:
        print("\n" + "="*40)
        print("FINAL OUTCOME")
        print("="*40)
    
    is_solved_flag = False
    unknown_status = False
    top_groups = sorted_groups[:2]
    
    if len(top_groups) > 0:
        if top_groups[0].get("is_correct") is None:
            unknown_status = True
        elif top_groups[0]["is_correct"]:
            is_solved_flag = True
        elif len(top_groups) > 1 and top_groups[1]["is_correct"]:
            is_solved_flag = True
        
    if unknown_status:
        if verbose >= 1:
            print("Outcome: SUBMITTED (No Ground Truth)")
    elif is_solved_flag:
        if verbose >= 1:
            print("Outcome: SOLVED")
    else:
        if verbose >= 1:
            print("Outcome: FAILED")

    if verbose >= 1:
        print("\n--- Debug Info ---")
        if not top_groups:
            print("No solutions generated.")
        else:
            for i, group in enumerate(top_groups):
                correctness = group.get('is_correct')
                c_str = "Unknown" if correctness is None else str(correctness)
                priority = get_group_priority(group)
                print(f"Group {i+1}: Count={group['count']}, Priority={priority}, Correct={c_str}")
                print(f"  Models: {', '.join(group['models'])}")

    # Check for other correct groups
    other_correct = []
    for i, group in enumerate(sorted_groups):
        if i < 2:
            continue
        if group.get('is_correct') is True:
            other_correct.append((i + 1, group))
            
    if other_correct and verbose >= 1:
        print("\n--- Other Correct Groups ---")
        for rank, group in other_correct:
            print(f"Group {rank}: Count={group['count']}, Correct={group['is_correct']}")
            print(f"  Models: {', '.join(group['models'])}")
            
    return top_groups, is_solved_flag, {}