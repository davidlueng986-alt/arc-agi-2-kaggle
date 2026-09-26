import sys
from concurrent.futures import ThreadPoolExecutor
from src.audit_prompts import build_logic_prompt, build_consistency_prompt, build_duo_pick_prompt
from src.judges import run_judge, run_duo_pick_judge

def pick_solution_v2(candidates_object, reasoning_store, task, test_index, openai_client, anthropic_client, google_keys, judge_model="gpt-5.2-xhigh", verbose: int = 0, openai_background: bool = False, judge_consistency_enable: bool = False, judge_duo_pick_enable: bool = True, total_attempts: int = 0):
    """
    Advanced solution picker using LLM Judges.
    - If judge_duo_pick_enable: Runs a "Council of 3 Duo Judges" to pick top solutions.
    - Fallback: Consensus (Vote Count) and Auditor Choice (Max Score).
    """
    if verbose >= 1:
        print("\n[pick_solution_v2] Starting Advanced Solution Picker")
    
    # 0. Preparation
    train_examples = task.train
    test_input = task.test[test_index-1].input if task.test and len(task.test) >= test_index else None
    
    # Flatten candidates for easy indexing
    candidates_list = []
    for idx, (grid_tuple, val) in enumerate(candidates_object.items()):
        candidates_list.append({
            "id": idx,
            "grid": val.get("grid"),
            "models": val.get("models"),
            "count": val.get("count"),
            "is_correct": val.get("is_correct"),
            "reasoning": {} 
        })
    
    if verbose >= 1:
        print(f"[pick_solution_v2] Total unique candidates found: {len(candidates_list)}")

    # Init Metadata
    selection_metadata = {
        "judges": {},
        "selection_process": {}
    }

    if not candidates_list and not judge_duo_pick_enable:
        print("[pick_solution_v2] No candidates found.")
        return [], False, selection_metadata

    # 1. Extract Reasoning
    for cand in candidates_list:
        for model_id in cand["models"]:
            if model_id in reasoning_store:
                cand["reasoning"][model_id] = reasoning_store[model_id]

    # 2. Council of Duo Judges
    if judge_duo_pick_enable:
        duo_prompt = build_duo_pick_prompt(train_examples, test_input, candidates_list, reasoning_store, total_attempts)
        
        council_results = []
        for i in range(3):
            council_results.append({ "run_index": i, "prompt": duo_prompt, "response": None, "picked_grids": None })

        # Run 3 judges in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = []
            for i in range(3):
                futures.append(executor.submit(
                    run_duo_pick_judge, 
                    duo_prompt, 
                    judge_model, 
                    openai_client, 
                    anthropic_client, 
                    google_keys, 
                    council_results[i], 
                    verbose, 
                    openai_background
                ))
            for f in futures:
                f.result() # Wait for all

        selection_metadata["judges"]["duo_pick_council"] = council_results

        # Scoring System
        scoreboard = {} # grid_tuple -> {points, grid, origin, source_runs}

        for run in council_results:
            grids = run.get("picked_grids")
            if not grids:
                continue
            
            # Points: 1st choice = 2 pts, 2nd choice = 1 pt
            for i, res_grid in enumerate(grids):
                points = 2 if i == 0 else 1
                res_tuple = tuple(tuple(row) for row in res_grid)
                
                if res_tuple not in scoreboard:
                    # Check if it matches an existing candidate
                    match_id = None
                    for cand in candidates_list:
                        if tuple(tuple(row) for row in cand['grid']) == res_tuple:
                            match_id = cand['id']
                            break
                    
                    scoreboard[res_tuple] = {
                        "points": 0,
                        "grid": res_grid,
                        "origin": "Existing Candidate" if match_id is not None else "Synthesized (New Grid)",
                        "matched_original_candidate_id": match_id,
                        "voted_by_judges": []
                    }
                
                scoreboard[res_tuple]["points"] += points
                scoreboard[res_tuple]["voted_by_judges"].append(run["run_index"])

        # Sort scoreboard by points descending
        sorted_scoreboard = sorted(scoreboard.items(), key=lambda x: x[1]["points"], reverse=True)
        
        # Log scoreboard for metadata
        selection_metadata["selection_process"]["scoreboard"] = [
            {
                "grid": v["grid"],
                "points": v["points"],
                "origin": v["origin"],
                "voted_by_judges": v["voted_by_judges"],
                "matched_original_candidate_id": v["matched_original_candidate_id"]
            } for k, v in sorted_scoreboard
        ]

        # Select Top 2 from Judges
        final_selection_groups = []
        for i in range(min(2, len(sorted_scoreboard))):
            grid_tuple, entry = sorted_scoreboard[i]
            
            if entry["matched_original_candidate_id"] is not None:
                # Use existing candidate metadata
                group = candidates_object[grid_tuple]
                
                feedback = f"\n\n--- COUNCIL OF JUDGES CHOICE (Score: {entry['points']}, Origin: {entry['origin']}) ---"
                # Add a bit of reasoning from the first run that voted for it
                first_voter_idx = entry["voted_by_judges"][0]
                feedback += council_results[first_voter_idx].get("response", "")[:500]
                
                existing_summary = group.get("reasoning_summary", "")
                group["reasoning_summary"] = feedback + "\n\n" + existing_summary
                final_selection_groups.append(group)
            else:
                # Create a new candidate for the judge's synthesized solution
                new_group = {
                    "grid": entry["grid"],
                    "count": 0,
                    "models": [f"duo_pick_council_synth_{i}"],
                    "is_correct": None,
                    "reasoning_summary": f"--- COUNCIL OF JUDGES SYNTHESIZED SOLUTION (Score: {entry['points']}) ---\n" + council_results[entry["voted_by_judges"][0]].get("response", "")
                }
                final_selection_groups.append(new_group)

        # Fallback to Voting if < 2 candidates
        if len(final_selection_groups) < 2:
            print(f"[pick_solution_v2] Judges only provided {len(final_selection_groups)} grids. Falling back to voting for remaining slots.", file=sys.stderr)
            selection_metadata["selection_process"]["fallback_triggered"] = True
            
            # Sort candidates by consensus count
            voted_candidates = sorted(candidates_list, key=lambda c: c['count'], reverse=True)
            
            for cand in voted_candidates:
                if len(final_selection_groups) >= 2:
                    break
                
                cand_tuple = tuple(tuple(row) for row in cand['grid'])
                # Avoid duplicates
                is_duplicate = False
                for existing in final_selection_groups:
                    if tuple(tuple(row) for row in existing['grid']) == cand_tuple:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    group = candidates_object[cand_tuple]
                    group["reasoning_summary"] = group.get("reasoning_summary", "") + "\n\n--- FALLBACK SELECTION (Consensus) ---"
                    final_selection_groups.append(group)

        selection_metadata["selection_process"]["type"] = "Council of Duo Judges"
        selection_metadata["selection_process"]["final_count"] = len(final_selection_groups)

        # Final Success Check
        is_solved_flag = any(g.get("is_correct") for g in final_selection_groups)
        return final_selection_groups, is_solved_flag, selection_metadata

    # 3. Standard Logic (Complete Fallback if Duo Pick Disabled)
    # Filter Candidates for Judging
    multi_vote_candidates = [c for c in candidates_list if c['count'] >= 2]
    if len(multi_vote_candidates) >= 2:
        candidates_for_judging = multi_vote_candidates
    else:
        candidates_for_judging = candidates_list

    # Build Prompts for Standard Judges
    full_prompt_logic = build_logic_prompt(train_examples, test_input, candidates_for_judging)
    full_prompt_cons = build_consistency_prompt(train_examples, test_input, candidates_for_judging)

    # Run Standard Judges
    scores = {c['id']: 0.0 for c in candidates_list}
    logic_data = { "prompt": full_prompt_logic, "response": None, "parsed": None }
    cons_data = { "prompt": full_prompt_cons, "response": None, "parsed": None }

    with ThreadPoolExecutor(max_workers=20) as executor:
        future_logic = executor.submit(run_judge, "Logic", full_prompt_logic, judge_model, openai_client, anthropic_client, google_keys, logic_data, verbose, openai_background)
        future_cons = None
        if judge_consistency_enable:
            future_cons = executor.submit(run_judge, "Consistency", full_prompt_cons, judge_model, openai_client, anthropic_client, google_keys, cons_data, verbose, openai_background)
        logic_res = future_logic.result()
        cons_res = future_cons.result() if future_cons else None

    # Update Scores
    if logic_res and "candidates" in logic_res:
        for c in logic_res["candidates"]:
            cid = c.get("candidate_id")
            if cid in scores: scores[cid] = max(scores[cid], c.get("score", 0))
    if cons_res and "candidates" in cons_res:
        for c in cons_res["candidates"]:
            cid = c.get("candidate_id")
            if cid in scores: scores[cid] = max(scores[cid], c.get("score", 0))

    # Determine Attempts
    candidates_by_consensus = sorted(candidates_list, key=lambda c: (c['count'], scores[c['id']]), reverse=True)
    attempt_1_candidate = candidates_by_consensus[0]
    
    sorted_by_score = sorted(candidates_list, key=lambda c: scores[c['id']], reverse=True)
    attempt_2_candidate = next((c for c in sorted_by_score if c['id'] != attempt_1_candidate['id']), None)
            
    final_selection = [attempt_1_candidate]
    if attempt_2_candidate: final_selection.append(attempt_2_candidate)
    
    # Populate Metadata
    selection_metadata["judges"]["logic"] = logic_data
    selection_metadata["judges"]["consistency"] = cons_data
    selection_metadata["selection_process"] = {
        "type": "Standard (Consensus/Auditor)",
        "attempt_1": {"candidate_id": attempt_1_candidate['id'], "votes": attempt_1_candidate['count']},
        "attempt_2": {"candidate_id": attempt_2_candidate['id'] if attempt_2_candidate else None}
    }

    # Prepare Judge Feedback Map
    judge_feedback_map = {}
    if logic_res and "candidates" in logic_res:
        for c in logic_res["candidates"]:
            cid = c.get("candidate_id")
            feedback = []
            if "rule_summary" in c: feedback.append(f"Judge Rule Summary: {c['rule_summary']}")
            if "example_audit" in c and "summary" in c["example_audit"]:
                feedback.append(f"Judge Audit Summary: {c['example_audit']['summary']}")
            if feedback: judge_feedback_map[cid] = "\n\n--- JUDGE FEEDBACK ---" + "\n".join(feedback)

    # Construct Return Output
    top_groups = []
    for cand in final_selection:
        grid_tuple = tuple(tuple(row) for row in cand['grid'])
        group = candidates_object[grid_tuple]
        
        final_summary_parts = []
        if cand['id'] in judge_feedback_map:
            final_summary_parts.append(judge_feedback_map[cand['id']])
        if cand["reasoning"]:
            first_model_id = next(iter(cand["reasoning"]))
            raw_reasoning = cand["reasoning"][first_model_id]
            final_summary_parts.append("\n\n--- EXAMPLE REASONING ---" + raw_reasoning)
        
        group["reasoning_summary"] = "".join(final_summary_parts).strip()
        top_groups.append(group)

    return top_groups, any(g.get("is_correct") for g in top_groups), selection_metadata