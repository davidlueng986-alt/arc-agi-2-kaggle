from pathlib import Path

def find_task_path(task_id: str) -> Path:
    if task_id.endswith(".json"):
        p = Path(task_id)
        if p.exists():
            return p
        task_id = p.stem
    candidate = Path("data/arc-agi-2-evaluation") / f"{task_id}.json"
    if candidate.exists():
        return candidate
    candidate_training = Path("data/arc-agi-2-training") / f"{task_id}.json"
    if candidate_training.exists():
        return candidate_training
    raise FileNotFoundError(f"Task file for '{task_id}' not found in data/arc-agi-2-evaluation/ or data/arc-agi-2-training/.")