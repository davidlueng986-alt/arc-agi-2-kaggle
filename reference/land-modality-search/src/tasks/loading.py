import json
from pathlib import Path
from typing import List, Union, Dict
from src.types import Example, Task

def load_task(json_source: Union[Path, Dict], answer_path: Path = None) -> Task:
    if isinstance(json_source, dict):
        data = json_source
    else:
        data = json.loads(json_source.read_text())

    def to_examples(items: List[dict], is_test: bool = False) -> List[Example]:
        examples = []
        for i, ex in enumerate(items):
            input_grid = ex["input"]
            output_grid = ex.get("output")
            examples.append(Example(input=input_grid, output=output_grid))
        return examples

    train_examples = to_examples(data["train"])
    test_examples = to_examples(data["test"], is_test=True)

    if answer_path and answer_path.exists():
        try:
            answer_data = json.loads(answer_path.read_text())
            if "test" in answer_data:
                answer_tests = answer_data["test"]
                for i, ex in enumerate(test_examples):
                    if i < len(answer_tests) and ex.output is None:
                        ex.output = answer_tests[i].get("output")
        except Exception as e:
            print(f"Warning: Failed to load answers from {answer_path}: {e}")

    return Task(train=train_examples, test=test_examples)

def load_task_paths(list_path: Path) -> List[Path]:
    payload = json.loads(list_path.read_text())
    if not isinstance(payload, dict) or "tasks" not in payload:
        raise ValueError(f"Task list in {list_path} must be a dict with a 'tasks' key.")
    
    tasks = payload["tasks"]
    if not isinstance(tasks, list):
        raise ValueError(f"The 'tasks' value in {list_path} must be a list.")

    result = []
    for item in tasks:
        if not isinstance(item, str):
            raise ValueError("Each task entry must be a string path.")
        result.append(Path(item))
    return result
