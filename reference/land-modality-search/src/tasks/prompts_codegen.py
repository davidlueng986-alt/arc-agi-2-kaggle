from typing import List
from src.types import Example

# Import implementation from submodules to maintain backward compatibility
from src.tasks.codegen_prompts.v1 import build_prompt_codegen_v1, build_prompt_codegen_v1b
from src.tasks.codegen_prompts.v2 import build_prompt_codegen_v2, build_prompt_codegen_v2b
from src.tasks.codegen_prompts.v3 import build_prompt_codegen_v3_stage1, build_prompt_codegen_v3_stage2
from src.tasks.codegen_prompts.v4 import build_prompt_codegen_v4

def build_prompt_codegen(train_examples: List[Example], test_examples: List[Example] = None, version: str = "v2", model_name: str = None) -> str:
    if version == "v1":
        return build_prompt_codegen_v1(train_examples)
    elif version == "v1b":
        if test_examples is None:
             raise ValueError("V1B prompt requires test_examples")
        return build_prompt_codegen_v1b(train_examples, test_examples, model_name=model_name)
    elif version == "v2b":
        if test_examples is None:
             raise ValueError("V2B prompt requires test_examples")
        return build_prompt_codegen_v2b(train_examples, test_examples)
    elif version == "v4":
        if test_examples is None:
             raise ValueError("V4 prompt requires test_examples")
        return build_prompt_codegen_v4(train_examples, test_examples, model_name=model_name)
    elif version == "v3":
        if test_examples is None:
             raise ValueError("V3 prompt requires test_examples")
        return build_prompt_codegen_v3_stage1(train_examples, test_examples)
    return build_prompt_codegen_v2(train_examples)