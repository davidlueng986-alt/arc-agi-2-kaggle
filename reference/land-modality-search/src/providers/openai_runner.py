import base64
import mimetypes
import sys
from typing import Optional, List, Dict, Any

from openai import OpenAI
from anthropic import Anthropic

from src.types import ModelConfig, ModelResponse
from src.llm_utils import run_with_retry, orchestrate_two_stage
from src.logging import get_logger
from src.providers.openai_utils import _map_openai_exception
from src.providers.openai_background import OpenAIBackgroundSolver

logger = get_logger("providers.openai")

class OpenAIRequestRunner:
    """Helper class to encapsulate context and logic for OpenAI requests."""
    
    def __init__(
        self,
        client: OpenAI,
        config: ModelConfig,
        anthropic_client: Optional[Anthropic] = None,
        task_id: Optional[str] = None,
        test_index: Optional[int] = None,
        step_name: Optional[str] = None,
        run_timestamp: Optional[str] = None,
        model_alias: Optional[str] = None,
        timing_tracker: Optional[List[Dict]] = None,
        verbose: bool = False,
    ):
        self.client = client
        self.config = config
        self.anthropic_client = anthropic_client
        self.task_id = task_id
        self.test_index = test_index
        self.step_name = step_name
        self.run_timestamp = run_timestamp
        self.model_alias = model_alias
        self.timing_tracker = timing_tracker
        self.verbose = verbose

        self.model = config.base_model
        self.reasoning_effort = str(config.config)
        self.full_model_name = model_alias if model_alias else f"{self.model}-{self.reasoning_effort}"
        
        self.last_failed_job_id = None
        self.is_downgraded_retry = False
        
        # Instantiate the background solver strategy
        self.background_solver = OpenAIBackgroundSolver(self)

    def _prepare_content(self, prompt: str, image_path: Optional[str] = None) -> List[Dict[str, Any]]:
        content = [{"type": "input_text", "text": prompt}]
        if image_path:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            mime_type, _ = mimetypes.guess_type(image_path)
            if mime_type is None:
                mime_type = 'application/octet-stream'
            content.append({
                "type": "input_image",
                "image_url": f"data:{mime_type};base64,{base64_image}",
            })
        return content

    def solve_stream(self, prompt: str, image_path: Optional[str] = None, enable_code_execution: bool = False) -> ModelResponse:
        content = self._prepare_content(prompt, image_path)

        kwargs = {
            "model": self.model,
            "input": [{"role": "user", "content": content}],
            "timeout": 3300,
            "stream": True,
        }
        if self.reasoning_effort != "none":
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        
        if enable_code_execution:
            print("\n\n!!! WARNING: CODE INTERPRETER REQUESTED BUT FORCE-DISABLED !!!\n", file=sys.stderr)
            # kwargs["tools"] = [{
            #     "type": "code_interpreter",
            #     "container": {"type": "auto"}
            # }]
            # kwargs["tool_choice"] = "auto"
            # kwargs["max_tool_calls"] = 100
            # kwargs["include"] = ["code_interpreter_call.outputs"]

        def _call_and_accumulate():
            try:
                stream = self.client.responses.create(**kwargs)
                collected_content = []
                usage_data = None
                response_id = None
                
                # Detailed logging accumulators
                detailed_logs = []
                current_code_block = ""
                
                for chunk in stream:
                    chunk_type = getattr(chunk, "type", "")
                    
                    if chunk_type == "response.created":
                        if hasattr(chunk, "response") and hasattr(chunk.response, "id"):
                            response_id = chunk.response.id

                    elif chunk_type == "response.output_text.delta":
                        if hasattr(chunk, "delta") and chunk.delta:
                            text_delta = chunk.delta
                            collected_content.append(text_delta)
                            # Append to logs. If the last log was text, append to it to keep it clean?
                            # For simplicity/stream-likeness, we can append chunks or aggregate later.
                            # Let's aggregate continuously if type matches.
                            if detailed_logs and detailed_logs[-1]["type"] == "text":
                                detailed_logs[-1]["content"] += text_delta
                            else:
                                detailed_logs.append({"type": "text", "content": text_delta})
                    
                    elif chunk_type == "response.reasoning_text.delta":
                        if hasattr(chunk, "delta") and chunk.delta:
                            thought_delta = chunk.delta
                            if detailed_logs and detailed_logs[-1]["type"] == "thought":
                                detailed_logs[-1]["content"] += thought_delta
                            else:
                                detailed_logs.append({"type": "thought", "content": thought_delta})
                    
                    elif chunk_type == "response.code_interpreter_call.delta":
                        # Capturing code generation
                        if hasattr(chunk, "delta") and hasattr(chunk.delta, "code_interpreter_call") and hasattr(chunk.delta.code_interpreter_call, "input"):
                            code_delta = chunk.delta.code_interpreter_call.input
                            if code_delta:
                                if detailed_logs and detailed_logs[-1]["type"] == "code":
                                    detailed_logs[-1]["code"] += code_delta
                                else:
                                    detailed_logs.append({"type": "code", "code": code_delta, "language": "python"})

                    elif chunk_type == "response.code_interpreter_call.output":
                         # Capturing execution output
                         if hasattr(chunk, "output") and hasattr(chunk.output, "content"):
                             # There might be multiple content parts (logs, images)
                             for content_item in chunk.output.content:
                                 if content_item.type == "logs":
                                     detailed_logs.append({
                                         "type": "execution_result", 
                                         "output": content_item.logs, 
                                         "outcome": "completed"
                                     })
                                 elif content_item.type == "image":
                                     detailed_logs.append({
                                         "type": "execution_result",
                                         "output": "<image_data>",
                                         "outcome": "image_generated"
                                     })

                    if chunk_type == "response.completed":
                        if hasattr(chunk, "response") and hasattr(chunk.response, "usage"):
                            usage_data = chunk.response.usage

                return {
                    "text": "".join(collected_content),
                    "usage": usage_data,
                    "id": response_id,
                    "detailed_logs": detailed_logs
                }

            except Exception as e:
                _map_openai_exception(e, self.full_model_name)

        result = run_with_retry(
            lambda: _call_and_accumulate(),
            task_id=self.task_id,
            test_index=self.test_index,
            run_timestamp=self.run_timestamp,
            model_name=self.full_model_name,
            timing_tracker=self.timing_tracker
        )

        text_output = result["text"]
        if not text_output:
            # Fallback: if we have code logs but no text, maybe the model just ran code?
            # But the caller expects text.
            if not result.get("detailed_logs"):
                 raise RuntimeError(f"OpenAI Responses API Step 1 did not return text output. Result ID: {result.get('id')}")

        usage = result["usage"]
        resp_obj = ModelResponse(
            text=text_output,
            prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
            cached_tokens=0,
            completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            strategy=None,
            detailed_logs=result.get("detailed_logs")
        )
        
        class MockRawResponse:
            def __init__(self, rid):
                self.id = rid
        
        resp_obj._raw_response = MockRawResponse(result["id"])
        return resp_obj

    def explain(self, prompt: str, prev_resp: ModelResponse) -> Optional[ModelResponse]:
        try:
            kwargs = {
                "model": self.model,
                "previous_response_id": prev_resp._raw_response.id,
                "input": [{"role": "user", "content": prompt}],
                "timeout": 3300
            }
            
            def _create_safe():
                try:
                    return self.client.responses.create(**kwargs)
                except Exception as e:
                    _map_openai_exception(e, self.full_model_name)

            response = run_with_retry(
                lambda: _create_safe(),
                task_id=self.task_id,
                test_index=self.test_index,
                run_timestamp=self.run_timestamp,
                model_name=self.full_model_name,
                timing_tracker=self.timing_tracker
            )
            
            text_output = ""
            if hasattr(response, "output"):
                for item in response.output:
                    if item.type == "message":
                        for content_part in item.content:
                            if content_part.type == "output_text":
                                text_output += content_part.text
            
            usage = getattr(response, "usage", None)
            return ModelResponse(
                text=text_output,
                prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                cached_tokens=0,
                completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
            )
        except Exception as e:
            logger.error(f"Step 2 strategy extraction failed: {e}")
            return None

    def run(self, prompt: str, image_path: Optional[str] = None, return_strategy: bool = False, use_background: bool = False, enable_code_execution: bool = False) -> ModelResponse:
        if use_background:
            return run_with_retry(
                lambda: self.background_solver.solve(prompt, image_path, enable_code_execution=enable_code_execution),
                task_id=self.task_id,
                test_index=self.test_index,
                run_timestamp=self.run_timestamp,
                model_name=self.full_model_name,
                timing_tracker=self.runner.timing_tracker if hasattr(self, 'runner') and hasattr(self.runner, 'timing_tracker') else self.timing_tracker,
                log_success=False
            )
        
        return orchestrate_two_stage(
            lambda p: self.solve_stream(p, image_path, enable_code_execution=enable_code_execution),
            self.explain,
            prompt,
            return_strategy,
            self.verbose,
            image_path
        )
