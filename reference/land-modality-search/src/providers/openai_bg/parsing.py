from typing import Any, List, Dict, Tuple
from src.types import ModelResponse

def parse_job_output(job: Any, start_attempt_ts: float, timing_tracker: List[Dict] = None, full_model_name: str = "openai-model") -> ModelResponse:
    text_output = ""
    detailed_logs = []

    if hasattr(job, "output") and job.output:
        for item in job.output:
            # Use model_dump for safe access
            d = item.model_dump() if hasattr(item, "model_dump") else dict(item)
            
            item_type = d.get("type")

            if item_type == "message":
                content = d.get("content") or []
                for content_part in content:
                    if content_part.get("type") == "output_text":
                        txt = content_part.get("text", "")
                        text_output += txt
                        detailed_logs.append({"type": "text", "content": txt})
            
            elif item_type == "reasoning":
                # Capture Thoughts
                thought_content = ""
                content = d.get("content") or []
                for part in content:
                    if part.get("type") in ("reasoning_text", "text"):
                        thought_content += part.get("text", "")
                
                if thought_content:
                    detailed_logs.append({"type": "thought", "content": thought_content})
                elif d.get("reasoning"): 
                    detailed_logs.append({"type": "thought", "content": str(d.get("reasoning"))})

            elif item_type == "code_interpreter_call":
                # Capture Code - use "code" or "input"
                code = d.get("code") or d.get("input") or ""
                
                if code:
                    detailed_logs.append({"type": "code", "code": code, "language": "python"})
                
                # Capture Outputs (logs)
                outputs = d.get("outputs") or d.get("results") or []
                    
                if outputs:
                    for output in outputs:
                        out_type = output.get("type")
                        if out_type == "logs":
                            detailed_logs.append({
                                "type": "execution_result",
                                "output": output.get("logs"),
                                "outcome": "completed"
                            })
                        elif out_type == "image":
                                detailed_logs.append({
                                "type": "execution_result",
                                "output": "<image_data>",
                                "outcome": "image_generated"
                            })
    
    if not text_output and hasattr(job, "output_text") and job.output_text:
        text_output = job.output_text
    
    usage = getattr(job, "usage", None)
    
    import time
    duration = time.perf_counter() - start_attempt_ts
    if timing_tracker is not None:
        timing_tracker.append({
            "type": "attempt",
            "model": full_model_name,
            "duration": duration,
            "status": "success"
        })

    return ModelResponse(
        text=text_output,
        prompt_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
        cached_tokens=0,
        completion_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
        strategy=None,
        detailed_logs=detailed_logs
    )
