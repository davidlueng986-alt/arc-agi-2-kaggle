import time
import random
from typing import Optional, TYPE_CHECKING
from src.llm_utils import run_with_retry
from src.errors import RetryableProviderError, NonRetryableProviderError, UnknownProviderError
from src.providers.openai_utils import _map_openai_exception
from src.providers.openai_bg.parsing import parse_job_output

if TYPE_CHECKING:
    from src.providers.openai_runner import OpenAIRequestRunner

def submit_job(runner: 'OpenAIRequestRunner', prompt: str, image_path: Optional[str], enable_code_execution: bool) -> str:
    content = runner._prepare_content(prompt, image_path)
    
    kwargs = {
        "model": runner.model,
        "input": [{"role": "user", "content": content}],
        "timeout": 60,
        "background": True,
        "store": True,
        "max_output_tokens": 120000,
    }
    if runner.reasoning_effort != "none":
        kwargs["reasoning"] = {"effort": runner.reasoning_effort}
    
    if enable_code_execution:
        kwargs["tools"] = [{
            "type": "code_interpreter",
            "container": {"type": "auto"}
        }]
        kwargs["tool_choice"] = "auto"
        kwargs["max_tool_calls"] = 100

    if runner.last_failed_job_id:
        kwargs["previous_response_id"] = runner.last_failed_job_id

    def _submit():
        try:
            return runner.client.responses.create(**kwargs)
        except Exception as e:
            _map_openai_exception(e, runner.full_model_name)
    
    job = run_with_retry(
        lambda: _submit(), 
        task_id=runner.task_id, 
        test_index=runner.test_index, 
        run_timestamp=runner.run_timestamp, 
        model_name=runner.full_model_name, 
        timing_tracker=runner.timing_tracker, 
        log_success=False
    )
    return job.id

def poll_job(runner: 'OpenAIRequestRunner', job_id: str, prompt: str, image_path: Optional[str], start_attempt_ts: float):
    # Poll until done or timeout
    max_wait_time = 3300  # 55 minutes
    start_time = time.time()
    poll_interval_base = 2.0
    last_log_time = time.time()
    
    while True:
        # Check Timeout
        elapsed = time.time() - start_time
        if elapsed > max_wait_time:
            if runner.is_downgraded_retry:
                raise NonRetryableProviderError(f"OpenAI Background Job {job_id} timed out after {max_wait_time}s (Downgraded Retry Failed)")

            raise RetryableProviderError(f"OpenAI Background Job {job_id} timed out after {max_wait_time}s")

        # Logging every ~30s
        if runner.verbose and (time.time() - last_log_time > 30):
            print(f"[BACKGROUND] [{runner.model}] Job {job_id} still processing... ({int(elapsed)}s elapsed)")
            last_log_time = time.time()

        # Retrieve Status
        def _retrieve():
            try:
                # Ensure we request outputs during retrieval as well
                return runner.client.responses.retrieve(job_id)
            except Exception as e:
                _map_openai_exception(e, runner.full_model_name)

        job = run_with_retry(
            lambda: _retrieve(), 
            task_id=runner.task_id, 
            test_index=runner.test_index, 
            run_timestamp=runner.run_timestamp, 
            model_name=runner.full_model_name, 
            timing_tracker=runner.timing_tracker, 
            log_success=False
        )

        if job.status in ("queued", "in_progress"):
            sleep_time = poll_interval_base + random.uniform(0, 1.0)
            time.sleep(sleep_time)
            continue
        
        # Terminal States
        if job.status == "completed":
            return parse_job_output(job, start_attempt_ts, runner.timing_tracker, runner.full_model_name)
        
        elif job.status == "failed":
            err_msg = f"Code: {job.error.code}, Message: {job.error.message}" if job.error else "Unknown error"
            raise RetryableProviderError(f"OpenAI Background Job {job_id} FAILED: {err_msg}")
        
        elif job.status in ("cancelled", "incomplete"):
            reason = getattr(job, 'incomplete_details', 'Unknown')
            reason_str = str(reason)
            if "max_output_tokens" in reason_str or "token_limit" in reason_str:
                raise RetryableProviderError(f"OpenAI Background Job {job_id} hit token limit: {reason}")
            
            raise NonRetryableProviderError(f"OpenAI Background Job {job_id} ended with status={job.status}, reason={reason}")
        
        else:
            raise UnknownProviderError(f"OpenAI Background Job {job_id} ended in unexpected status={job.status}")
