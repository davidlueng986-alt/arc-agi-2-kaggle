import sys
import os
import warnings
import random
from typing import Optional
import PIL.Image
import httpx
import concurrent.futures

from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions

from src.config import get_api_keys, get_http_client, KeepAliveTransport
from src.types import ModelConfig, ModelResponse
from src.llm_utils import run_with_retry, orchestrate_two_stage
from src.logging import get_logger
from src.errors import RetryableProviderError, NonRetryableProviderError, UnknownProviderError

logger = get_logger("providers.gemini")

def call_gemini(
    keys: list[str],
    prompt: str,
    config: ModelConfig,
    image_path: str = None,
    return_strategy: bool = False,
    verbose: bool = False,
    task_id: str = None,
    test_index: int = None,
    run_timestamp: str = None,
    model_alias: str = None,
    timing_tracker: list[dict] = None,
    enable_code_execution: bool = False,
) -> ModelResponse:
    
    model = config.base_model
    thinking_level = str(config.config)
    full_model_name = model_alias if model_alias else f"{model}-{thinking_level}"
    
    # Randomly select a key
    if not keys:
        raise RuntimeError("No Gemini API keys provided.")
    selected_key = random.choice(keys)
    # We can log the key index for debugging
    key_index = keys.index(selected_key)
    if verbose:
        logger.info(f"Using Gemini Key index {key_index}")

    # Instantiate a local client for this call (thread-safe) using shared configuration
    http_client = get_http_client(
        timeout=3300.0,
        transport=KeepAliveTransport(retries=3),
        limits=httpx.Limits(keepalive_expiry=3300)
    )
    
    # Pass the custom client to the Google GenAI SDK
    client = genai.Client(api_key=selected_key, http_options={'httpx_client': http_client})

    # Use thinking_level with string literals "LOW" or "HIGH" (case insensitive usually, but standard is upper/lower matching the enum)
    # Typically the SDK accepts "low" / "high" strings for this field if typed as ThinkingLevel
    level_val = "low" if thinking_level == "low" else "high"
    
    # Configure tools
    tools = []
    if enable_code_execution:
        tools.append(types.Tool(code_execution=types.ToolCodeExecution()))

    gen_config = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=65536,
        tools=tools if tools else None,
        thinking_config=types.ThinkingConfig(
            include_thoughts=True, 
            thinking_level=level_val
        )
    )

    # Shared chat object for state within this function call
    chat = client.chats.create(model=model, config=gen_config)

    def _safe_send(message):
        def _inner_send():
            # Suppress Pydantic serialization warnings from the SDK
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message=".*Pydantic serializer warnings.*")
                return chat.send_message(message)

        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(_inner_send)
                # Enforce hard wall-clock timeout (slightly larger than socket timeout)
                return future.result(timeout=3360)
        except concurrent.futures.TimeoutError as e:
            # LOUD DEBUG LOGGING
            err_msg = (
                f"\n{'!'*50}\n"
                f"!!! GEMINI HARD TIMEOUT TRIGGERED (3360s) !!!\n"
                f"!!! Key Index: {key_index} | Model: {model}\n"
                f"!!! The call hung indefinitely. Killing and retrying.\n"
                f"{'!'*50}\n"
            )
            print(err_msg, file=sys.stderr)
            sys.stderr.flush()
            raise RetryableProviderError(f"Gemini Hard Wall-Clock Timeout (Key #{key_index}, Model: {model}): Call exceeded 3360s") from e
        except Exception as e:
            # 1. Known SDK Retryables
            if isinstance(e, (google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable, google_exceptions.InternalServerError, google_exceptions.TooManyRequests)):
                 raise RetryableProviderError(f"Gemini Transient Error (Key #{key_index}, Model: {model}): {e}") from e
            
            # 2. Known SDK Non-Retryables
            if isinstance(e, (google_exceptions.InvalidArgument, google_exceptions.PermissionDenied, google_exceptions.Unauthenticated)):
                 raise NonRetryableProviderError(f"Gemini Fatal Error (Key #{key_index}, Model: {model}): {e}") from e

            # 3. String matching for other errors
            err_str = str(e)
            if (
                "500" in err_str 
                or "UNAVAILABLE" in err_str 
                or "overloaded" in err_str.lower()
                or "Server disconnected" in err_str
                or "RemoteProtocolError" in err_str
                or "connection closed" in err_str.lower()
                or "peer closed connection" in err_str.lower()
                or "incomplete chunked read" in err_str.lower()
            ):
                raise RetryableProviderError(f"Network/Protocol Error (Key #{key_index}, Model: {model}): {e}") from e

            # 4. Loud Retry
            raise UnknownProviderError(f"Unexpected Gemini Error (Key #{key_index}, Model: {model}): {e}") from e

    def _solve(p: str) -> ModelResponse:
        # Pass raw string to avoid Pydantic warnings; SDK handles wrapping
        message = [p]
        if image_path:
            img = PIL.Image.open(image_path)
            message.append(img)

        response = run_with_retry(
            lambda: _safe_send(message),
            task_id=task_id,
            test_index=test_index,
            run_timestamp=run_timestamp,
            model_name=full_model_name,
            timing_tracker=timing_tracker
        )
        
        try:
            text_parts = []
            detailed_logs = []
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.thought:
                        detailed_logs.append({"type": "thought", "content": part.thought})
                    
                    if part.executable_code:
                        detailed_logs.append({
                            "type": "code", 
                            "code": part.executable_code.code,
                            "language": part.executable_code.language
                        })
                    
                    if part.code_execution_result:
                        detailed_logs.append({
                            "type": "execution_result",
                            "outcome": part.code_execution_result.outcome,
                            "output": part.code_execution_result.output
                        })

                    if part.function_call:
                        detailed_logs.append({
                            "type": "function_call",
                            "name": part.function_call.name,
                            "args": part.function_call.args
                        })

                    if part.text:
                        text_parts.append(part.text)
                        detailed_logs.append({"type": "text", "content": part.text})
            
            usage = response.usage_metadata
            return ModelResponse(
                text="".join(text_parts).strip(),
                prompt_tokens=usage.prompt_token_count if usage and usage.prompt_token_count is not None else 0,
                cached_tokens=0,
                completion_tokens=usage.candidates_token_count if usage and usage.candidates_token_count is not None else 0,
                thought_tokens=getattr(usage, "thoughts_token_count", 0) if usage else 0,
                detailed_logs=detailed_logs
            )
        except Exception as e:
             raise RuntimeError(f"Failed to parse Gemini response: {e} - Raw: {response}")

    def _explain(p: str, prev_resp: ModelResponse) -> Optional[ModelResponse]:
        try:
            # Chat object maintains history automatically
            message = p
            response = run_with_retry(
                lambda: _safe_send(message),
                task_id=task_id,
                test_index=test_index,
                run_timestamp=run_timestamp,
                model_name=full_model_name,
                timing_tracker=timing_tracker
            )
            
            text_parts = []
            detailed_logs = []
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.thought:
                        detailed_logs.append({"type": "thought", "content": part.thought})
                    
                    if part.executable_code:
                        detailed_logs.append({
                            "type": "code", 
                            "code": part.executable_code.code,
                            "language": part.executable_code.language
                        })
                    
                    if part.code_execution_result:
                        detailed_logs.append({
                            "type": "execution_result",
                            "outcome": part.code_execution_result.outcome,
                            "output": part.code_execution_result.output
                        })

                    if part.function_call:
                        detailed_logs.append({
                            "type": "function_call",
                            "name": part.function_call.name,
                            "args": part.function_call.args
                        })

                    if part.text:
                        text_parts.append(part.text)
                        detailed_logs.append({"type": "text", "content": part.text})
            
            usage = response.usage_metadata
            return ModelResponse(
                text="".join(text_parts).strip(),
                prompt_tokens=usage.prompt_token_count if usage and usage.prompt_token_count is not None else 0,
                cached_tokens=0,
                completion_tokens=usage.candidates_token_count if usage and usage.candidates_token_count is not None else 0,
                thought_tokens=getattr(usage, "thoughts_token_count", 0) if usage else 0,
                detailed_logs=detailed_logs
            )
        except Exception as e:
            logger.error(f"Step 2 strategy extraction failed: {e}")
            return None

    return orchestrate_two_stage(_solve, _explain, prompt, return_strategy, verbose, image_path)