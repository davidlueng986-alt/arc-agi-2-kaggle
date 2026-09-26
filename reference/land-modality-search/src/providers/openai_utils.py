import openai
from src.errors import RetryableProviderError, NonRetryableProviderError, UnknownProviderError, RateLimitProviderError

def _map_openai_exception(e: Exception, model_name: str):
    """Maps OpenAI SDK exceptions to internal provider errors."""
    if isinstance(e, openai.RateLimitError):
        raise RateLimitProviderError(f"OpenAI Rate Limit (Model: {model_name}): {e}") from e

    if isinstance(e, (openai.APIConnectionError, openai.InternalServerError)):
        raise RetryableProviderError(f"OpenAI Transient Error (Model: {model_name}): {e}") from e
    
    if isinstance(e, (openai.BadRequestError, openai.AuthenticationError, openai.PermissionDeniedError)):
        raise NonRetryableProviderError(f"OpenAI Fatal Error (Model: {model_name}): {e}") from e

    err_str = str(e)
    if (
        "Connection error" in err_str
        or "500" in err_str
        or "server_error" in err_str
        or "upstream connect error" in err_str
        or "timed out" in err_str
        or "Server disconnected" in err_str
        or "RemoteProtocolError" in err_str
        or "connection closed" in err_str.lower()
        or "peer closed connection" in err_str.lower()
        or "incomplete chunked read" in err_str.lower()
    ):
        raise RetryableProviderError(f"Network/Protocol Error (Model: {model_name}): {e}") from e

    raise UnknownProviderError(f"Unexpected OpenAI Error (Model: {model_name}): {e}") from e
