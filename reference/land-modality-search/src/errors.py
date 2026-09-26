class ARCError(Exception):
    """Base class for all application exceptions."""

class ProviderError(ARCError):
    """Base for errors originating from LLM providers."""

class RetryableProviderError(ProviderError):
    """
    Base for all errors that trigger a retry.
    Includes transient network issues, rate limits, and server errors.
    """

class RateLimitProviderError(RetryableProviderError):
    """
    Specific error for 429 Rate Limits, allowing for higher retry counts.
    """

class NonRetryableProviderError(ProviderError):
    """
    Terminal errors that should not be retried.
    Includes 400 Bad Request, Authentication failures, and Context Length Exceeded.
    """

class UnknownProviderError(RetryableProviderError):
    """
    The catch-all for unclassified exceptions.
    Action: RETRY, but log prominently.
    """
