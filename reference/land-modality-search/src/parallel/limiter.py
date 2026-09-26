from src.rate_limiter import RateLimiter
from src.config import PROVIDER_RATE_LIMITS

# Initialize global rate limiters per provider
LIMITERS = {
    name: RateLimiter(**config)
    for name, config in PROVIDER_RATE_LIMITS.items()
}

_SCALED = False

def set_rate_limit_scaling(factor: float):
    """
    Scales the rate limits for all providers by a factor.
    Used when running multiple worker processes to divide the global rate limit.
    """
    global _SCALED
    if _SCALED:
        return
    _SCALED = True

    if factor == 1.0:
        return

    for name, limiter in LIMITERS.items():
        original_rate = limiter.rate
        new_rate = original_rate * factor
        
        # Allow fractional rates (e.g. 0.04 RPM) for high worker counts
        if new_rate < 1e-6:
             new_rate = 1e-6 # Safety against zero

        limiter.rate = new_rate
        limiter.per_seconds = 60.0 / new_rate
