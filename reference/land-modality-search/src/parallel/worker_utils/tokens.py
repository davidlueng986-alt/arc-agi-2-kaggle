import sys
from src.models import parse_model_arg
from src.parallel.limiter import LIMITERS

def acquire_rate_limit_token(model_name: str, verbose: bool = False, prefix: str = ""):
    try:
        model_config = parse_model_arg(model_name)
        provider = model_config.provider
        if provider == "gemini": # Map internal name to config key
            provider = "google"
        
        if provider in LIMITERS:
            if verbose:
                print(f"{prefix} Waiting for rate limit token ({provider})...")
            LIMITERS[provider].acquire()
    except Exception as e:
        print(f"{prefix} Warning: Failed to acquire rate limit token: {e}", file=sys.stderr)
