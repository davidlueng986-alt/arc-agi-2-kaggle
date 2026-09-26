import argparse
import os
import socket
import sys
import httpx
from pathlib import Path
from typing import Optional

from src.types import SUPPORTED_MODELS, ORDERED_MODELS

# Centralized config for rate limits
PROVIDER_RATE_LIMITS = {
    "openai": {"rate": 15, "period": 60},
    "anthropic": {"rate": 15, "period": 60},
    "google": {"rate": 15, "period": 60}
}

class KeepAliveTransport(httpx.HTTPTransport):
    def __init__(self, *args, **kwargs):
        # Get existing options or start empty
        # Note: socket_options can be None in kwargs if not provided, handle that.
        options = list(kwargs.get("socket_options") or [])
        
        # Enable TCP Keep-Alive
        options.append((socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1))
        
        # Linux specific constants (Kaggle runs on Linux)
        # TCP_KEEPIDLE: Start sending keep-alives after 30 seconds of silence
        # TCP_KEEPINTVL: Send subsequent probes every 15 seconds
        # TCP_KEEPCNT: Allow 5 failed probes before killing
        try:
             # Depending on python version/OS, these constants might be in socket module
             TCP_KEEPIDLE = getattr(socket, 'TCP_KEEPIDLE', 4)
             TCP_KEEPINTVL = getattr(socket, 'TCP_KEEPINTVL', 5)
             TCP_KEEPCNT = getattr(socket, 'TCP_KEEPCNT', 6)
             
             options.append((socket.IPPROTO_TCP, TCP_KEEPIDLE, 30))
             options.append((socket.IPPROTO_TCP, TCP_KEEPINTVL, 15))
             options.append((socket.IPPROTO_TCP, TCP_KEEPCNT, 5))
        except AttributeError:
             pass # Fallback for non-Linux if testing locally

        kwargs["socket_options"] = options
        super().__init__(*args, **kwargs)

def get_http_client(**kwargs) -> httpx.Client:
    """Returns a shared httpx client, potentially with insecure SSL if configured."""
    insecure = os.getenv("ARC_AGI_INSECURE_SSL", "").lower() == "true"
    if insecure:
        kwargs["verify"] = False
    
    # Use our custom transport if not explicitly overridden
    if "transport" not in kwargs:
        kwargs["transport"] = KeepAliveTransport(verify=kwargs.get("verify", True))
        # Remove verify from kwargs as it is now passed to transport
        if "verify" in kwargs:
            del kwargs["verify"]

    return httpx.Client(**kwargs)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send ARC-AGI tasks to OpenAI.")
    parser.add_argument(
        "task_source",
        type=str,
        help="JSON file containing a list of task paths (e.g. data/first_100.json) OR a single task ID (e.g. 15696249).",
    )
    parser.add_argument(
        "--model",
        choices=sorted(SUPPORTED_MODELS),
        default="gpt-5.1-none",
        help="Model to use (e.g. gpt-5.1-none, gpt-5.1-low).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=20,
        help="Number of parallel worker threads (default: 20).",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Suggested strategy to include in the prompt.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Output full prompt and model answer to stderr.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of times to run the test suite.",
    )
    parser.add_argument(
        "--extract-strategy",
        action="store_true",
        help="Enable two-stage strategy extraction (Solve -> Explain).",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Enable self-consistency verification using LOOCV on training examples.",
    )
    parser.add_argument(
        "--image",
        action="store_true",
        help="Generate an image for the task and include it in the prompt.",
    )
    return parser.parse_args()

def get_api_keys() -> tuple[Optional[str], Optional[str], list[str]]:
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    
    claude_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    
    google_keys = []
    main_key = os.getenv("GEMINI_API_KEY")
    if main_key:
        google_keys.append(main_key)
        
    # Look for GEMINI_API_KEY_1, GEMINI_API_KEY_2, etc.
    # We'll check a reasonable range, e.g., 1 to 100
    for i in range(1, 101):
        key = os.getenv(f"GEMINI_API_KEY_{i}")
        if key:
            google_keys.append(key)
            
    return openai_key, claude_key, google_keys
