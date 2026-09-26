import time
import threading

class RateLimiter:
    """
    Thread-safe Token Bucket Rate Limiter.
    Ensures that no more than `rate` requests are made within `period` seconds.
    Blocks the calling thread until a token is available.
    """
    def __init__(self, rate: float, period: float = 60.0):
        self.rate = rate
        self.period = period
        self.capacity = max(rate, 1.0) # Ensure we can hold at least 1 token
        self.tokens = self.capacity  # Start with full bucket
        self.last_update = time.monotonic()
        self.lock = threading.Lock()

    def _refill(self, now: float):
        """Refills tokens based on time elapsed."""
        elapsed = now - self.last_update
        refill = elapsed * (self.rate / self.period)
        self.tokens = min(self.capacity, self.tokens + refill)
        self.last_update = now

    def acquire(self):
        """
        Acquires a token. Blocks if none are available.
        Releases the lock while sleeping to avoid blocking other threads.
        """
        while True:
            with self.lock:
                now = time.monotonic()
                self._refill(now)

                if self.tokens >= 1:
                    self.tokens -= 1
                    return  # Token acquired

                # Calculate wait time
                needed = 1 - self.tokens
                tokens_per_second = self.rate / self.period
                wait_time = needed / tokens_per_second

            # Sleep outside the lock
            if wait_time > 0:
                time.sleep(wait_time)
