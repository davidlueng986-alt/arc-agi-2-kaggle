"""Hard watchdog: guarantees a valid submission.json exists before the buffer.

Runs in a daemon thread. At the generation deadline it force-writes whatever the
scheduler has accumulated; at the submission deadline it writes once more and, if
`hard_exit` is set, terminates the process so a wedged vLLM client or a hung CUDA
context cannot cost the whole run.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

from .budget import BudgetGovernor
from .submission import SubmissionWriter

Logger = Callable[[str], None]


class Watchdog:
    def __init__(
        self,
        governor: BudgetGovernor,
        writer: SubmissionWriter,
        logger: Optional[Logger] = None,
        hard_exit: bool = False,
        poll_seconds: float = 5.0,
        clock: Optional[Callable[[], float]] = None,
        sleeper: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.governor = governor
        self.writer = writer
        self.log: Logger = logger or (lambda _m: None)
        self.hard_exit = hard_exit
        self.poll_seconds = poll_seconds
        self._now = clock or time.monotonic
        self._sleep = sleeper or time.sleep
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.finalized = False
        self.triggered_at: Optional[float] = None
        self.errors: list = []

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="arc-watchdog", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.poll_seconds * 2 + 1)

    def _safe_write(self) -> bool:
        try:
            errors = self.writer.validate()
            if errors:
                self.errors.extend(errors[:10])
                self.log(f"watchdog: submission has {len(errors)} schema errors (writing anyway)")
            self.writer.write()
            return True
        except Exception as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            self.log(f"watchdog: write failed: {exc}")
            return False

    def _loop(self) -> None:
        while not self._stop.is_set():
            now = self._now()
            if not self.finalized and now >= self.governor.t_gen:
                self.triggered_at = now
                self.log(f"watchdog: generation deadline hit at t+{now - self.governor.t0:.0f}s, finalizing")
                self._safe_write()
                self.finalized = True
            if now >= self.governor.t_submit - self.governor.config.finalize_reserve_seconds:
                self._safe_write()
                if self.hard_exit:
                    self.log("watchdog: submission buffer reached, hard exit")
                    self._sleep(1.0)
                    os._exit(0)
                self.log("watchdog: final submission written, nothing left to guard")
                break
            self._sleep(self.poll_seconds)

    def tick(self) -> None:
        """Synchronous equivalent of one watchdog loop iteration (used by tests)."""
        now = self._now()
        if not self.finalized and now >= self.governor.t_gen:
            self.triggered_at = now
            self.log(f"watchdog: generation deadline hit at t+{now - self.governor.t0:.0f}s, finalizing")
            self._safe_write()
            self.finalized = True
