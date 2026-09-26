import time

from harness.budget import Deadline, TokenLedger


class TestTokenLedger:
    def test_billed_vs_effective_with_cache_key(self):
        led = TokenLedger(per_task_effective_cap=1000)
        led.begin_task("t1")
        led.add("judge", 15_000, 4_000, cache_key="judge:t1:0")
        led.add("judge", 15_000, 4_000, cache_key="judge:t1:0")  # prefix-cache hit
        led.add("judge", 15_000, 4_000, cache_key="judge:t1:0")
        totals = led.end_task()
        # billed: 3 x 19K = 57K; effective: 15K once + 3 x 4K = 27K
        assert totals["billed"] == 57_000
        assert totals["effective"] == 27_000
        snap = led.snapshot()
        assert snap["components"]["judge"]["billed"] == 57_000
        assert snap["components"]["judge"]["effective"] == 27_000
        assert snap["components"]["judge"]["calls"] == 3

    def test_no_cache_key_counts_full(self):
        led = TokenLedger()
        led.begin_task("t")
        led.add("solver_expert_a", 4_000, 1_500)
        led.add("solver_expert_a", 4_200, 1_600)
        totals = led.end_task()
        assert totals["billed"] == totals["effective"] == 11_300

    def test_cache_key_scoped_per_task(self):
        led = TokenLedger()
        led.begin_task("t1")
        led.add("judge", 1000, 100, cache_key="k")
        led.add("judge", 1000, 100, cache_key="k")  # hit within t1
        led.end_task()
        led.begin_task("t2")
        led.add("judge", 1000, 100, cache_key="k")  # fresh prefill in t2
        totals = led.end_task()
        assert totals["effective"] == 1100

    def test_over_budget_flag(self):
        led = TokenLedger(per_task_effective_cap=100)
        led.begin_task("t")
        led.add("solver", 60, 10)
        assert not led.task_over_budget("t")
        led.add("solver", 60, 10)
        assert led.task_over_budget("t")


class TestDeadline:
    def _at(self, fraction, total=1000.0):
        return Deadline(total, start=time.monotonic() - fraction * total)

    def test_levels(self):
        assert self._at(0.10).level() == 0
        assert self._at(0.59).level() == 0
        assert self._at(0.61).level() == 1
        assert self._at(0.76).level() == 2
        assert self._at(0.89).level() == 3
        assert self._at(0.94).level() == 4

    def test_remaining_and_expired(self):
        d = self._at(0.5)
        assert 450 < d.remaining() <= 500
        assert not d.expired()
        d2 = self._at(1.05)
        assert d2.remaining() == 0.0
        assert d2.expired()

    def test_fraction_used_clamped(self):
        assert self._at(2.0).fraction_used() == 1.0

    def test_task_deadline_cap(self):
        d = self._at(0.0, total=1_000_000)
        assert d.task_deadline(10, cap_s=900) == 900  # fair share exceeds cap
        d2 = Deadline(100.0)
        assert d2.task_deadline(10, cap_s=900) == 15.0  # 100/10*1.5
