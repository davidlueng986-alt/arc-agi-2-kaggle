"""Budget accounting, deadlines and the watchdog."""

from __future__ import annotations

import threading
import time

import pytest

from harness.budget import BudgetLedger, Deadline, TaskBudget, Usage, Watchdog


class TestUsage:
    def test_add(self):
        a = Usage(prompt_tokens=10, completion_tokens=5, calls=1, seconds=1.0)
        b = Usage(prompt_tokens=20, completion_tokens=5, calls=2, seconds=2.0, python_calls=3)
        a.add(b)
        assert a.prompt_tokens == 30
        assert a.completion_tokens == 10
        assert a.total_tokens == 0
        assert a.calls == 3
        assert a.python_calls == 3
        assert a.seconds == pytest.approx(3.0)

    def test_tokens_per_call(self):
        usage = Usage(total_tokens=100, calls=4)
        assert usage.tokens_per_call == 25.0
        assert Usage().tokens_per_call == 0.0

    def test_as_dict_is_json_safe(self):
        import json

        payload = Usage(prompt_tokens=1, completion_tokens=2, total_tokens=3).as_dict()
        assert json.loads(json.dumps(payload))["total_tokens"] == 3


class TestDeadline:
    def test_remaining_shrinks(self):
        deadline = Deadline(1.0)
        first = deadline.remaining()
        time.sleep(0.05)
        assert deadline.remaining() < first

    def test_expiry(self):
        deadline = Deadline(0.05)
        assert deadline.expired() is False
        time.sleep(0.08)
        assert deadline.expired() is True

    def test_manual_stop(self):
        deadline = Deadline(60)
        assert deadline.stop_requested is False
        deadline.request_stop()
        assert deadline.stop_requested is True
        assert deadline.expired() is True

    def test_zero_duration_still_has_a_positive_span(self):
        deadline = Deadline(0)
        assert deadline.end > deadline.start
        assert deadline.end - deadline.start == pytest.approx(1e-3)


class TestTaskBudget:
    def test_turn_cap(self):
        budget = TaskBudget(max_turns=3, max_tokens=10**9, max_seconds=60)
        assert budget.exceeded(Usage(), 2) is None
        assert "turn cap" in budget.exceeded(Usage(), 3)

    def test_token_cap(self):
        budget = TaskBudget(max_turns=100, max_tokens=1000, max_seconds=60)
        assert "token cap" in budget.exceeded(Usage(total_tokens=1000), 0)

    def test_time_cap(self):
        budget = TaskBudget(max_turns=100, max_tokens=10**9, max_seconds=0.05)
        time.sleep(0.08)
        assert "time cap" in budget.exceeded(Usage(), 0)

    def test_global_deadline(self):
        deadline = Deadline(0.05)
        budget = TaskBudget(max_turns=100, max_tokens=10**9, max_seconds=60, deadline=deadline)
        time.sleep(0.08)
        assert "global deadline" in budget.exceeded(Usage(), 0)

    def test_elapsed(self):
        budget = TaskBudget(max_turns=1, max_tokens=1, max_seconds=1)
        time.sleep(0.02)
        assert budget.elapsed() > 0


class TestWatchdog:
    def test_soft_then_hard_fire_in_order(self):
        deadline = Deadline(0.6)
        events: list[str] = []
        lock = threading.Lock()

        def record(name):
            with lock:
                events.append(name)

        watchdog = Watchdog(
            deadline,
            on_soft=lambda: record("soft"),
            on_hard=lambda: record("hard"),
            soft_fraction=0.3,
            hard_lead_seconds=0.15,
            poll_seconds=0.02,
        )
        watchdog.start()
        watchdog.join(timeout=5)
        assert events == ["soft", "hard"]
        assert watchdog.soft_fired and watchdog.hard_fired

    def test_soft_handler_can_set_the_stop_flag(self):
        deadline = Deadline(0.5)
        watchdog = Watchdog(
            deadline,
            on_soft=deadline.request_stop,
            on_hard=lambda: None,
            soft_fraction=0.2,
            hard_lead_seconds=0.1,
            poll_seconds=0.02,
        )
        watchdog.start()
        watchdog.join(timeout=5)
        assert deadline.stop_requested is True

    def test_handler_exception_does_not_kill_the_watchdog(self):
        deadline = Deadline(0.5)
        fired = []

        def boom():
            fired.append(True)
            raise RuntimeError("bad handler")

        watchdog = Watchdog(
            deadline,
            on_soft=boom,
            on_hard=lambda: fired.append("hard"),
            soft_fraction=0.2,
            hard_lead_seconds=0.1,
            poll_seconds=0.02,
        )
        watchdog.start()
        watchdog.join(timeout=5)
        assert "hard" in fired

    def test_is_a_daemon(self):
        watchdog = Watchdog(Deadline(60), on_soft=lambda: None, on_hard=lambda: None)
        assert watchdog.daemon is True


class TestLedger:
    def test_record_and_snapshot(self):
        ledger = BudgetLedger()
        ledger.record("a", Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15, calls=1))
        ledger.record("b", Usage(total_tokens=25, calls=1))
        snapshot = ledger.snapshot()
        assert snapshot["total"]["total_tokens"] == 40
        assert snapshot["total"]["calls"] == 2
        assert set(snapshot["per_task"]) == {"a", "b"}

    def test_mean_turn_tokens(self):
        ledger = BudgetLedger()
        ledger.record_turn(100, 50)
        ledger.record_turn(200, 50)
        assert ledger.mean_turn_tokens() == 200.0  # mean of 150 and 250
        assert BudgetLedger().mean_turn_tokens() == 0.0

    def test_finish_task_counters(self):
        ledger = BudgetLedger()
        ledger.finish_task("a", verified=True)
        ledger.finish_task("b", verified=False)
        assert ledger.tasks_done == 2
        assert ledger.tasks_solved_verified == 1
        assert ledger.snapshot()["tasks_verified"] == 1

    def test_thread_safe(self):
        ledger = BudgetLedger()

        def worker():
            for _ in range(200):
                ledger.record("t", Usage(total_tokens=1, calls=1))

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert ledger.total.total_tokens == 1600
        assert ledger.total.calls == 1600
