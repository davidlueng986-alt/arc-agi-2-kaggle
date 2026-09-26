from __future__ import annotations

import pytest

from harness.budget import BudgetConfig, BudgetGovernor, ManualClock, TokenLedger

CFG = BudgetConfig(
    total_seconds=43200.0,
    submission_buffer_seconds=600.0,
    startup_seconds=1020.0,
    finalize_reserve_seconds=120.0,
    safety_reserve_fraction=0.10,
    planned_tok_per_s=450.0,
    pessimistic_tok_per_s=280.0,
    optimistic_tok_per_s=700.0,
)


def make_governor(config: BudgetConfig = CFG, start: float = 0.0):
    clock = ManualClock(start)
    return BudgetGovernor(config, clock=clock), clock


def test_deadline_arithmetic():
    gov, _ = make_governor()
    assert gov.t_hard == pytest.approx(43200.0)
    assert gov.t_submit == pytest.approx(42600.0)
    assert gov.t_gen == pytest.approx(42480.0)
    assert gov.t_gen_start == pytest.approx(1020.0)
    expected_plan = 42480.0 - 0.10 * (42480.0 - 1020.0)
    assert gov.t_plan == pytest.approx(expected_plan)


def test_planning_horizon_is_strictly_inside_generation_deadline():
    gov, _ = make_governor()
    assert gov.t_plan < gov.t_gen < gov.t_submit < gov.t_hard


def test_projected_throughput_defaults_to_plan_before_measurement():
    gov, _ = make_governor()
    assert gov.measured_tok_per_s() is None
    assert gov.projected_tok_per_s() == pytest.approx(450.0)


def test_measured_throughput_after_enough_data():
    gov, clock = make_governor()
    clock.advance(1020.0)
    gov.record(0, 100_000, phase="A", effort="low")
    clock.advance(480.0)
    measured = gov.measured_tok_per_s()
    assert measured is not None
    assert measured == pytest.approx(100_000 / 480.0, rel=1e-6)


def test_measured_throughput_ignored_until_window_is_meaningful():
    gov, clock = make_governor()
    clock.advance(1020.0)
    gov.record(0, 5_000, phase="A", effort="low")
    clock.advance(5.0)
    assert gov.measured_tok_per_s() is None


def test_affordable_tokens_uses_plan_horizon_and_projection():
    gov, clock = make_governor()
    clock.advance(1020.0)
    expected = (gov.t_plan - clock.now()) * 450.0
    assert gov.affordable_completion_tokens() == pytest.approx(expected, rel=1e-9)


def test_plan_wave_full_when_budget_allows():
    gov, clock = make_governor()
    clock.advance(1020.0)
    allowed, reason = gov.plan_wave(240, 3, "xhigh")
    assert allowed == 3
    assert reason == "full"


def test_plan_wave_trims_when_budget_tight():
    gov, clock = make_governor()
    clock.set(gov.t_plan - 10_000.0)
    allowed, reason = gov.plan_wave(240, 4, "xhigh")
    assert 0 < allowed < 4
    assert "trimmed" in reason
    affordable = 10_000.0 * CFG.planned_tok_per_s
    assert allowed == int(affordable // (240 * CFG.xhigh_tokens_per_sample))


def test_plan_wave_returns_zero_when_exhausted():
    gov, clock = make_governor()
    clock.set(gov.t_plan)
    allowed, reason = gov.plan_wave(240, 3, "xhigh")
    assert allowed == 0
    assert "exhausted" in reason or "deadline" in reason


def test_plan_wave_zero_after_generation_deadline():
    gov, clock = make_governor()
    clock.set(gov.t_gen + 1)
    allowed, reason = gov.plan_wave(240, 3, "xhigh")
    assert allowed == 0
    assert reason == "generation-deadline-passed"


def test_plan_wave_empty_inputs():
    gov, clock = make_governor()
    clock.advance(1020.0)
    assert gov.plan_wave(0, 3, "xhigh") == (0, "empty-wave")
    assert gov.plan_wave(240, 0, "xhigh") == (0, "empty-wave")


def test_must_stop_flags():
    gov, clock = make_governor()
    clock.advance(1020.0)
    assert not gov.must_stop_generating()
    assert not gov.must_finalize_now()
    clock.set(gov.t_gen)
    assert gov.must_stop_generating()
    clock.set(gov.t_submit - CFG.finalize_reserve_seconds)
    assert gov.must_finalize_now()


def test_startup_budget_flag():
    gov, clock = make_governor()
    assert not gov.startup_budget_exceeded()
    clock.advance(CFG.startup_seconds + 1)
    assert gov.startup_budget_exceeded()


def test_effort_token_costs():
    assert CFG.tokens_per_sample("xhigh") == 10000
    assert CFG.tokens_per_sample("low") == 1300
    assert CFG.tokens_per_sample("medium") == 4000
    assert CFG.tokens_per_sample("unknown") == 10000


def test_ledger_accumulates_by_phase_and_effort():
    ledger = TokenLedger()
    ledger.record(100, 500, 400, phase="A", effort="low")
    ledger.record(100, 9000, 8500, phase="B1", effort="xhigh")
    ledger.record(0, 0, 0, phase="B1", effort="xhigh", failed=True)
    assert ledger.requests == 3
    assert ledger.failures == 1
    assert ledger.completion_tokens == 9500
    assert ledger.total == 9700
    assert ledger.by_phase["A"]["completion"] == 500
    assert ledger.by_effort["xhigh"]["requests"] == 2
    snapshot = ledger.as_dict()
    assert snapshot["prompt_tokens"] == 200


def test_snapshot_keys():
    gov, clock = make_governor()
    clock.advance(1020.0)
    snap = gov.snapshot()
    for key in (
        "elapsed_s",
        "plan_seconds_left",
        "gen_seconds_left",
        "completion_tokens",
        "projected_tok_per_s",
        "affordable_completion_tokens",
    ):
        assert key in snap


def test_pessimistic_throughput_still_affords_four_xhigh_samples():
    """The central robustness claim of the budget model."""
    cfg = BudgetConfig(**{**CFG.__dict__, "planned_tok_per_s": 280.0})
    gov, clock = make_governor(cfg)
    clock.advance(cfg.startup_seconds)
    usable = cfg.total_seconds - cfg.submission_buffer_seconds - cfg.startup_seconds - cfg.finalize_reserve_seconds
    planned = usable * (1 - cfg.safety_reserve_fraction)
    tokens = planned * 280.0
    phase_a = 240 * 2 * cfg.low_tokens_per_sample
    samples = (tokens - phase_a) / 240 / cfg.xhigh_tokens_per_sample
    assert samples >= 4.0


def test_seconds_for_tokens():
    gov, _ = make_governor()
    assert gov.seconds_for_tokens(4500.0) == pytest.approx(10.0)


def test_estimate_wave_tokens():
    gov, _ = make_governor()
    assert gov.estimate_wave_tokens(240, 3, "xhigh") == 240 * 3 * 10000
