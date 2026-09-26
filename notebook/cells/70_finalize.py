"""Final validation and report. Self-scores in non-rerun mode (evaluation split has truth)."""
import json
import os

from harness.scoring import score_submission
from harness.submission import SubmissionWriter

SUBMISSION_PATH = os.path.join(WORKING, "submission.json")
expected_ids = {t.task_id for t in tasks}
expected_outputs = sum(t.n_test for t in tasks)

if not os.path.exists(SUBMISSION_PATH):
    log("FATAL: %s does not exist" % SUBMISSION_PATH)
else:
    blob = json.load(open(SUBMISSION_PATH, encoding="utf-8"))
    writer = SubmissionWriter(SUBMISSION_PATH, tasks)
    writer.data = blob
    errors = writer.validate()
    n_outputs = sum(len(v) for v in blob.values())
    log("submission: %d task_ids, %d test-output entries (expected %d / %d)" % (
        len(blob), n_outputs, len(expected_ids), expected_outputs))
    log("missing task_ids: %d" % len(expected_ids - set(blob)))
    log("schema errors: %d %s" % (len(errors), errors[:5]))
    log("size: %.2f MB" % (os.path.getsize(SUBMISSION_PATH) / 1e6))

    report = REPORT.as_dict()
    print()
    print("=" * 78)
    print("H1 pure-xhigh run summary")
    print("=" * 78)
    print("serving rung          : %s" % (SERVING["rung"] if SERVING else "NONE"))
    print("effort mechanism      : %s" % EFFORT_MECHANISM)
    print("tasks / test outputs  : %d / %d" % (report["n_tasks"], report["n_test_outputs"]))
    print("wall clock            : %.0f s" % report["elapsed_s"])
    print("samples/task          : %.2f (xhigh %.2f)" % (
        report["samples_per_task"], report["xhigh_samples_per_task"]))
    print("completion tokens     : %s" % format(report["tokens"]["completion_tokens"], ","))
    print("reasoning tokens      : %s" % format(report["tokens"]["reasoning_tokens"], ","))
    print("requests / failures   : %s / %s" % (report["tokens"]["requests"], report["tokens"]["failures"]))
    print("measured tok/s        : %s" % report["measured_tok_per_s"])
    print("planned tok/s         : %.0f (pessimistic %.0f)" % (PLANNED_TOK_S, PESSIMISTIC_TOK_S))
    print("consensus-stopped     : %d" % report["consensus_stopped_tasks"])
    print("tasks with no parse   : %d" % report["tasks_with_no_parse"])
    print("parse methods         : %s" % report["parse_method_counts"])
    for wave in report["waves"]:
        print("  wave %-4s effort=%-6s tasks=%3d reqs=%4d tok=%10s %7.1fs consensus=%3d%s" % (
            wave["phase"], wave["effort"], wave["tasks"], wave["requests"],
            format(wave["completion_tokens"], ","), wave["seconds"], wave["consensus_after"],
            (" [%s]" % wave["trimmed_reason"]) if wave["trimmed_reason"] else ""))

    if SCORE is not None:
        print("-" * 78)
        print("SCORE                 : %.2f%%  (%d/%d outputs)" % (
            SCORE.score * 100, SCORE.total_correct, SCORE.total_outputs))
        print("  attempt_1 only      : %.2f%%" % (SCORE.attempt_1_score * 100))
        print("  2nd-attempt gain    : %.2f pts" % (SCORE.second_attempt_gain * 100))
    if report.get("coverage"):
        cov = report["coverage"]
        print("-" * 78)
        print("pass@1 (per sample)   : %.2f%%" % (cov["pass_at_1"] * 100))
        print("coverage (pass@N)     : %.2f%%   <- HARD CEILING" % (cov["coverage_pass_at_n"] * 100))
        print("uncovered outputs     : %d" % cov["uncovered_outputs"])
        print("selection efficiency  : %.1f%%" % (cov["selection_efficiency"] * 100))
        print("VERDICT               : %s" % cov["verdict"])
    print("=" * 78)

    log("remaining %.2f h (submission buffer is %.0f s)" % (remaining_s() / 3600.0, SUBMISSION_BUFFER_S))
