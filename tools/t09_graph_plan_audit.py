"""Reproduce T9's plan evidence or disable freezes in memory as a negative control.

python tools/t09_graph_plan_audit.py
python tools/t09_graph_plan_audit.py --negative-control

No source files are changed and no device graph is captured.
"""

import argparse
from copy import copy
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--negative-control", action="store_true")
    args = parser.parse_args()
    import pytest

    target = str(Path(__file__).resolve().parents[1] / "tests/test_attention_graph_plan.py")
    if not args.negative_control:
        raise SystemExit(pytest.main(["-q", "-s", target]))

    from flashinfer.attention.execution_identity import AttentionGraphResourceContract
    from flashinfer.attention.planner import AttentionFrameworkSession

    prepare = AttentionFrameworkSession.prepare_plan
    validate = AttentionGraphResourceContract._validate_plan

    def without_session_freezes(self, spec, metadata):
        probe = copy(self)
        probe._graph_enabled = False
        return prepare(probe, spec, metadata)

    class Results:
        def __init__(self):
            self.outcomes = []

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.failed or report.skipped:
                self.outcomes.append((report.nodeid, report.outcome))

    results = Results()
    print("NEGATIVE CONTROL: session batch/q_len and graph contract freezes disabled in memory")
    try:
        AttentionFrameworkSession.prepare_plan = without_session_freezes
        AttentionGraphResourceContract._validate_plan = lambda *args, **kwargs: None
        code = pytest.main(["-q", "-s", target + "::test_graph_plan_freezes_dimension"], plugins=[results])
    finally:
        AttentionFrameworkSession.prepare_plan = prepare
        AttentionGraphResourceContract._validate_plan = validate
    expected = {"batch_size", "resolved_backend", "q_len_per_req", "block_table_width"}
    failed = {nodeid.rsplit("[", 1)[-1].rstrip("]") for nodeid, outcome in results.outcomes if outcome == "failed"}
    if code != pytest.ExitCode.TESTS_FAILED or failed != expected or len(results.outcomes) != 4:
        raise SystemExit("negative control did not fail exactly the four freeze tests: %r" % results.outcomes)
    print("PASS negative control: 4/4 freeze tests failed with guards removed; source files unchanged")


if __name__ == "__main__":
    main()
