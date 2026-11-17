"""Check graph declarations and rerun descriptor-driven caller-owned evidence.

python tools/t11_replay_param_audit.py --interfaces
python tools/t11_replay_param_audit.py --declarations
python tools/t11_replay_param_audit.py --negative-control
python tools/t11_replay_param_audit.py --portable-skips
python tools/t11_replay_param_audit.py

Capture/replay belongs to the tests/caller, never to flashinfer runtime.
"""

import argparse
import ast
from collections import Counter
from dataclasses import replace
from pathlib import Path


def audit_declarations(descriptors, *, root=None):
    """Evidence gate, not a universal proof of capturability of arbitrary code.

    Hardware verification still runs the cited caller-owned capture/replay
    tests. CPU-only CI checks evidence bindings and test existence; a skipped
    hardware test is never new evidence for promoting a declaration.
    """
    from flashinfer.runtime import GraphCapability, SchemaError
    from flashinfer.runtime._graph_params import _validate_graph_declaration

    root = Path(__file__).resolve().parents[1] if root is None else Path(root)
    descriptors = tuple(descriptors)
    verified = []
    for descriptor in descriptors:
        _validate_graph_declaration(descriptor)
        if descriptor.graph_capability == GraphCapability.NOT_CAPTURABLE:
            continue
        spec = descriptor.graph_replay
        for selector in spec.validation_tests:
            path, separator, function = selector.partition("::")
            source = (root / path).resolve()
            if (not separator or not path.startswith("tests/") or not source.is_relative_to(root.resolve())
                    or not source.is_file() or not any(
                        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function
                        for node in ast.walk(ast.parse(source.read_text())))):
                raise SchemaError("%s: missing graph validation test %s" % (descriptor.kernel_id, selector))
        verified.append(descriptor)
    print("records:", len(descriptors), "graph_capability:",
          dict(Counter(d.graph_capability.value for d in descriptors)))
    for descriptor in verified:
        print("evidence:", descriptor.kernel_id, "strategy=" + descriptor.graph_replay.strategy,
              "updates=" + repr(descriptor.graph_param_updates))
        print("tests:", ", ".join(descriptor.graph_replay.validation_tests))
        print("parameter_sources:", descriptor.graph_replay.parameter_sources)
        print("validated_environment:", descriptor.graph_replay.validated_environment)
    print("PASS declaration evidence gate: %d capturable, %d conservative" %
          (len(verified), len(descriptors) - len(verified)))
    return tuple(verified)


def negative_control(target):
    import pytest
    from flashinfer.runtime import GraphCapability, SchemaError, get_global_registry

    registry = get_global_registry()
    unverified = registry.get("reference/attention/decode_v1")
    lie = replace(unverified, graph_capability=GraphCapability.CAPTURE_SAFE)
    try:
        audit_declarations((lie,))
    except SchemaError as error:
        print("PASS unverified capture_safe declaration rejected:", error, flush=True)
    else:
        raise SystemExit("FAIL unverified capturable declaration passed")

    # Retain the real declaration and evidence; inject a runner that works in
    # eager but fails during capture. The cited hardware test must catch it.
    original = registry.get("cann/attention/single_decode_fia_v1")

    def cannot_capture(*args, **kwargs):
        import torch
        if torch.npu.is_current_stream_capturing():
            raise RuntimeError("injected declared-capturable runner fails during capture")
        return original.runner(*args, **kwargs)

    class Results:
        calls = []

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.failed or report.skipped:
                self.calls.append((report.when, report.outcome, str(report.longrepr)))

    results = Results()
    bad = replace(original, runner=cannot_capture)
    registry._descriptors[original.kernel_id] = bad
    try:
        code = pytest.main(["-q", "-s", target +
                            "::test_fia_full_capture_replay_needs_parameter_update[ramp-float16]"],
                           plugins=[results])
    finally:
        registry._descriptors[original.kernel_id] = original
    if results.calls and all(outcome == "skipped" for _, outcome, _ in results.calls):
        print("SKIP live declaration negative control: requires verified NPU")
        return
    if (code != pytest.ExitCode.TESTS_FAILED or len(results.calls) != 1
            or results.calls[0][:2] != ("call", "failed")
            or "injected declared-capturable runner fails during capture" not in results.calls[0][2]):
        raise SystemExit("FAIL declaration negative control did not fail its intended hardware test")
    print("PASS declared capturable but capture fails -> 1/1 cited hardware test failed; declaration restored")


def portable_skips():
    """Run device tests with dependencies/device deliberately unavailable."""
    import os
    import subprocess
    import sys

    code = r'''
import importlib.abc
import sys
mode = sys.argv[1]
if mode == 'no_device':
    import torch
    import torch_npu
    torch.npu.is_available = lambda: False
else:
    blocked = {'torch'} if mode == 'no_torch' else {'torch_npu'}
    class Missing(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split('.')[0] in blocked:
                raise ModuleNotFoundError('simulated missing dependency: ' + fullname, name=fullname)
    sys.meta_path.insert(0, Missing())
import pytest
raise SystemExit(pytest.main(['-q', 'tests/framework/test_replay_param_npu.py']))
'''
    for mode in ("no_torch", "no_torch_npu", "no_device"):
        print("environment:", mode, flush=True)
        result = subprocess.run(
            [sys.executable, "-c", code, mode], cwd=Path(__file__).resolve().parents[1],
            env={**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"},
            text=True, capture_output=True,
        )
        print(result.stdout, end="", flush=True)
        if result.returncode or " skipped" not in result.stdout:
            raise SystemExit("portable skip check failed: " + result.stderr)


def interfaces():
    import inspect
    import torch
    import torch_npu

    print("torch:", torch.__version__, "torch_npu:", torch_npu.__version__)
    print("NPU:", torch.npu.get_device_name(0))
    for name in ("graph_task_update_begin", "graph_task_update_end", "graph_task_group_begin",
                 "graph_task_group_end", "ExternalEvent", "NPUGraph", "graph"):
        print(name, inspect.signature(getattr(torch.npu, name)))
    for name in ("wait", "reset", "record"):
        print("ExternalEvent." + name, inspect.signature(getattr(torch.npu.ExternalEvent, name)))
    handle = torch_npu._C._NPUTaskGroupHandle
    print("handle type:", handle)
    try:
        print("handle signature:", inspect.signature(handle))
    except ValueError as error:
        print(type(error).__name__ + ":", error)
    try:
        handle()
    except TypeError as error:
        print("handle constructor:", error)
    print("FIA dispatcher:", torch.ops.npu.npu_fused_infer_attention_score.out._schema)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--interfaces", action="store_true")
    modes.add_argument("--declarations", action="store_true")
    modes.add_argument("--negative-control", action="store_true")
    modes.add_argument("--portable-skips", action="store_true")
    args = parser.parse_args()
    if args.interfaces:
        interfaces()
        return
    if args.portable_skips:
        portable_skips()
        return
    from flashinfer.runtime import get_global_registry
    verified = audit_declarations(get_global_registry().descriptors())
    target = str(Path(__file__).resolve().parents[1] / "tests/framework/test_replay_param_npu.py")
    if args.negative_control:
        negative_control(target)
    elif not args.declarations:
        import pytest
        # The current safety tests share one file. New kernels may cite other
        # caller-owned hardware tests; run those too rather than merely finding
        # their source. Avoid running selectors twice under a whole-file target.
        selectors = sorted({test for d in verified for test in d.graph_replay.validation_tests
                            if not test.startswith("tests/framework/test_replay_param_npu.py::")})
        raise SystemExit(pytest.main(["-q", "-s", target, *selectors]))


if __name__ == "__main__":
    main()
