"""Installed-wheel numerical checks for the packaged production registry.

Each kernel runs in a fresh process because CANN custom-op registration is
process-scoped. Only the installed package's AOT directory is eligible, and JIT
is disabled even when the caller has enabled it or configured extra AOT roots.
Neither repository tests nor build dependencies are used by these checks.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys


_RESULT_PREFIX = "FLASHINFER_SELFTEST_RESULT="


@dataclass(frozen=True)
class SelftestResult:
    kernel_id: str
    status: str
    reason: str


class _SelftestSkip(Exception):
    pass


def _package_root() -> Path:
    from ._resources import data_root

    return data_root() / "op_packages"


def _production_descriptors():
    from .runtime import load_kernel_manifest

    from ._resources import resource_path

    return load_kernel_manifest(resource_path("kernels/registry.json"))


def _load_dependencies():
    try:
        import torch
        import torch_npu  # noqa: F401
    except (ImportError, OSError) as error:
        raise _SelftestSkip("runtime dependency missing: %s" % error) from error
    if not torch.npu.is_available() or torch.npu.device_count() == 0:
        raise _SelftestSkip("no NPU device is available")
    return torch


def _compare(torch, actual, expected, *, atol, rtol):
    actual = actual.cpu().double()
    expected = expected.double()
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    return "max_abs=%.6g, atol=%g, rtol=%g" % (
        (actual - expected).abs().max().item(),
        atol,
        rtol,
    )


def _matmul_bias_silu(torch, kernel_id):
    from .gemm import matmul_bias_silu

    a = (torch.randn(16, 48) * 0.25).bfloat16()
    b = (torch.randn(48, 32) * 0.25).bfloat16()
    bias = (torch.randn(32) * 0.1).bfloat16()
    actual = matmul_bias_silu(
        *(tensor.npu() for tensor in (a, b, bias)),
        kernel_id=kernel_id,
    )
    expected = torch.nn.functional.silu(a.double() @ b.double() + bias.double())
    return _compare(torch, actual, expected, atol=0.008, rtol=0.008)


def _matmul(torch, kernel_id):
    from .gemm import matmul

    a = (torch.randn(32, 48) * 0.25).bfloat16()
    b = (torch.randn(48, 64) * 0.25).bfloat16()
    actual = matmul(a.npu(), b.npu(), kernel_id=kernel_id)
    return _compare(torch, actual, a.double() @ b.double(), atol=0.008, rtol=0.008)


def _flash_attention(torch, kernel_id):
    from .attention.catlass_prefill import single_prefill_with_kv_cache_catlass

    comparisons = []
    for dtype in (torch.float16, torch.bfloat16):
        cpu = tuple(
            (torch.rand(shape) * 2 - 1).to(dtype)
            for shape in ((128, 2, 128), (256, 2, 128), (256, 2, 128))
        )
        actual = single_prefill_with_kv_cache_catlass(
            *(tensor.npu() for tensor in cpu),
            kernel_id=kernel_id,
        )
        if actual.dtype != dtype:
            raise ValueError("FlashAttention output dtype must match its inputs")
        # CPU FP64 reference uses the exact quantized values copied to the NPU.
        query, key, value = (tensor.double().transpose(0, 1) for tensor in cpu)
        probability = torch.softmax((query @ key.transpose(1, 2)) / (128**0.5), dim=-1)
        expected = (probability @ value).transpose(0, 1)
        atol, rtol = (0.001, 0.001) if dtype == torch.float16 else (0.004, 0.008)
        comparisons.append(
            str(dtype) + ": " + _compare(torch, actual, expected, atol=atol, rtol=rtol)
        )
    return "; ".join(comparisons)


# This maps numerical oracles, not the operator inventory: every registry entry
# is visited, and a newly registered op without an oracle fails explicitly.
_NUMERICAL_CASES = {
    "gemm.matmul": _matmul,
    "gemm.matmul_bias_silu": _matmul_bias_silu,
    "ascendc.single_prefill_catlass": _flash_attention,
}


def _check_kernel(descriptor) -> SelftestResult:
    kernel_id = descriptor.kernel_id
    try:
        case = _NUMERICAL_CASES.get(descriptor.op)
        if case is None:
            from .operators.catalog import for_kernel
            from .operators.testing import selftest_case

            operator = for_kernel(kernel_id)
            if operator is not None:
                case = lambda torch, selected: selftest_case(torch, operator.name, kernel_id=selected)
        if case is None:
            raise ValueError(
                "no numerical selftest registered for op=%s" % descriptor.op
            )
        torch = _load_dependencies()
        from .ascendc import load_contract
        from .jit.discovery import AotMissingError, AotPackageIndex
        from .jit.env import JitCompilationPolicy
        from .runtime import get_global_registry

        _, _, spec = load_contract(kernel_id)
        root = _package_root()
        index = AotPackageIndex(roots=(root,))
        try:
            index.find(spec, descriptor=descriptor, target_soc=spec.target_soc)
        except AotMissingError as error:
            raise _SelftestSkip(
                "AOT artifact not bundled in %s; compilation=disabled; %s"
                % (root, error)
            ) from error

        from .runtime.hardware import current_hardware
        soc = current_hardware().soc_model
        if soc != spec.target_soc:
            raise ValueError("selftest kernel requires %s; local SoC is %s" % (spec.target_soc, soc))

        runner = get_global_registry().get(kernel_id).runner
        if runner is None or runner.provenance is not None:
            raise RuntimeError("selftest requires a fresh production AOT runner")
        runner.index = index
        runner.policy = JitCompilationPolicy.DISABLED
        torch.manual_seed(20261026)
        comparison = case(torch, kernel_id)
        provenance = runner.provenance
        if (
            provenance is None
            or provenance.source != "aot"
            or provenance.directory.resolve().parent != root.resolve()
            or provenance.spec_fingerprint != spec.fingerprint
        ):
            raise RuntimeError("execution did not use the bundled AOT artifact")
        return SelftestResult(
            kernel_id,
            "PASS",
            "bundled AOT hit; compilation=disabled; CPU reference matched (%s); spec=%s"
            % (comparison, spec.fingerprint),
        )
    except _SelftestSkip as error:
        return SelftestResult(kernel_id, "SKIP", str(error))
    except Exception as error:
        return SelftestResult(
            kernel_id, "FAIL", "%s: %s" % (type(error).__name__, error)
        )


def _run_kernel_process(kernel_id) -> SelftestResult:
    environment = dict(
        os.environ, FLASHINFER_NPU_JIT_DISABLE="1", FLASHINFER_AOT_PATHS=""
    )
    command = [sys.executable, "-B", "-m", "flashinfer.selftest", "--kernel", kernel_id]
    try:
        completed = subprocess.run(
            command,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return SelftestResult(kernel_id, "FAIL", "selftest process failed: %s" % error)
    records = [
        line[len(_RESULT_PREFIX) :]
        for line in completed.stdout.splitlines()
        if line.startswith(_RESULT_PREFIX)
    ]
    try:
        if len(records) != 1:
            raise ValueError("expected one selftest result record")
        result = SelftestResult(**json.loads(records[0]))
        if (
            result.kernel_id != kernel_id
            or result.status not in {"PASS", "FAIL", "SKIP"}
            or not isinstance(result.reason, str)
            or not result.reason
            or completed.returncode != (0 if result.status == "PASS" else 1)
        ):
            raise ValueError("invalid selftest result or exit status")
        return result
    except (TypeError, ValueError) as error:
        details = (completed.stderr or completed.stdout).strip()
        return SelftestResult(
            kernel_id,
            "FAIL",
            "selftest process exit=%s: %s; %s" % (completed.returncode, error, details),
        )


def _wheel_metadata():
    from ._resources import resource_path

    path = resource_path("wheel.json")
    return json.loads(path.read_text()) if path.is_file() else None


def _local_wheel_descriptors(descriptors):
    metadata = _wheel_metadata()
    if metadata is None:
        return descriptors
    from ._wheel import distribution_name, family_socs
    from .runtime.hardware import current_hardware

    family = metadata.get("family")
    socs = family_socs(family)
    if (metadata.get("schema_version") != 2
            or metadata.get("soc_models") != list(socs)
            or metadata.get("distribution") != distribution_name(family)
            or metadata.get("kernel_ids") != sorted(d.kernel_id for d in descriptors)
            or any(d.target_soc not in socs for d in descriptors)):
        raise ValueError("wheel metadata does not match the production registry")
    _load_dependencies()
    soc = current_hardware().soc_model
    if soc not in socs:
        raise ValueError("wheel family %s does not match local SoC %s" % (family, soc))
    local = tuple(d for d in descriptors if d.target_soc == soc)
    if not local:
        raise ValueError("wheel contains no production kernel for local SoC " + soc)
    print("Scope: %s; %d of %d packaged kernels; other SoCs are not runtime-validated"
          % (soc, len(local), len(descriptors)), flush=True)
    return local


def run_selftest(*, kernel_ids=()) -> int:
    """Check the wheel's physical local target; skips remain incomplete validation."""
    try:
        descriptors = _production_descriptors()
        if kernel_ids:
            requested = tuple(kernel_ids)
            known = {d.kernel_id for d in descriptors}
            if len(requested) != len(set(requested)) or set(requested) - known:
                raise ValueError("selftest requires unique registered kernel IDs")
            descriptors = tuple(d for d in descriptors if d.kernel_id in requested)
        else:
            descriptors = _local_wheel_descriptors(descriptors)
        if not descriptors:
            raise ValueError("production registry is empty")
    except Exception as error:
        print("FAIL registry: %s: %s" % (type(error).__name__, error), flush=True)
        print("Summary: 0 PASS, 1 FAIL, 0 SKIP (total=1)", flush=True)
        return 1
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for descriptor in descriptors:
        result = _run_kernel_process(descriptor.kernel_id)
        counts[result.status] += 1
        print("%s %s: %s" % (result.status, descriptor.op, result.reason), flush=True)
    print(
        "Summary: %d PASS, %d FAIL, %d SKIP (total=%d)"
        % (
            counts["PASS"],
            counts["FAIL"],
            counts["SKIP"],
            len(descriptors),
        ),
        flush=True,
    )
    return int(bool(counts["FAIL"] or counts["SKIP"]))


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    args = parser.parse_args()
    descriptors = _production_descriptors()
    descriptor = next(
        (item for item in descriptors if item.kernel_id == args.kernel), None
    )
    if descriptor is None:
        result = SelftestResult(
            args.kernel, "FAIL", "kernel is absent from the production registry"
        )
    else:
        result = _check_kernel(descriptor)
    print(_RESULT_PREFIX + json.dumps(asdict(result), sort_keys=True), flush=True)
    return 0 if result.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(_main())
