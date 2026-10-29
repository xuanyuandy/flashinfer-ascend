"""Fresh-process T2' acceptance through the installed production registry/API.

This harness instruments compilation; it never replaces a descriptor, registry,
runner factory, loader, or numerical implementation.
"""

import json
import os
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]


def run_scenario(scenario, store_root):
    import torch
    import torch_npu  # noqa: F401

    from flashinfer import gemm
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID, load_contract
    from flashinfer.jit import AscendCBackend, AscendCToolchain, DiskJitBuildStore
    from flashinfer.jit.discovery import AotCorruptError, AotMissingError
    from flashinfer.aclnn import _native
    from flashinfer.runtime import get_global_registry
    from flashinfer.runtime.schema import DispatchError

    assert torch.npu.is_available(), "real NPU is required, never count a skip as PASS"
    counts = {"compile": 0, "toolchain": 0}
    original_compile = AscendCBackend.compile
    original_discover = AscendCToolchain.discover
    workspace_requests = []

    def counted_compile(self, spec, *, build_directory):
        counts["compile"] += 1
        return original_compile(self, spec, build_directory=build_directory)

    def counted_discover(*args, **kwargs):
        counts["toolchain"] += 1
        assert scenario == "missing_enabled", "compiler discovery on a forbidden path"
        return original_discover(*args, **kwargs)

    AscendCBackend.compile = counted_compile
    AscendCToolchain.discover = counted_discover
    os.environ["FLASHINFER_NPU_CACHE_DIR"] = str(store_root)
    enabled = scenario in {"missing_enabled", "corrupt"}
    os.environ["FLASHINFER_NPU_JIT_DISABLE"] = "0" if enabled else "1"
    _, recipe, spec = load_contract()
    store = DiskJitBuildStore(store_root)
    assert store.lookup(spec) is None, "acceptance requires a cold JIT store"
    registry = get_global_registry()
    descriptor = registry.get(MATMUL_BIAS_SILU_KERNEL_ID)
    assert descriptor.runner is not None
    assert descriptor.graph_capability.value == "not_capturable"
    assert descriptor.runner.provenance is None

    torch.manual_seed(20261025)

    def inputs(m, n, k):
        # BF16-quantized CPU inputs are the exact values transferred to NPU.
        a = (torch.randn(m, k) * 0.25).bfloat16()
        b = (torch.randn(k, n) * 0.25).bfloat16()
        bias = (torch.randn(n) * 0.1).bfloat16()
        return (a, b, bias), tuple(t.npu() for t in (a, b, bias))

    cpu, tensors = inputs(16, 32, 48)
    # Device initialization precedes discovery, as in ordinary application use.
    assert torch.npu.is_initialized()
    if scenario == "hit_disabled":
        # Keep CANN runtime platform data: host tiling requires it even when
        # using precompiled kernels. Hide compiler executables from PATH and
        # forbid toolchain discovery independently of the runtime data root.
        os.environ["PATH"] = "/usr/bin:/bin"
        assert shutil.which("bisheng") is None

    if scenario == "out_of_range":
        _, invalid = inputs(272, 32, 48)
        try:
            gemm.matmul_bias_silu(*invalid)
        except DispatchError as error:
            assert "256" in str(error) and "272" in str(error)
            assert descriptor.runner.provenance is None
            assert counts == {"compile": 0, "toolchain": 0}
            print(json.dumps(dict(scenario=scenario, result="PASS", category="dispatch",
                                  kernel_id=descriptor.kernel_id, compiler_calls=0,
                                  runner_resolved=False, diagnostic=str(error)), sort_keys=True))
            return
        raise AssertionError("out-of-range shape reached execution")

    try:
        output = gemm.matmul_bias_silu(*tensors)
    except (AotCorruptError, AotMissingError) as error:
        if scenario == "corrupt":
            assert isinstance(error, AotCorruptError)
            assert "digest mismatch" in str(error)
        elif scenario == "incompatible":
            assert isinstance(error, AotMissingError)
            assert "all incompatible" in str(error) and "target_soc mismatch" in str(error)
        else:
            assert scenario == "missing_disabled", str(error)
            assert isinstance(error, AotMissingError)
            assert "no matching AOT op-package" in str(error)
            assert "disabled" in str(error)
        assert counts == {"compile": 0, "toolchain": 0}
        assert store.lookup(spec) is None
        print(json.dumps(dict(scenario=scenario, result="PASS", category=error.category,
                              kernel_id=descriptor.kernel_id, compiler_calls=0,
                              policy="enabled" if enabled else "disabled", fallback=False,
                              diagnostic=str(error)), sort_keys=True))
        return

    assert scenario in {"hit_disabled", "missing_enabled", "numerical"}, "expected rejection"
    provenance = descriptor.runner.provenance
    expected_source = "jit" if scenario == "missing_enabled" else "aot"
    assert provenance.source == expected_source
    assert counts["compile"] == counts["toolchain"] == (expected_source == "jit")
    assert provenance.spec_fingerprint == spec.fingerprint
    if scenario == "hit_disabled":
        assert provenance.directory.parent == ROOT / "flashinfer/data/op_packages"

    errors = []

    def compare(cpu_inputs, result):
        a, b, bias = cpu_inputs
        expected64 = torch.nn.functional.silu(a.double() @ b.double() + bias.double())
        expected32 = torch.nn.functional.silu(a.float() @ b.float() + bias.float())
        actual = result.cpu().double()
        binding = next(iter(_native._owners.values()))
        workspace_requests.append(_native.last_workspace_size(binding, recipe))
        assert workspace_requests[-1] >= 4 * a.shape[0] * b.shape[1]
        # BF16 output rounding dominates; explicitly compare both references.
        atol, rtol = 0.008, 0.008
        torch.testing.assert_close(actual, expected64, atol=atol, rtol=rtol)
        torch.testing.assert_close(actual, expected32.double(), atol=atol, rtol=rtol)
        errors.append(dict(shape=[a.shape[0], b.shape[1], a.shape[1]],
                           max_abs_fp64=(actual - expected64).abs().max().item(),
                           max_abs_fp32=(actual - expected32).abs().max().item(),
                           aclnn_workspace_bytes=workspace_requests[-1],
                           atol=atol, rtol=rtol))

    compare(cpu, output)
    if scenario == "numerical":
        for m, n, k in ((16, 16, 16), (32, 64, 80), (80, 48, 32),
                        (128, 128, 128), (256, 256, 256), (256, 16, 256),
                        (16, 256, 16)):
            cpu_case, device_case = inputs(m, n, k)
            supplied_out = torch.empty((m, n), dtype=torch.bfloat16, device="npu")
            result = gemm.matmul_bias_silu(*device_case, out=supplied_out)
            assert result is supplied_out
            compare(cpu_case, result)
        assert len(set(workspace_requests)) > 1, "host workspace must vary with shape"
    again = gemm.matmul_bias_silu(*tensors)
    torch.testing.assert_close(again.cpu(), output.cpu(), atol=0, rtol=0)
    assert descriptor.runner.provenance is provenance
    print(json.dumps(dict(scenario=scenario, result="PASS", kernel_id=descriptor.kernel_id,
                          source=provenance.source, policy="enabled" if enabled else "disabled",
                          compiler_calls=counts["compile"], toolchain_calls=counts["toolchain"],
                          compiler_path_visible=shutil.which("bisheng") is not None,
                          compiler_home_visible=(Path(os.environ["ASCEND_HOME_PATH"]) / "bin/bisheng").is_file(),
                          directory=str(provenance.directory), errors=errors,
                          graph_capability=descriptor.graph_capability.value,
                          repeated_public_call=True), sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if __package__:
            from .matmul_bias_silu import main
        else:
            from matmul_bias_silu import main
        main()
    else:
        run_scenario(*sys.argv[1:])
