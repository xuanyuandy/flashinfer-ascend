"""Fresh-process T4 acceptance using the pinned CATLASS example's golden data.

Only compilation/discovery/workspace observations are instrumented. The public
API, registry, runner, loader and numerical kernel are the production versions.
The upstream generator is imported read-only and writes into the supplied store's
parent directory, never into the CATLASS checkout.
"""

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("FLASHINFER_TEST_PYTHON", sys.executable))
CATLASS_ROOT = Path(os.environ.get(
    "FLASHINFER_TEST_CATLASS_ROOT", os.environ.get("FLASHINFER_CATLASS_ROOT", "/mnt/fi/catlass-ref"),
))
CATLASS_REVISION = "41bf90da655bba3c66d0acd7e00abe33960ecfd6"
GENERATOR = CATLASS_ROOT / "examples/49_ascend950_flash_attention_infer/gen_data.py"
GENERATOR_SHA256 = "c988b56dc2113990f2d0978185db3eda2cd18db05018b2f042ae3c2f4a6f2fb9"
GOLDEN_SHAPES = (
    (128, 128, 1), (128, 256, 2), (256, 384, 4),
    (384, 512, 3), (512, 512, 8), (512, 128, 1),
)


def golden_generator():
    """Execute the exact pinned generator, preserving its original arithmetic."""
    revision = subprocess.check_output(
        ["git", "-C", str(CATLASS_ROOT), "rev-parse", "HEAD"], text=True,
    ).strip()
    assert revision == CATLASS_REVISION
    assert hashlib.sha256(GENERATOR.read_bytes()).hexdigest() == GENERATOR_SHA256
    module_spec = importlib.util.spec_from_file_location("t4_upstream_gen_data", GENERATOR)
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_spec.name] = module
    module_spec.loader.exec_module(module)
    return module


def golden_inputs(generator, root, shape, dtype="float16"):
    import numpy as np
    import torch

    s, t, heads = shape
    if dtype == "bfloat16":
        torch.manual_seed(1)
        cpu = tuple((torch.rand(length, heads, 128) * 2 - 1).bfloat16() for length in (s, t, t))
        query, key, value = (tensor.double().transpose(0, 1) for tensor in cpu)
        reference = (torch.softmax(query @ key.transpose(1, 2) / 128**.5, -1) @ value).transpose(0, 1)
        return tuple(tensor.npu() for tensor in cpu), reference.float()
    assert dtype == "float16"
    destination = root / ("s%d-t%d-h%d" % shape)
    (destination / "data").mkdir(parents=True, exist_ok=True)
    generator.WORKSPACE = str(destination)
    generator.np.random.seed(1)
    test = generator.TestFlashAttentionInfer()
    parameters = test.GenDataParams(
        [s], [t], heads, heads, 128, (t + 127) // 128, 128, 0, np.float16, 0,
    )
    test.calc_data(parameters)
    tensors = tuple(
        torch.from_numpy(np.fromfile(destination / "data" / (name + ".bin"),
                                     dtype=np.float16).reshape(length, heads, 128)).npu()
        for name, length in (("q", s), ("k", t), ("v", t))
    )
    golden = torch.from_numpy(np.fromfile(destination / "data/golden.bin",
                                         dtype=np.float32).reshape(s, heads, 128))
    return tensors, golden


def run_scenario(scenario, store_root):
    import torch
    import torch_npu  # noqa: F401

    import flashinfer
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID, load_contract
    from flashinfer.jit import AscendCBackend, AscendCToolchain, DiskJitBuildStore
    from flashinfer.jit.discovery import AotCorruptError, AotMissingError
    from flashinfer.aclnn import _native
    from flashinfer.runtime import SchemaError, get_global_registry
    from flashinfer.runtime.schema import DispatchError

    assert Path(sys.executable) == PYTHON, "acceptance requires the specified interpreter"
    assert torch.npu.is_available(), "real NPU is required; a skip is not acceptance"
    assert torch.npu.device_count() == 8, "expected the declared eight Ascend devices"
    store_root = Path(store_root)
    counts = {"compile": 0, "toolchain": 0}
    original_compile = AscendCBackend.compile
    original_discover = AscendCToolchain.discover
    workspace_requests = []

    def counted_compile(self, spec, *, build_directory):
        counts["compile"] += 1
        return original_compile(self, spec, build_directory=build_directory)

    def counted_discover(*args, **kwargs):
        counts["toolchain"] += 1
        assert scenario in {"missing_enabled", "missing_dependency"}, (
            "compiler discovery on a forbidden path"
        )
        return original_discover(*args, **kwargs)

    AscendCBackend.compile = counted_compile
    AscendCToolchain.discover = counted_discover
    os.environ["FLASHINFER_NPU_CACHE_DIR"] = str(store_root)
    enabled = scenario in {"missing_enabled", "corrupt", "missing_dependency"}
    os.environ["FLASHINFER_NPU_JIT_DISABLE"] = "0" if enabled else "1"
    _, recipe, spec = load_contract(FLASH_ATTENTION_KERNEL_ID)
    store = DiskJitBuildStore(store_root)
    assert store.lookup(spec) is None, "acceptance requires a cold JIT store"
    descriptor = get_global_registry().get(FLASH_ATTENTION_KERNEL_ID)
    assert descriptor.runner is not None and descriptor.runner.provenance is None
    assert descriptor.graph_capability.value == "not_capturable"

    def evidence(**values):
        print(json.dumps(dict(scenario=scenario, result="PASS",
                              kernel_id=descriptor.kernel_id, **values), sort_keys=True))

    if scenario == "missing_dependency":
        from tools.build_flash_attention import build_release

        absent = store_root.parent / "absent-catlass-checkout"
        assert not absent.exists()
        os.environ["FLASHINFER_CATLASS_ROOT"] = str(absent)
        try:
            build_release(store_root.parent / "missing-dependency-build")
        except SchemaError as error:
            diagnostic = str(error)
            assert "catlass" in diagnostic.lower() and "dependency" in diagnostic.lower()
            assert "missing" in diagnostic.lower() and str(absent) in diagnostic
            assert counts == {"compile": 0, "toolchain": 1}
            assert descriptor.runner.provenance is None
            evidence(category="dependency", compiler_calls=0, toolchain_calls=1,
                     stage="build_source_resolution", checkout=str(absent),
                     diagnostic=diagnostic)
            return
        raise AssertionError("missing CATLASS dependency did not reject the build")

    generator = golden_generator()
    tensors, golden = golden_inputs(generator, store_root.parent / "golden", GOLDEN_SHAPES[0])
    assert torch.npu.is_initialized()
    if scenario == "hit_disabled":
        assert shutil.which("bisheng") is None
        assert not (Path(os.environ["ASCEND_HOME_PATH"]) / "bin/bisheng").exists()
        assert not Path(os.environ["FLASHINFER_CATLASS_ROOT"]).exists()

    if scenario == "out_of_range":
        q, k, v = tensors
        invalid = (
            ("sequence_maximum", (torch.empty((640, 1, 128), device="npu", dtype=q.dtype), k, v), {}),
            ("sequence_minimum", (q[:64].contiguous(), k, v), {}),
            ("sequence_alignment", (torch.empty((129, 1, 128), device="npu", dtype=q.dtype), k, v), {}),
            ("head_dim", (q[:, :, :64].contiguous(), k[:, :, :64].contiguous(),
                          v[:, :, :64].contiguous()), {}),
            ("heads_maximum", tuple(t.repeat(1, 9, 1) for t in tensors), {}),
            ("unequal_heads", (q.repeat(1, 2, 1), k, v), {}),
            ("dtype", tuple(t.float() for t in tensors), {}),
            ("batch_rank", tuple(t.unsqueeze(0) for t in tensors), {}),
            ("noncontiguous", tuple(t.transpose(0, 2) for t in tensors), {}),
            ("causal", tensors, {"causal": True}),
            ("layout", tensors, {"kv_layout": "HND"}),
            ("mask", tensors, {"custom_mask": torch.zeros((128, 128), device="npu", dtype=torch.bool)}),
            ("out_alias", tensors, {"out": q}),
        )
        diagnostics = []
        for name, values, options in invalid:
            try:
                flashinfer.single_prefill_with_kv_cache_catlass(*values, **options)
            except DispatchError as error:
                diagnostics.append(dict(case=name, diagnostic=str(error)))
                assert descriptor.runner.provenance is None
                assert counts == {"compile": 0, "toolchain": 0}
            else:
                raise AssertionError("out-of-envelope %s reached execution" % name)
        evidence(category="dispatch", runner_resolved=False, compiler_calls=0,
                 rejected=diagnostics)
        return

    try:
        output = flashinfer.single_prefill_with_kv_cache_catlass(*tensors)
    except (AotCorruptError, AotMissingError) as error:
        if scenario == "corrupt":
            assert isinstance(error, AotCorruptError)
            assert "digest mismatch" in str(error)
        else:
            assert scenario == "missing_disabled", str(error)
            assert isinstance(error, AotMissingError)
            assert "no matching AOT op-package" in str(error) and "disabled" in str(error)
        assert counts == {"compile": 0, "toolchain": 0}
        assert store.lookup(spec) is None
        evidence(category=error.category, compiler_calls=0, toolchain_calls=0,
                 policy="enabled" if enabled else "disabled", fallback=False,
                 diagnostic=str(error))
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

    def compare(shape, expected, result, dtype="float16"):
        actual = result.cpu().float()
        assert str(result.dtype).removeprefix("torch.") == dtype
        atol, rtol = (0.001, 0.001) if dtype == "float16" else (0.004, 0.008)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        binding = next(iter(_native._owners.values()))
        workspace_requests.append(_native.last_workspace_size(binding, recipe))
        assert workspace_requests[-1] > 0
        error = (actual - expected).abs()
        errors.append(dict(shape=[shape[0], shape[1], shape[2], 128], dtype=dtype,
                           max_abs_golden=error.max().item(),
                           max_tolerance_ratio=(error / (atol + rtol * expected.abs())).max().item(),
                           aclnn_workspace_bytes=workspace_requests[-1], atol=atol, rtol=rtol))

    compare(GOLDEN_SHAPES[0], golden, output)
    if scenario == "numerical":
        for shape in GOLDEN_SHAPES[1:]:
            device_case, expected = golden_inputs(generator, store_root.parent / "golden", shape)
            supplied_out = torch.empty_like(device_case[0])
            result = flashinfer.single_prefill_with_kv_cache_catlass(*device_case, out=supplied_out)
            assert result is supplied_out
            compare(shape, expected, result)
    for shape in (GOLDEN_SHAPES if scenario == "numerical" else GOLDEN_SHAPES[:1]):
        device_case, expected = golden_inputs(generator, store_root.parent / "golden", shape, "bfloat16")
        supplied_out = torch.empty_like(device_case[0])
        result = flashinfer.single_prefill_with_kv_cache_catlass(*device_case, out=supplied_out)
        assert result is supplied_out
        compare(shape, expected, result, "bfloat16")
    again = flashinfer.single_prefill_with_kv_cache_catlass(*tensors)
    torch.testing.assert_close(again.cpu(), output.cpu(), atol=0, rtol=0)
    assert descriptor.runner.provenance is provenance
    evidence(source=provenance.source, policy="enabled" if enabled else "disabled",
             compiler_calls=counts["compile"], toolchain_calls=counts["toolchain"],
             compiler_path_visible=shutil.which("bisheng") is not None,
             compiler_home_visible=(Path(os.environ["ASCEND_HOME_PATH"]) / "bin/bisheng").is_file(),
             catlass_root_available=Path(os.environ["FLASHINFER_CATLASS_ROOT"]).is_dir(),
             directory=str(provenance.directory), errors=errors,
             graph_capability=descriptor.graph_capability.value, repeated_public_call=True,
             golden_generator=str(GENERATOR), golden_generator_sha256=GENERATOR_SHA256,
             catlass_revision=CATLASS_REVISION,
             golden_precision="FP16: pinned upstream ref_output rounded to float16; BF16: CPU FP64 from exact BF16 inputs")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        if __package__:
            from .flash_attention import main
        else:
            from flash_attention import main
        main()
    else:
        run_scenario(*sys.argv[1:])
