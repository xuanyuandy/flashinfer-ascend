#!/usr/bin/env python3
"""Exercise real dispatch policies with isolated, metadata-only example records.

No kernel is registered globally, compiled or executed. In particular this does
not extend the production BF16 Matmul to FP16. Run from the repository root.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flashinfer.runtime import (
    Backend, CapabilityCase, CapabilitySpec, CompileMode, DispatchError, FeatureConstraint,
    HardwareSignature, InferenceSpec, InputSignature, KernelDescriptor, KernelRegistry,
    QuantSpec, ShapeRange, dispatch, functional_feature_catalog, get_global_registry,
)


HARDWARE = HardwareSignature("Ascend950", 56, 221184)
SIMPLE_OP = "example.matmul_dtype_routing"
COUPLED_OP = "example.matmul_coupled_routing"


@contextmanager
def policies(**overrides):
    values = {
        "FLASHINFER_NPU_BACKEND": "auto", "FLASHINFER_NPU_JIT_DISABLE": "0",
        "FLASHINFER_NPU_TUNING_DB": "", "FLASHINFER_NPU_DISABLE_DISPATCH_CACHE": "1",
        "FLASHINFER_NPU_DISPATCH_ORDER": "external,cann,aot,jit,reference",
        **overrides,
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def candidate(kernel_id, op, backend, dtype, m_bounds=None):
    return KernelDescriptor(
        kernel_id=kernel_id, backend=backend,
        compile_mode=CompileMode.JIT if backend == Backend.TRITON else CompileMode.NONE,
        capability=CapabilitySpec(
            op=op, dtypes=(dtype,), soc_models=("Ascend950",), physical_layouts=("ND",),
            shape_ranges=() if m_bounds is None else (ShapeRange("M", 0, 0, *m_bounds),),
        ),
    )


def signature(op, dtype, m=32):
    return InputSignature(op, dtypes=(dtype,) * 3,
                          shapes=((m, 32), (32, 64), (m, 64)), physical_layouts=("ND",) * 3)


def run():
    simple = KernelRegistry((
        candidate("illustration/cann/bf16", SIMPLE_OP, Backend.CANN, "bfloat16"),
        candidate("illustration/triton/fp16", SIMPLE_OP, Backend.TRITON, "float16"),
    ))
    simple.freeze()
    # Union of two shape/dtype cases, not the Cartesian product of both sets.
    coupled = KernelRegistry((
        candidate("illustration/triton/small_bf16", COUPLED_OP, Backend.TRITON, "bfloat16", (16, 64)),
        candidate("illustration/triton/large_fp16", COUPLED_OP, Backend.TRITON, "float16", (65, 256)),
    ))
    coupled.freeze()
    evidence = []

    def check(label, registry, sig, backend, expected, **env):
        with policies(**env):
            try:
                decision = dispatch(sig, HARDWARE, backend=backend, registry=registry)
            except DispatchError as error:
                assert expected is None, (label, str(error))
                result = {"selected": None, "outcome": "EXPECTED_REJECTION", "diagnostic": str(error)}
            else:
                assert decision.kernel_id == expected, (label, decision.kernel_id, expected)
                result = {"selected": decision.kernel_id, "outcome": "EXPECTED_SELECTION",
                          "reason": decision.reason,
                          "candidates": [{"kernel_id": report.kernel_id,
                                          "accepted": report.accepted, "reasons": list(report.reasons)}
                                         for report in decision.reports]}
        evidence.append({"case": label, "dtype": sig.dtypes[0], "M": sig.shapes[0][0],
                         "backend_argument": backend, "environment_overrides": env, **result})

    bf16, fp16 = signature(SIMPLE_OP, "bfloat16"), signature(SIMPLE_OP, "float16")
    check("omitted_backend_bf16", simple, bf16, None, "illustration/cann/bf16")
    check("omitted_backend_fp16", simple, fp16, None, "illustration/triton/fp16")
    check("forced_cann_on_fp16", simple, fp16, "cann", None)
    check("forced_triton_on_bf16", simple, bf16, "triton", None)
    check("omitted_obeys_global_cann", simple, fp16, None, None, FLASHINFER_NPU_BACKEND="cann")
    check("explicit_auto_overrides_global_cann", simple, fp16, "auto", "illustration/triton/fp16",
          FLASHINFER_NPU_BACKEND="cann")
    check("unsupported_float32", simple, signature(SIMPLE_OP, "float32"), "auto", None)
    check("jit_disabled_fp16", simple, fp16, "auto", None, FLASHINFER_NPU_JIT_DISABLE="1")
    check("small_M_bf16", coupled, signature(COUPLED_OP, "bfloat16", 32), "auto", "illustration/triton/small_bf16")
    check("large_M_fp16", coupled, signature(COUPLED_OP, "float16", 128), "auto", "illustration/triton/large_fp16")
    check("small_M_fp16_is_not_supported", coupled, signature(COUPLED_OP, "float16", 32), "auto", None)
    check("large_M_bf16_is_not_supported", coupled, signature(COUPLED_OP, "bfloat16", 128), "auto", None)
    return {"kind": "HOST_ROUTING_ONLY_NO_KERNEL_EXECUTION", "status": "PASS",
            "cases_passed": len(evidence), "global_registry_modified": False, "cases": evidence}


def functional_scenarios():
    """Concrete call facts and support cases, deliberately without shape limits.

    The example ops describe hypothetical frontends accepting these contracts,
    not additions to flashinfer.matmul or the production Attention registry.
    Tensor roles: GEMM=(A, B, out), Attention=(Q, K, V, out).
    """
    scenarios = []

    def add(name, domain, backend, dtypes, features, *, quantization=(), fusion_ops=(),
            auxiliary_outputs=(), logits_soft_cap=None, deterministic=False):
        op = "example.functional." + domain
        inference = InferenceSpec(features=features, quantization=quantization,
                                  fusion_ops=fusion_ops, auxiliary_outputs=auxiliary_outputs,
                                  logits_soft_cap=logits_soft_cap, deterministic=deterministic)
        case = CapabilityCase(name, dtype_signatures=(dtypes,),
                              features=tuple(FeatureConstraint(key, (value,)) for key, value in features),
                              quantization=quantization, fusion_ops=fusion_ops,
                              auxiliary_outputs=auxiliary_outputs,
                              supports_logits_soft_cap=bool(logits_soft_cap),
                              supports_deterministic=deterministic)
        mode = {Backend.CANN: CompileMode.NONE, Backend.ASCENDC: CompileMode.AOT,
                Backend.TRITON: CompileMode.JIT}[backend]
        descriptor = KernelDescriptor(f"{backend.value}/example/{name}", backend=backend,
                                      capability=CapabilitySpec(op, cases=(case,)), compile_mode=mode)
        scenarios.append((name, InputSignature(op, dtypes=dtypes, inference=inference), descriptor))

    def gemm_features(compute="float16", accumulator="float32", mode="dense"):
        return (("gemm.mode", mode), ("gemm.transpose_a", False), ("gemm.transpose_b", False),
                ("compute.dtype", compute), ("accumulator.dtype", accumulator))

    add("gemm_bf16", "gemm", Backend.CANN, ("bfloat16",) * 3, gemm_features("bfloat16"))
    add("gemm_fp16", "gemm", Backend.TRITON, ("float16",) * 3, gemm_features())
    w8 = QuantSpec("symmetric", "int8", "float16", "float32", granularity="channel", axis=(1,))
    add("gemm_w8a16", "gemm", Backend.ASCENDC, ("float16", "int8", "float16"),
        gemm_features(), quantization=((1, w8),))
    w4 = QuantSpec("symmetric", "int4_packed", "float16", "float32", granularity="group",
                   group_size=(128,), axis=(0,), packing_order="low_nibble_first")
    add("gemm_w4a16", "gemm", Backend.TRITON, ("float16", "uint8", "float16"),
        gemm_features(), quantization=((1, w4),))
    a8 = QuantSpec("symmetric", "int8", "int8", "int32", granularity="token", axis=(0,))
    b8 = QuantSpec("symmetric", "int8", "int8", "int32", granularity="channel", axis=(1,))
    add("gemm_a8w8", "gemm", Backend.CANN, ("int8", "int8", "bfloat16"),
        gemm_features("int8", "int32"), quantization=((0, a8), (1, b8)))
    fp8 = QuantSpec("symmetric", "float8_e4m3fn", "float8_e4m3fn", "float32",
                   granularity="block", group_size=(128, 128), axis=(0, 1))
    add("gemm_fp8_block", "gemm", Backend.CANN, ("float8_e4m3fn", "float8_e4m3fn", "bfloat16"),
        gemm_features("float8_e4m3fn"), quantization=((0, fp8), (1, fp8)))
    add("gemm_grouped_bias_silu", "gemm", Backend.TRITON, ("bfloat16",) * 3,
        gemm_features("bfloat16", mode="grouped"), fusion_ops=("bias", "silu"))

    def attention_features(mode, variant, cache, *, dtype="bfloat16", layout="NHD", mask="causal",
                           position="NONE", causal=True, window=False, fp16_reduction=False):
        return (("attention.mode", mode), ("attention.variant", variant), ("kv_cache.kind", cache),
                ("kv_cache.layout", layout), ("mask.kind", mask), ("position.encoding", position),
                ("attention.causal", causal), ("attention.sliding_window", window),
                ("attention.fp16_qk_reduction", fp16_reduction),
                ("compute.dtype", dtype), ("accumulator.dtype", "float32"))

    add("attention_prefill_mha", "attention", Backend.CANN, ("bfloat16",) * 4,
        attention_features("prefill", "mha", "dense"))
    add("attention_ragged_gqa_lse", "attention", Backend.TRITON, ("float16",) * 4,
        attention_features("prefill", "gqa", "ragged", dtype="float16", mask="packed_custom",
                           position="ALIBI", causal=False, fp16_reduction=True), auxiliary_outputs=("lse",))
    kv8 = QuantSpec("symmetric", "int8", "bfloat16", "float32", granularity="token", axis=(0,))
    add("attention_decode_mqa_kv8", "attention", Backend.ASCENDC,
        ("bfloat16", "int8", "int8", "bfloat16"),
        attention_features("decode", "mqa", "paged", layout="HND", mask="none", causal=False),
        quantization=((1, kv8), (2, kv8)))
    add("attention_decode_gqa_kv4", "attention", Backend.TRITON,
        ("float16", "uint8", "uint8", "float16"),
        attention_features("decode", "gqa", "paged", dtype="float16", mask="none", causal=False),
        quantization=((1, w4), (2, w4)))
    add("attention_chunked_window_rope", "attention", Backend.CANN, ("bfloat16",) * 4,
        attention_features("chunked_prefill", "gqa", "paged", position="ROPE_LLAMA", window=True),
        logits_soft_cap=50.0, auxiliary_outputs=("lse",))
    add("attention_multi_token_decode", "attention", Backend.TRITON, ("bfloat16",) * 4,
        attention_features("multi_token_decode", "gqa", "paged"), deterministic=True)
    add("attention_block_sparse", "attention", Backend.ASCENDC, ("bfloat16",) * 4,
        attention_features("prefill", "gqa", "dense", mask="custom", causal=False)
        + (("sparse.format", "block_sparse"),))
    return tuple(scenarios)


def run_functional():
    before = tuple((d.kernel_id, d.fingerprint) for d in get_global_registry().descriptors())
    scenarios = functional_scenarios()
    registry = KernelRegistry(descriptor for _, _, descriptor in scenarios)
    registry.freeze()
    evidence = []
    with policies():
        for name, signature, expected in scenarios:
            assert not signature.shapes and not expected.capability.shape_ranges
            for selector in ({"backend": "auto"}, {"backend": expected.backend},
                             {"kernel_id": expected.kernel_id}):
                decision = dispatch(signature, HARDWARE, registry=registry, **selector)
                assert decision.kernel_id == expected.kernel_id, (name, selector, decision)
            evidence.append({"scenario": name, "request": signature.to_dict(),
                             "selected": expected.kernel_id, "backend": expected.backend.value,
                             "case": expected.capability.cases[0].to_dict(), "selection_checks": 3})
    assert tuple((d.kernel_id, d.fingerprint) for d in get_global_registry().descriptors()) == before
    return {"kind": "HOST_FUNCTIONAL_ROUTING_ONLY_NO_KERNEL_EXECUTION", "status": "PASS",
            "scenario_count": len(scenarios), "cases_passed": len(scenarios) * 3,
            "global_registry_modified": False, "kernel_launches": 0,
            "functional_dimensions": len(functional_feature_catalog()), "cases": evidence}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/dispatcher-docs/capability-demo.json"))
    parser.add_argument("--functional", action="store_true", help="Exercise GEMM/Attention functional combinations")
    args = parser.parse_args()
    evidence = run_functional() if args.functional else run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {evidence['cases_passed']} isolated routing checks; no kernel execution; no global registry changes.")
    print(f"Evidence: {args.output}")


if __name__ == "__main__":
    main()
