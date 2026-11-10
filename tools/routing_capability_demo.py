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

from flashinfer.runtime import (
    Backend, CapabilitySpec, CompileMode, DispatchError, HardwareSignature,
    InputSignature, KernelDescriptor, KernelRegistry, ShapeRange, dispatch,
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("build/dispatcher-docs/capability-demo.json"))
    args = parser.parse_args()
    evidence = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {evidence['cases_passed']} isolated routing checks; no kernel execution; no global registry changes.")
    print(f"Evidence: {args.output}")


if __name__ == "__main__":
    main()
