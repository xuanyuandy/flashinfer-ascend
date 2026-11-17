"""signatures regression coverage."""


import unittest

from flashinfer.runtime import QuantSpec, SchemaError, WorkloadSpec, WorkspaceFormula


class QuantSpecTests(unittest.TestCase):
    def test_valid_groupwise_int4_is_stable(self):
        spec = QuantSpec(
            scheme="symmetric",
            storage_dtype="int4_packed",
            compute_dtype="float16",
            accumulator_dtype="float32",
            scale_dtype="float16",
            granularity="group",
            group_size=[128],
            axis=[1],
            physical_layout="nk_interleaved_v1",
            packing_order="low_nibble_first",
        )
        reconstructed = QuantSpec.from_dict(spec.to_dict())
        self.assertEqual(spec, reconstructed)
        self.assertEqual(spec.fingerprint, reconstructed.fingerprint)

    def test_packed_int4_requires_packing_order(self):
        with self.assertRaisesRegex(SchemaError, "packing_order"):
            QuantSpec(
                scheme="symmetric",
                storage_dtype="int4_packed",
                compute_dtype="float16",
                accumulator_dtype="float32",
            )

    def test_asymmetric_requires_zero_point(self):
        with self.assertRaisesRegex(SchemaError, "zero point"):
            QuantSpec(
                scheme="asymmetric",
                storage_dtype="int8",
                compute_dtype="int8",
                accumulator_dtype="int32",
            )


class WorkloadSpecTests(unittest.TestCase):
    def test_attribute_order_does_not_change_fingerprint(self):
        first = WorkloadSpec(
            op="gemm",
            dtypes=("int8", "int8", "float16"),
            attributes=(("n", "4096"), ("k", "4096")),
        )
        second = WorkloadSpec(
            op="gemm",
            dtypes=("int8", "int8", "float16"),
            attributes=(("k", "4096"), ("n", "4096")),
        )
        self.assertEqual(first.fingerprint, second.fingerprint)

    def test_workspace_formula_is_aligned(self):
        workload = WorkloadSpec(
            op="attention",
            dtypes=("float16",),
            dynamic_bounds=(3, 17),
        )
        formula = WorkspaceFormula(
            constant_bytes=7,
            dynamic_coefficients=(10, 2),
            alignment=32,
        )
        self.assertEqual(formula.size_for(workload), 96)


from dataclasses import FrozenInstanceError

import pytest

from flashinfer.runtime.signature import (
    InputSignature,
    build_signature,
    tensor_spec,
)


class FakeTensor:
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = tuple(shape)


def test_same_inputs_same_signature_and_hash():
    tensors = (FakeTensor("float16", (2, 8, 128)), FakeTensor("float16", (2, 16, 4, 128)))
    sig_a = build_signature("attention.decode", tensors, specializations={"causal": True})
    sig_b = build_signature("attention.decode", tensors, specializations={"causal": True})
    assert sig_a == sig_b
    assert hash(sig_a) == hash(sig_b)


def test_specializations_normalized_by_sorting():
    sig_a = InputSignature(
        op="op", specializations=(("b", "2"), ("a", "1"))
    )
    sig_b = InputSignature(
        op="op", specializations=(("a", "1"), ("b", "2"))
    )
    assert sig_a == sig_b
    assert sig_a.specializations == (("a", "1"), ("b", "2"))


def test_any_component_change_changes_signature():
    base = build_signature(
        "attention.decode",
        (FakeTensor("float16", (1, 8, 128)),),
        specializations={"causal": False},
    )
    assert base != build_signature(
        "attention.decode", (FakeTensor("bfloat16", (1, 8, 128)),),
        specializations={"causal": False},
    )
    assert base != build_signature(
        "attention.decode", (FakeTensor("float16", (2, 8, 128)),),
        specializations={"causal": False},
    )
    assert base != build_signature(
        "attention.decode", (FakeTensor("float16", (1, 8, 128)),),
        specializations={"causal": True},
    )
    assert base != build_signature(
        "attention.prefill", (FakeTensor("float16", (1, 8, 128)),),
        specializations={"causal": False},
    )


def test_signature_is_frozen():
    sig = InputSignature(op="op")
    with pytest.raises(FrozenInstanceError):
        sig.op = "other"  # type: ignore[misc]


def test_tensor_spec_strips_torch_prefix():
    dtype, shape = tensor_spec(FakeTensor("torch.float16", (2, 3)))
    assert dtype == "float16"
    assert shape == (2, 3)


def test_layouts_length_mismatch_rejected():
    with pytest.raises(ValueError):
        build_signature(
            "op",
            (FakeTensor("float16", (1,)), FakeTensor("float16", (1,))),
            layouts=("NHD",),
        )


from flashinfer.jit import build_cache_key, toolchain_fingerprint
from flashinfer.runtime.hardware import HardwareSignature
from flashinfer.runtime.signature import InputSignature

_HW = HardwareSignature(soc_model="Ascend910B", aicore_count=20)
_SIG = InputSignature(
    op="attention.decode",
    dtypes=("float16",),
    shapes=((1, 8, 128),),
    specializations=(("causal", "True"),),
)


def test_same_inputs_same_key():
    assert build_cache_key(_SIG, _HW, toolchain="t1") == build_cache_key(
        _SIG, _HW, toolchain="t1"
    )


def test_input_signature_change_changes_key():
    base = build_cache_key(_SIG, _HW, toolchain="t1")
    changed_dtype = InputSignature(
        op=_SIG.op, dtypes=("bfloat16",), shapes=_SIG.shapes,
        specializations=_SIG.specializations,
    )
    changed_shape = InputSignature(
        op=_SIG.op, dtypes=_SIG.dtypes, shapes=((2, 8, 128),),
        specializations=_SIG.specializations,
    )
    changed_spec = InputSignature(
        op=_SIG.op, dtypes=_SIG.dtypes, shapes=_SIG.shapes,
        specializations=(("causal", "False"),),
    )
    for sig in (changed_dtype, changed_shape, changed_spec):
        assert build_cache_key(sig, _HW, toolchain="t1") != base


def test_hardware_change_changes_key():
    base = build_cache_key(_SIG, _HW, toolchain="t1")
    other_soc = HardwareSignature(soc_model="Ascend910C", aicore_count=20)
    other_aicore = HardwareSignature(soc_model="Ascend910B", aicore_count=24)
    assert build_cache_key(_SIG, other_soc, toolchain="t1") != base
    assert build_cache_key(_SIG, other_aicore, toolchain="t1") != base


def test_toolchain_change_changes_key():
    base = build_cache_key(_SIG, _HW, toolchain="t1")
    assert build_cache_key(_SIG, _HW, toolchain="t2") != base


def test_default_toolchain_fingerprint_is_stable_on_host():
    # host 无工具链时 fingerprint 各项为 "none"，但两次构建必须一致
    assert toolchain_fingerprint() == toolchain_fingerprint()
    assert build_cache_key(_SIG, _HW) == build_cache_key(_SIG, _HW)
