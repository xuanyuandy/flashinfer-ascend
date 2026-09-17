"""InputSignature：不可变、可哈希、规范化（§14.1）。"""

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
