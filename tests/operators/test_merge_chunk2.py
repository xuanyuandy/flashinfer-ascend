"""Arch22 public merge_chunk2 invalid-input checks."""

import pytest

import flashinfer
from flashinfer.operators.fused_moe.merge_chunk2 import cases
from flashinfer.runtime import DispatchError

pytestmark = [pytest.mark.real_compile, pytest.mark.requires_soc("Ascend910B", "Ascend910C"),
              pytest.mark.usefixtures("aot_runtime")]


@pytest.mark.parametrize("problem,error,match", [
    ("negative", ValueError, "nonnegative"), ("sum1", ValueError, "row count"),
    ("sum2", ValueError, "row count"), ("group_shape", ValueError, "identical shape"),
    ("data_rows", ValueError, "token counts"), ("mixed_dtype", ValueError, "dtype"),
    ("group_dtype", ValueError, "int32"), ("hidden", DispatchError, "H1"),
])
def test_invalid_input(npu_runtime, problem, error, match):
    torch, torch_npu = npu_runtime
    data, _, _ = cases.sample(torch, torch_npu, h1=17 if problem == "hidden" else 16)
    a, b, c, d, g1, g2 = data
    if problem == "negative":
        g1[0], g1[1] = -1, 3  # Same total, but invalid addresses.
    elif problem == "sum1":
        g1[0] += 1
    elif problem == "sum2":
        g2[0] -= 1
    elif problem == "group_shape":
        g2 = g2[:-1]
    elif problem == "data_rows":
        c = c[:-1]
    elif problem == "mixed_dtype":
        d = d.bfloat16()
    elif problem == "group_dtype":
        g2 = g2.long()
    with pytest.raises(error, match=match):
        flashinfer.merge_chunk2(a, b, c, d, g1, g2)
