"""Public numerics for the K-dim grouped matmul across both transposed states."""

import pytest

import flashinfer
from flashinfer.operators.gemm.grouped_matmul_k_dim import cases

pytestmark = [
    pytest.mark.real_compile,
    pytest.mark.requires_soc("Ascend910B", "Ascend910C"),
    pytest.mark.usefixtures("aot_runtime"),
]


@pytest.mark.parametrize("configuration", cases.CASE_PARAMETERS)
def test_accuracy(npu_runtime, configuration):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, **configuration)
    actual = flashinfer.gemm.grouped_matmul_k_dim(*args, **options)
    (atol, rtol), = cases.tolerances(configuration)
    torch.testing.assert_close(actual.cpu().double(), expected.double(), atol=atol, rtol=rtol)


def test_output_shape_follows_group_count(npu_runtime):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, m=32, n=64, groups=(16, 0, 48))
    actual = flashinfer.gemm.grouped_matmul_k_dim(*args, **options)
    assert tuple(actual.shape) == tuple(expected.shape) == (3, 32, 64)


def test_empty_group_is_zero_filled(npu_runtime):
    torch, torch_npu = npu_runtime
    args, options, _ = cases.sample(torch, torch_npu, m=32, n=64, groups=(32, 0, 32))
    actual = flashinfer.gemm.grouped_matmul_k_dim(*args, **options)
    assert torch.count_nonzero(actual[1]) == 0


def test_out_argument_is_reused(npu_runtime):
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, groups=(32, 0, 48))
    out = torch.empty_like(expected).npu()
    returned = flashinfer.gemm.grouped_matmul_k_dim(*args, out=out, **options)
    assert returned.data_ptr() == out.data_ptr()
    (atol, rtol), = cases.tolerances({})
    torch.testing.assert_close(returned.cpu().double(), expected.double(), atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "prefixes",
    [
        (32, 16, 96),   # decreasing
        (32, 32, 80),   # does not end at K
        (32, 32, 100),  # past K
    ],
)
def test_malformed_prefixes_are_rejected(npu_runtime, prefixes):
    torch, torch_npu = npu_runtime
    args, options, _ = cases.sample(torch, torch_npu, groups=(32, 0, 48, 16))
    x, weight, _ = args
    bad = torch.tensor(prefixes, dtype=torch.int64).npu()
    with pytest.raises(ValueError, match="group_list"):
        flashinfer.gemm.grouped_matmul_k_dim(x, weight, bad, **options)


def test_reference_backend_matches_golden(npu_runtime):
    """The CPU reference must agree with the author oracle without any device kernel."""
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(torch, torch_npu, groups=(32, 0, 48, 16), fp32_output=True)
    actual = flashinfer.gemm.grouped_matmul_k_dim(*args, backend="reference", **options)
    assert torch.equal(actual.cpu(), expected)


@pytest.mark.parametrize(
    "m,n",
    [
        (3072, 3072),  # dimM=dimN=12: both directions reach the >=12 threshold
        (3584, 1792),  # dimM=14, dimN=7: dimN>6 and dimM/dimN>=2
        (1792, 5376),  # dimM=7, dimN=21: dimM>6 and dimN/dimM>=3
    ],
)
def test_diagonal_sweep_large_tiles(npu_runtime, m, n):
    """Large M/N grids take the diagonal tile-order branch in the kernel;
    small row-major-only grids never validate its mIdx/nIdx remap."""
    torch, torch_npu = npu_runtime
    args, options, expected = cases.sample(
        torch, torch_npu, m=m, n=n, groups=(128,), dtype="float16"
    )
    actual = flashinfer.gemm.grouped_matmul_k_dim(*args, **options)
    (atol, rtol), = cases.tolerances({"dtype": "float16"})
    torch.testing.assert_close(actual.cpu().double(), expected.double(), atol=atol, rtol=rtol)
