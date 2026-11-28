"""Check the schedule exchanged by the two C4A4 AOT packages."""

import pytest

import flashinfer
from flashinfer.aclnn._ops import ensure_registered
from flashinfer.operators.attention.quant_flash_attn.cases import TOLERANCES, sample


def test_metadata_fake_uses_fixed_capacity_and_accepts_out():
    import torch
    from torch._subclasses.fake_tensor import FakeTensorMode

    def forward(cu_seqlens_q, seqused_kv):
        return flashinfer.quant_flash_attn_metadata(32, 2, 256, cu_seqlens_q, seqused_kv, quant_mode=5)

    ensure_registered()
    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    with FakeTensorMode(allow_fallback_kernels=False):
        cu_seqlens_q = torch.empty(2, dtype=torch.int32)
        seqused_kv = torch.empty(1, dtype=torch.int32)
        args = (32, 2, 256, cu_seqlens_q, seqused_kv)
        metadata = compiled(cu_seqlens_q, seqused_kv)
        assert metadata.shape == (1, 57344)
        assert metadata.dtype == torch.int32
        assert metadata.device == cu_seqlens_q.device
        assert flashinfer.quant_flash_attn_metadata(*args, out=metadata) is metadata
        for shape in ((1, 45056), (2, 57344), (57344,)):
            with pytest.raises(ValueError, match="out shape/dtype"):
                bad_out = torch.empty(shape, dtype=torch.int32)
                flashinfer.quant_flash_attn_metadata(*args, out=bad_out)


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend950")
def test_split_packages_exchange_flash_decode_metadata(npu_runtime, aot_runtime):
    torch, torch_npu = npu_runtime
    args, options, expected = sample(
        torch, torch_npu, dim=256, kv_layout="PA_NZ", kv_lengths=(4096,), heads=32,
        metadata_builder=flashinfer.quant_flash_attn_metadata,
    )
    inputs = args + tuple(options[name] for name in ("block_table", "cu_seqlens_q", "seqused_kv"))

    def forward(q, k, v, qs, ks, vs, block_table, cu_seqlens_q, seqused_kv):
        metadata = flashinfer.quant_flash_attn_metadata(
            32, 2, 256, cu_seqlens_q, seqused_kv, layout_kv="PA_NZ", backend="ascendc",
        )
        out, lse = flashinfer.quant_flash_attn(
            q, k, v, qs, ks, vs, block_table=block_table, cu_seqlens_q=cu_seqlens_q, seqused_kv=seqused_kv,
            metadata=metadata, layout_kv="PA_NZ", softmax_scale=256**-0.5, return_softmax_lse=True,
            quant_schema=0, backend="ascendc",
        )
        return metadata, out, lse

    graphs = []

    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward

    metadata, out, lse = forward(*inputs)
    assert metadata.shape == (1, 57344)
    header = metadata[0, :6].tolist()
    assert header[1] == 1  # HEAD_IS_FD_INDEX
    schedule_size = 16 * (1 + header[0] * (header[4] + header[5]))
    atol, rtol = TOLERANCES[0]
    torch.testing.assert_close(out.cpu(), expected[0], rtol=rtol, atol=atol)
    torch.testing.assert_close(lse.cpu(), expected[1], rtol=0, atol=0.001)
    for bad_metadata in (torch.cat((metadata, metadata)), metadata[0]):
        with pytest.raises(ValueError, match=r"metadata requires int32 \[1, C\]"):
            flashinfer.quant_flash_attn(*args, **{**options, "metadata": bad_metadata}, backend="ascendc")

    compiled = torch.compile(forward, backend=backend, fullgraph=True)
    compiled_metadata, compiled_out, compiled_lse = compiled(*inputs)
    assert len(graphs) == 1
    ops = {tuple(str(node.target).split(".")[:2]) for node in graphs[0].graph.nodes if node.op == "call_function"}
    assert {("flashinfer", "quant_flash_attn_metadata"), ("flashinfer", "quant_flash_attn")} <= ops
    torch.testing.assert_close(compiled_metadata[0, :schedule_size], metadata[0, :schedule_size], rtol=0, atol=0)
    torch.testing.assert_close(compiled_out, out, rtol=0, atol=0)
    torch.testing.assert_close(compiled_lse, lse, rtol=0, atol=0.001)
