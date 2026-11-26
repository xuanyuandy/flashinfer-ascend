"""ACLGraph evidence for the public Ascend950 C4A4 fixed-buffer entry point."""

import pytest

import flashinfer
from flashinfer.operators.attention.quant_flash_attn.cases import sample


@pytest.mark.real_compile
@pytest.mark.requires_soc("Ascend950")
@pytest.mark.parametrize("dim,layout,kv_length,lse", (
    (128, "PA_BBND", 128, False),
    (256, "PA_BNBD", 4096, True),
    (256, "PA_NZ", 4096, True),
))
def test_c4a4_public_fixed_output_replay(device_runtime, aot_runtime, dim, layout, kv_length, lse):
    torch, torch_npu = device_runtime
    heads = 32
    args, options, _ = sample(torch, torch_npu, dim=dim, kv_layout=layout, kv_lengths=(kv_length,),
                              heads=heads, schema=2, lse=lse)
    options["metadata"] = flashinfer.quant_flash_attn_metadata(
        heads, 2, dim, cu_seqlens_q=options["cu_seqlens_q"], seqused_kv=options["seqused_kv"],
        layout_kv=layout, backend="ascendc",
    )
    values = args[2].view(torch.uint8)
    original = values.clone()
    alternate = (original.cpu() ^ 0x88).to(original.device)

    def run(**kwargs):
        return flashinfer.quant_flash_attn(*args, **options, **kwargs, backend="ascendc")

    expected = []
    for payload in (original, alternate):
        values.copy_(payload)
        expected.append(run())
    assert not torch.equal(expected[0][0], expected[1][0])
    outputs = tuple(torch.empty_like(tensor) for tensor in expected[0])
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    try:
        with torch.npu.graph(graph):
            result = run(out=outputs)
            assert all(result[index] is output for index, output in enumerate(outputs))
        for index in (0, 1, 0, 1):
            values.copy_((original, alternate)[index])
            for output in outputs:
                if output.numel():
                    output.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            for actual, reference in zip(outputs, expected[index]):
                assert torch.equal(actual, reference)
    finally:
        graph.reset()
