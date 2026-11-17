"""Validate torch.compile graphs, output buffers and mutations.

Run from the repository root:
  python -B -m tools.check_torch_compile --device cpu
  TORCH_LOGS=graph_breaks python -B -m tools.check_torch_compile --probe

The probe checks graph-break logging. The ordinary run validates public APIs
with Inductor; neither mode is evidence of NPU graph capture/replay."""

import argparse
import os


def probe(torch):
    import torch_npu

    print("torch:", torch.__version__, flush=True)
    print("torch_npu:", torch_npu.__version__, flush=True)
    print("npu available:", torch.npu.is_available(), flush=True)
    print("npu count:", torch.npu.device_count(), flush=True)
    for device in ("npu", "cpu"):
        result = torch.compile(lambda x: x + 1, fullgraph=True)(
            torch.ones(4, device=device),
        )
        print(device, "compile result:", result.cpu(), flush=True)

    def deliberate_break(x):
        torch._dynamo.graph_break()
        return x + 1

    print("GRAPH_BREAK_LOG_POSITIVE_CONTROL_BEGIN", flush=True)
    torch.compile(deliberate_break, backend="eager")(torch.ones(4))
    print("GRAPH_BREAK_LOG_POSITIVE_CONTROL_END", flush=True)


def smoke(torch, device):
    import flashinfer
    from flashinfer.utils import enable_torch_library

    if device == "npu":
        import torch_npu  # noqa: F401

        if "FLASHINFER_TEST_NPU_DEVICE" in os.environ:
            torch.npu.set_device(int(os.environ["FLASHINFER_TEST_NPU_DEVICE"]))

        if not torch.npu.is_available():
            raise RuntimeError("NPU unavailable; cannot validate NPU torch.compile")
    else:
        os.environ["FLASHINFER_NPU_BACKEND"] = "reference"

    def forward(x, residual, weight, out):
        y = flashinfer.rmsnorm(x, weight, eps=1e-5, out=out, enable_pdl=True)
        flashinfer.fused_add_rmsnorm(x, residual, weight, eps=1e-5, enable_pdl=True)
        return y, x, residual

    torch.manual_seed(8)
    x = torch.randn(4, 128, device=device)
    residual = torch.randn_like(x)
    weight = torch.randn(128, device=device)
    # Reference behavior is the original eager path, before opting in.
    expected = forward(x.clone(), residual.clone(), weight, None)
    enable_torch_library()
    enable_torch_library()
    compiled = torch.compile(forward, fullgraph=True)
    for supplied_out in (False, True):
        out = torch.empty_like(x) if supplied_out else None
        actual = compiled(x.clone(), residual.clone(), weight, out)
        if supplied_out:
            assert actual[0] is out
        for observed, wanted in zip(actual, expected):
            torch.testing.assert_close(observed.cpu(), wanted.cpu())
        explanation = torch._dynamo.explain(forward)(
            x.clone(), residual.clone(), weight, out,
        )
        assert explanation.graph_count == 1, explanation
        assert explanation.graph_break_count == 0, explanation
        assert explanation.break_reasons == [], explanation
        targets = [str(n.target) for n in explanation.graphs[0].graph.nodes
                   if n.op == "call_function"]
        assert any("flashinfer.rmsnorm" in name for name in targets), targets
        assert any("flashinfer.fused_add_rmsnorm" in name for name in targets), targets
        print(
            f"PASS device={device} backend=inductor fullgraph=True out={supplied_out}: "
            f"graph_count={explanation.graph_count} "
            f"graph_break_count={explanation.graph_break_count} "
            f"break_reasons={explanation.break_reasons}; numerics and mutations match eager",
            flush=True,
        )
        print("graph targets:", targets, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--device", choices=("cpu", "npu"), default="npu")
    args = parser.parse_args()
    import torch

    if args.probe:
        probe(torch)
    else:
        smoke(torch, args.device)


if __name__ == "__main__":
    main()
