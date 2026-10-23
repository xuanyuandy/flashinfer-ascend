"""Real FIA workspace evidence and in-memory negative controls; no graph capture.

python tools/t10_max_workspace_audit.py
python tools/t10_max_workspace_audit.py --negative-control
python tools/t10_max_workspace_audit.py --interfaces
"""

import argparse
from pathlib import Path


def workspace_allocations(query, key, value, *, workspace=None, **options):
    """Measure only FIA workspace with outputs preallocated, outside recording.

    The .out overload with workspace=None uses the same CANN workspace query
    and allocation as eager .default. NPU allocator 'alloc' events report actual
    allocated bytes (including allocator alignment), excluding outputs, masks,
    inputs and cached/reserved segments. A supplied workspace must remove those
    allocations. This helper executes an operator, never capture/replay.
    """
    import torch
    import torch_npu

    outputs = torch_npu._npu_fused_infer_attention_score_infer_output(
        query, value, input_layout=options["input_layout"], num_heads=options["num_heads"],
        num_key_value_heads=options["num_key_value_heads"],
        softmax_lse_flag=options.get("softmax_lse_flag", False),
    )
    torch.npu.synchronize(query.device)
    torch.npu.memory._record_memory_history(enabled="all", context=None)
    try:
        device_index = query.device.index
        before = len(torch.npu.memory._snapshot()["device_traces"][device_index])
        torch_npu.npu_fused_infer_attention_score.out(
            query, key, value, workspace=workspace, out=list(outputs), **options,
        )
        torch.npu.synchronize(query.device)
        traces = torch.npu.memory._snapshot()["device_traces"][device_index][before:]
        return tuple(event["size"] for event in traces if event["action"] == "alloc")
    finally:
        torch.npu.memory._record_memory_history(enabled=None)


def interfaces():
    import inspect
    import torch
    import torch_npu

    print("torch:", torch.__version__, "torch_npu:", torch_npu.__version__)
    print("NPU:", torch.npu.get_device_name(0))
    query = torch.empty((1, 1, 8, 128), dtype=torch.float16, device="npu:0")
    key = torch.empty((1, 1024, 2, 128), dtype=query.dtype, device=query.device)
    value = torch.empty_like(key)
    for name in ("_npu_fused_infer_attention_score_get_max_workspace",
                 "_npu_fused_infer_attention_score_v2_get_max_workspace"):
        op = getattr(torch_npu, name)
        print(name, "inspect.signature:", inspect.signature(op))
        print("dispatcher:", getattr(torch.ops.npu, name).default._schema)
        options = (dict(num_heads=8, scale=128 ** -.5) if "v2" not in name
                   else dict(num_query_heads=8, softmax_scale=128 ** -.5))
        result = op(query, key, value, input_layout="BSND", num_key_value_heads=2, **options)
        print("return:", type(result).__name__, "shape=", tuple(result.shape), "dtype=", result.dtype,
              "device=", result.device, "numel=", result.numel(), "element_size=", result.element_size(),
              "bytes=", result.numel() * result.element_size(), "data_ptr=", result.data_ptr())
        for label, changes in (
            ("kv4096", {}), ("kv1024", dict(sk=1024)), ("q128", dict(sq=128)),
            ("dim64", dict(dim=64)), ("heads4", dict(heads=4)), ("kv_heads4", dict(kv_heads=4)),
            ("bf16", dict(dtype=torch.bfloat16)), ("lse", dict(lse=True)),
            ("mask", dict(mask=True)), ("scale0.5", dict(scale=.5)),
            ("actual_kv128", dict(actual_kv=128)), ("BNSD", dict(layout="BNSD")),
        ):
            config = dict(sq=1, sk=4096, heads=8, kv_heads=2, dim=128, dtype=torch.float16,
                          lse=False, mask=False, scale=128 ** -.5, actual_kv=None, layout="BSND")
            config.update(changes)
            qshape = (1, config["sq"], config["heads"], config["dim"])
            kshape = (1, config["sk"], config["kv_heads"], config["dim"])
            if config["layout"] == "BNSD":
                qshape = (qshape[0], qshape[2], qshape[1], qshape[3])
                kshape = (kshape[0], kshape[2], kshape[1], kshape[3])
            q = torch.empty(qshape, dtype=config["dtype"], device=query.device)
            k = torch.empty(kshape, dtype=q.dtype, device=q.device)
            v = torch.empty_like(k)
            mask = (torch.empty((1, 1, config["sq"], config["sk"]), dtype=torch.bool, device=q.device)
                    if config["mask"] else None)
            opts = dict(input_layout=config["layout"], num_key_value_heads=config["kv_heads"], atten_mask=mask)
            if "v2" not in name:
                opts.update(num_heads=config["heads"], scale=config["scale"], softmax_lse_flag=config["lse"],
                            actual_seq_lengths_kv=None if config["actual_kv"] is None else [config["actual_kv"]])
            else:
                opts.update(num_query_heads=config["heads"], softmax_scale=config["scale"], return_softmax_lse=config["lse"],
                            actual_seq_kvlen=None if config["actual_kv"] is None else [config["actual_kv"]])
            result = op(q, k, v, **opts)
            print("parameter_probe:", name, label, "dtype=", result.dtype,
                  "numel=", result.numel(), "bytes=", result.numel() * result.element_size())


def negative_controls(target):
    import pytest
    from flashinfer.attention.public_batch import PublicBatchWrapper
    from flashinfer.cann import attention

    class Results:
        def __init__(self):
            self.calls = []

        def pytest_runtest_logreport(self, report):
            if report.when == "call" or report.failed or report.skipped:
                self.calls.append((report.nodeid, report.when, report.outcome))

    maximum = attention.maximum_workspace_bytes
    prepare = PublicBatchWrapper._prepare_native_workspace
    binding = PublicBatchWrapper._validate_native_workspace_binding

    def always_query(self, candidate):
        previous = self._native_workspace_query_only
        self._native_workspace_query_only = True
        try:
            return prepare(self, candidate)
        finally:
            self._native_workspace_query_only = previous

    cases = (
        ("maximum replaced with zero", attention, "maximum_workspace_bytes", lambda *a, **kw: 0,
         "test_maximum_covers_sequence_lengths[float16-64]"),
        ("eager query gate removed", PublicBatchWrapper, "_prepare_native_workspace", always_query,
         "test_eager_and_opt_out_never_query_maximum[eager-opt-in]"),
        ("fixed workspace binding guard removed", PublicBatchWrapper, "_validate_native_workspace_binding",
         lambda self: None, "test_workspace_repoint_is_rejected[plan]"),
    )
    for description, owner, name, mutation, test in cases:
        original = getattr(owner, name)
        results = Results()
        print("NEGATIVE CONTROL:", description, flush=True)
        try:
            setattr(owner, name, mutation)
            code = pytest.main(["-q", "-s", target + "::" + test], plugins=[results])
        finally:
            setattr(owner, name, original)
        if (code != pytest.ExitCode.TESTS_FAILED or len(results.calls) != 1
                or not results.calls[0][0].endswith("::" + test)
                or results.calls[0][1:] != ("call", "failed")):
            raise SystemExit("negative control did not fail the intended test: %r" % results.calls)
        print("PASS negative control:", description, "-> 1/1 intended test failed", flush=True)
    assert attention.maximum_workspace_bytes is maximum
    assert PublicBatchWrapper._prepare_native_workspace is prepare
    assert PublicBatchWrapper._validate_native_workspace_binding is binding
    print("PASS negative controls: 3/3 guarantees detected; source files unchanged")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--negative-control", action="store_true")
    modes.add_argument("--interfaces", action="store_true")
    args = parser.parse_args()
    if args.interfaces:
        interfaces()
        return
    target = str(Path(__file__).resolve().parents[1] / "tests/test_attention_max_workspace.py")
    if args.negative_control:
        negative_controls(target)
    else:
        import pytest
        raise SystemExit(pytest.main(["-q", "-s", target]))


if __name__ == "__main__":
    main()
