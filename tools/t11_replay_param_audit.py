"""Inspect installed ACL graph APIs or run the real T11A replay evidence.

python tools/t11_replay_param_audit.py --interfaces
python tools/t11_replay_param_audit.py

Capture/replay belongs to the tests/caller, never to flashinfer runtime.
"""

import argparse
from pathlib import Path


def interfaces():
    import inspect
    import torch
    import torch_npu

    print("torch:", torch.__version__, "torch_npu:", torch_npu.__version__)
    print("NPU:", torch.npu.get_device_name(0))
    for name in ("graph_task_update_begin", "graph_task_update_end", "graph_task_group_begin",
                 "graph_task_group_end", "ExternalEvent", "NPUGraph", "graph"):
        print(name, inspect.signature(getattr(torch.npu, name)))
    for name in ("wait", "reset", "record"):
        print("ExternalEvent." + name, inspect.signature(getattr(torch.npu.ExternalEvent, name)))
    handle = torch_npu._C._NPUTaskGroupHandle
    print("handle type:", handle)
    try:
        print("handle signature:", inspect.signature(handle))
    except ValueError as error:
        print(type(error).__name__ + ":", error)
    try:
        handle()
    except TypeError as error:
        print("handle constructor:", error)
    print("FIA dispatcher:", torch.ops.npu.npu_fused_infer_attention_score.out._schema)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interfaces", action="store_true")
    args = parser.parse_args()
    if args.interfaces:
        interfaces()
    else:
        import pytest
        target = Path(__file__).resolve().parents[1] / "tests/test_replay_param_npu.py"
        raise SystemExit(pytest.main(["-q", "-s", str(target)]))


if __name__ == "__main__":
    main()
