"""Probe the installed native sampler on real hardware, without fallback."""

import torch
import torch_npu

torch.npu.set_device(0)
x = torch.tensor([[0.55, 0.3, 0.1, 0.05]], dtype=torch.float32, device="npu")
k = torch.tensor([0], dtype=torch.int32, device="npu")
p = torch.tensor([1.0], dtype=torch.float32, device="npu")
q = torch.tensor([[1.0, 1.0, 1.0, 1.0]], device="npu")
print(torch.npu.get_device_name(0))
print(torch.ops.npu.npu_top_k_top_p_sample.default._schema)
for name, kwargs in (
    ("plain", {}),
    ("top_p", {"top_p": torch.tensor([0.6], device="npu")}),
    ("min_p", {"min_ps": torch.tensor([0.5], device="npu")}),
):
    call = dict(top_k=k, top_p=p, q=q, input_is_logits=False,
                is_need_logits=True, top_k_guess=0)
    call.update(kwargs)
    try:
        result = torch_npu.npu_top_k_top_p_sample(x, **call)
        torch.npu.synchronize()
        print(name, [t.cpu().tolist() for t in result])
    except Exception as error:
        print(name, type(error).__name__, str(error))
