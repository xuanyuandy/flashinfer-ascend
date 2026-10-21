"""Compare native FIA layouts for masked GQA LSE on the installed A5 stack."""

import json
import torch
import torch_npu

torch.manual_seed(19)
q = torch.randn(3, 4, 64, dtype=torch.float16)
k = torch.randn(7, 2, 64, dtype=torch.float16)
v = torch.randn_like(k)
mask = torch.arange(7)[None, :] > (torch.arange(3)[:, None] + 4)
scores = torch.einsum("qhd,khd->hqk", q.float(), k.float().repeat_interleave(2, dim=1)) / 8
expected = scores.masked_fill(mask, -torch.inf).logsumexp(-1)
for layout, rank, full_heads, repeat in (
    ("BNSD", 4, False, False), ("BNSD", 4, True, False),
    ("BNSD", 3, False, False), ("BSND", 4, False, False),
    ("BNSD", 4, False, True),
):
    key = k.repeat_interleave(2, dim=1) if repeat else k
    value = v.repeat_interleave(2, dim=1) if repeat else v
    convert = lambda x: (x.transpose(0, 1) if layout == "BNSD" else x).unsqueeze(0).contiguous().npu()
    m = mask[None, None] if rank == 4 else mask[None]
    if full_heads:
        m = m.expand(1, 4, 3, 7).contiguous()
    try:
        _, lse = torch_npu.npu_fused_infer_attention_score(
            convert(q), convert(key), convert(value), atten_mask=m.npu(),
            num_heads=4, num_key_value_heads=4 if repeat else 2,
            input_layout=layout, scale=1 / 8, softmax_lse_flag=True,
        )
        error = (lse.cpu().reshape(4, 3) - expected).abs().max().item()
        print(json.dumps(dict(layout=layout, rank=rank, full_heads=full_heads,
                              repeat_kv=repeat, max_abs_lse_error=error)))
    except Exception as error:
        print(layout, rank, full_heads, repeat, type(error).__name__, str(error))
