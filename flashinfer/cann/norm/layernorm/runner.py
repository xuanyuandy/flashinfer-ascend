"""Execution adapter for norm.layernorm (cann)."""

from __future__ import annotations
from flashinfer.cann.norm._shared import (Any)

def _layernorm_aclnn(
    out: Any, input: Any, weight: Any, bias: Any, eps: float
) -> None:
    import torch.nn.functional as F

    # npu_layer_norm_eval 仅有旧式实现；A5 要求 ACLNN 路径。
    y = F.layer_norm(
        input, [input.shape[-1]], weight=weight, bias=bias, eps=eps
    )
    out.copy_(y)

run = _layernorm_aclnn
