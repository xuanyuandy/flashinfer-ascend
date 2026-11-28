#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Ascend 950 C4A4 acceptance samples and CPU golden."""

from itertools import accumulate
import math

from flashinfer.operators.attention.quant_flash_attn_metadata.cases import golden as metadata_golden


def golden(torch, data, p, metadata):
    """Evaluate the MXFP4 attention and its P quantization on the CPU."""
    from ._cpu_golden import (
        decode_inputs, decode_sections, effective_lengths, iter_tiles, quantize_probability, round_bf16_away,
    )

    schedule = decode_sections(metadata, p)
    q, keys, values = decode_inputs(data, p)
    output = torch.zeros_like(q)
    lse = torch.full(q.shape[:2], float('inf'), dtype=torch.float32)
    task_n, group = schedule['task_n'], schedule['group']
    log_x = math.log2(7.25/1.5) if p['quant_schema'] == 2 else math.log2(7.25)
    qs, ln2 = effective_lengths(p), math.log(2.)
    ln_delta = math.log(1.5) if p['quant_schema'] == 2 else 0.0
    is_ocp = p['quant_schema'] == 0
    scale_bias, lse_shift = (125., 0.) if is_ocp else (124., 3.)
    for batch, head, first, last, parts in iter_tiles(p, schedule):
        rows = torch.arange(first, last)
        tokens = rows//group
        heads = head*group+rows%group
        kh = head if group > 1 else head//(p['N_q']//p['N_kv'])
        qi = q[p['cu_seqlens_q'][batch]+tokens, heads]
        key, value, klen = keys[batch][:, kh], values[batch][:, kh], p['seqused_kv'][batch]
        splits = []
        for task_start, task_stop in parts:
            pv = torch.zeros((len(rows), p['D']), dtype=torch.float32)
            total = torch.zeros(len(rows), dtype=torch.float32)
            running = torch.full_like(total, -1e30)
            # All P payloads in a 16-task group are generated before selecting
            # the common reference. C2 accumulates the tasks before one V2.
            for first_task in range(task_start, task_stop, 16):
                stop_task = min(first_task+16, task_stop)
                lo, hi = first_task*task_n, min(stop_task*task_n, klen)
                padded = (hi-lo+31)//32*32
                score = torch.full((len(rows), padded), -float('inf'))
                score[:, :hi-lo] = (qi @ key[lo:hi].T)*p['softmax_scale']
                if p['mask_mode'] == 3:
                    visible = torch.arange(lo, hi)[None, :] <= tokens[:, None]+klen-qs[batch]
                    score[:, :hi-lo].masked_fill_(~visible, -float('inf'))
                grouped = score.reshape(len(rows), -1, 32)
                maxima = grouped.amax(-1)
                valid = torch.isfinite(maxima)
                scaled_max = torch.where(valid, maxima, 0.)*(1./ln2)
                exponent = torch.floor(scaled_max) if is_ocp else torch.ceil(scaled_max-log_x)
                exp_reference = (exponent-2.)*ln2 if is_ocp else exponent*ln2-ln_delta
                quant = quantize_probability(torch.exp(grouped-exp_reference[..., None]))
                quant = torch.where(valid[..., None], quant, 0.)
                current = torch.maximum(running, torch.where(valid, exponent, -1e30).amax(-1))
                # Production UpdatePScale uses exp(delta_R * ln2), including
                # its FP32 rounding, instead of replacing it with exp2 here.
                alpha = torch.where(running > -1e30, torch.exp((running-current)*ln2), 0.)
                code = (exponent-current[:, None]+scale_bias).clamp(0., 254.)
                weights = (quant*torch.exp2(code-127.)[..., None]).reshape(len(rows), padded)[:, :hi-lo]
                group_pv = torch.zeros_like(pv)
                group_sum = torch.zeros_like(total)
                for task in range(first_task, stop_task):
                    a, b = task*task_n-lo, min((task+1)*task_n, hi)-lo
                    w = weights[:, a:b]
                    group_pv += w @ value[lo+a:lo+b]
                    group_sum += w.sum(-1)
                pv = pv*alpha[:, None]+group_pv
                total = total*alpha+group_sum
                running = current
            splits.append((pv, total, running))
        reference = torch.stack([x[2] for x in splits]).amax(0)
        factors = [torch.where(s > 0, torch.exp2(r-reference), 0.) for _, s, r in splits]
        denominator = torch.zeros_like(reference)
        for (_, s, _), a in zip(splits, factors):
            denominator += s*a
        safe = torch.where(denominator > 0, denominator, 1.)
        result = torch.zeros_like(splits[0][0])
        if len(splits) == 1:
            result = splits[0][0]/safe[:, None]
        else:
            # Current C4 FD merges raw PV using a_i/S, not normalized split O.
            for (pv, _, _), a in zip(splits, factors):
                result += pv*(a/safe)[:, None]
        output[p['cu_seqlens_q'][batch]+tokens, heads] = result
        lse[p['cu_seqlens_q'][batch]+tokens, heads] = torch.where(
            denominator > 0, torch.log(safe)+(reference+lse_shift)*ln2-ln_delta, float('inf'))
    statistics = lse.T.contiguous() if p['enable_lse'] else torch.empty(0, dtype=torch.float32)
    return round_bf16_away(output).to(torch.bfloat16), statistics


def sample(torch, torch_npu, *, dim=128, kv_layout="PA_BBND", qscale_layout="TND",
           causal=False, schema=0, lse=True, q_lengths=(3,), kv_lengths=(63,),
           used_q=None, page=128, heads=4, kv_heads=2, metadata_builder=None):
    from ._cpu_golden import convert_bbnd_tensor, generate_data

    device = torch.device("npu", torch.npu.current_device())
    scale = dim ** -0.5
    mask_mode = 3 if causal else 0
    p = dict(
        B=len(q_lengths), D=dim, N_q=heads, N_kv=kv_heads, block_size=page,
        cu_seqlens_q=(0, *accumulate(q_lengths)), seqused_q=used_q, seqused_kv=kv_lengths,
        q_scale_layout=qscale_layout, mask_mode=mask_mode, softmax_scale=scale,
        quant_schema=schema, enable_lse=lse, seed=421,
    )
    data = generate_data(p)

    def to_nd(tensor):
        return tensor.reshape(-1).to(device).reshape(tensor.shape)

    fp4 = torch.float4_e2m1fn_x2
    e8m0 = torch.float8_e8m0fnu
    args = (
        to_nd(data["q_fp4"]).view(fp4),
        to_nd(convert_bbnd_tensor(data["k_fp4"], kv_layout, "payload")).view(fp4),
        to_nd(convert_bbnd_tensor(data["v_fp4"], kv_layout, "payload")).view(fp4),
        to_nd(data["dequant_scale_q"]).view(e8m0),
        to_nd(convert_bbnd_tensor(data["dequant_scale_k"], kv_layout, "k_scale")).view(e8m0),
        to_nd(convert_bbnd_tensor(data["v_descale"], kv_layout, "v_scale")).view(e8m0),
    )
    lengths = dict(cu_seqlens_q=to_nd(data["cu_seqlens_q"]), seqused_kv=to_nd(data["seqused_kv"]))
    if used_q is not None:
        lengths["seqused_q"] = to_nd(data["seqused_q"])
    layout = dict(layout_kv=kv_layout, layout_q_descale=qscale_layout, mask_mode=mask_mode)
    if metadata_builder is None:
        props = torch_npu.npu.get_device_properties(device.index)
        schedule = metadata_golden(
            torch, dim=dim, heads=heads, kv_heads=kv_heads, q_lengths=q_lengths, kv_lengths=kv_lengths,
            used_q=used_q, qscale_layout=qscale_layout, mask_mode=mask_mode,
            aic=props.cube_core_num, aiv=props.vector_core_num,
        )
        metadata = to_nd(schedule)
    else:
        metadata = metadata_builder(heads, kv_heads, dim, **lengths, **layout, backend="ascendc")
        schedule = metadata.cpu()
    options = dict(
        block_table=to_nd(data["block_table_torch"]), metadata=metadata, softmax_scale=scale,
        max_seqlen_q=max(q_lengths), max_seqlen_kv=max(kv_lengths),
        return_softmax_lse=lse, quant_schema=schema, **lengths, **layout,
    )
    if causal:
        options["attn_mask"] = torch.ones((2048, 2048), dtype=torch.int8, device=device).triu(1)
    return args, options, golden(torch, data, p, schedule)


CASE_PARAMETERS = (
    dict(dim=128, heads=8, kv_heads=1, page=256,
         q_lengths=(17,), kv_lengths=(257,), used_q=(17,),
         qscale_layout="TND", kv_layout="PA_BBND", schema=0, causal=False, lse=False),
    dict(dim=128, heads=8, kv_heads=1, page=256,
         q_lengths=(17,), kv_lengths=(257,), used_q=(17,),
         qscale_layout="TND", kv_layout="PA_NZ", schema=1, causal=True, lse=True),
    dict(dim=128, heads=8, kv_heads=1, page=256,
         q_lengths=(17,), kv_lengths=(257,), used_q=None,
         qscale_layout="N2TGD", kv_layout="PA_BNBD", schema=1, causal=False, lse=True),
    dict(dim=128, heads=8, kv_heads=1, page=256,
         q_lengths=(17,), kv_lengths=(257,), used_q=None,
         qscale_layout="N2TGD", kv_layout="PA_BBND", schema=2, causal=True, lse=False),
    dict(dim=256, heads=8, kv_heads=1, page=256,
         q_lengths=(17,), kv_lengths=(257,), used_q=(17,),
         qscale_layout="TND", kv_layout="PA_BNBD", schema=1, causal=False, lse=False),
    dict(dim=256, heads=8, kv_heads=1, page=256,
         q_lengths=(17,), kv_lengths=(257,), used_q=(17,),
         qscale_layout="TND", kv_layout="PA_BBND", schema=2, causal=True, lse=True),
    dict(dim=256, heads=8, kv_heads=1, page=256,
         q_lengths=(17,), kv_lengths=(257,), used_q=None,
         qscale_layout="N2TGD", kv_layout="PA_NZ", schema=2, causal=False, lse=True),
    dict(dim=256, heads=8, kv_heads=1, page=256,
         q_lengths=(17,), kv_lengths=(257,), used_q=None,
         qscale_layout="N2TGD", kv_layout="PA_BNBD", schema=0, causal=True, lse=False),
    dict(dim=128, heads=8, kv_heads=1, page=128,
         q_lengths=(5, 0, 3), kv_lengths=(0, 33, 1), used_q=(3, 0, 3),
         qscale_layout="TND", kv_layout="PA_BNBD", schema=1, causal=False, lse=False),
    dict(dim=128, heads=8, kv_heads=1, page=128,
         q_lengths=(5, 0, 3), kv_lengths=(0, 33, 1), used_q=(3, 0, 3),
         qscale_layout="TND", kv_layout="PA_BBND", schema=2, causal=True, lse=True),
    dict(dim=128, heads=8, kv_heads=1, page=128,
         q_lengths=(5, 0, 3), kv_lengths=(0, 33, 1), used_q=(3, 0, 3),
         qscale_layout="N2TGD", kv_layout="PA_NZ", schema=2, causal=False, lse=True),
    dict(dim=128, heads=8, kv_heads=1, page=128,
         q_lengths=(5, 0, 3), kv_lengths=(0, 33, 1), used_q=(3, 0, 3),
         qscale_layout="N2TGD", kv_layout="PA_BNBD", schema=0, causal=True, lse=False),
    dict(dim=256, heads=8, kv_heads=1, page=128,
         q_lengths=(5, 0, 3), kv_lengths=(0, 33, 1), used_q=(3, 0, 3),
         qscale_layout="TND", kv_layout="PA_NZ", schema=2, causal=False, lse=False),
    dict(dim=256, heads=8, kv_heads=1, page=128,
         q_lengths=(5, 0, 3), kv_lengths=(0, 33, 1), used_q=(3, 0, 3),
         qscale_layout="TND", kv_layout="PA_BNBD", schema=0, causal=True, lse=True),
    dict(dim=256, heads=8, kv_heads=1, page=128,
         q_lengths=(5, 0, 3), kv_lengths=(0, 33, 1), used_q=(3, 0, 3),
         qscale_layout="N2TGD", kv_layout="PA_BBND", schema=0, causal=False, lse=True),
    dict(dim=256, heads=8, kv_heads=1, page=128,
         q_lengths=(5, 0, 3), kv_lengths=(0, 33, 1), used_q=(3, 0, 3),
         qscale_layout="N2TGD", kv_layout="PA_NZ", schema=1, causal=True, lse=False),
    dict(dim=128, heads=8, kv_heads=1, page=512,
         q_lengths=(3, 9), kv_lengths=(129, 513), used_q=(3, 9),
         qscale_layout="TND", kv_layout="PA_NZ", schema=2, causal=False, lse=False),
    dict(dim=128, heads=8, kv_heads=1, page=512,
         q_lengths=(3, 9), kv_lengths=(129, 513), used_q=(3, 9),
         qscale_layout="TND", kv_layout="PA_BNBD", schema=0, causal=True, lse=True),
    dict(dim=128, heads=8, kv_heads=1, page=512,
         q_lengths=(3, 9), kv_lengths=(129, 513), used_q=None,
         qscale_layout="N2TGD", kv_layout="PA_BBND", schema=0, causal=False, lse=True),
    dict(dim=128, heads=8, kv_heads=1, page=512,
         q_lengths=(3, 9), kv_lengths=(129, 513), used_q=None,
         qscale_layout="N2TGD", kv_layout="PA_NZ", schema=1, causal=True, lse=False),
    dict(dim=256, heads=8, kv_heads=1, page=512,
         q_lengths=(3, 9), kv_lengths=(129, 513), used_q=(3, 9),
         qscale_layout="TND", kv_layout="PA_BBND", schema=0, causal=False, lse=False),
    dict(dim=256, heads=8, kv_heads=1, page=512,
         q_lengths=(3, 9), kv_lengths=(129, 513), used_q=(3, 9),
         qscale_layout="TND", kv_layout="PA_NZ", schema=1, causal=True, lse=True),
    dict(dim=256, heads=8, kv_heads=1, page=512,
         q_lengths=(3, 9), kv_lengths=(129, 513), used_q=None,
         qscale_layout="N2TGD", kv_layout="PA_BNBD", schema=1, causal=False, lse=True),
    dict(dim=256, heads=8, kv_heads=1, page=512,
         q_lengths=(3, 9), kv_lengths=(129, 513), used_q=None,
         qscale_layout="N2TGD", kv_layout="PA_BBND", schema=2, causal=True, lse=False),
    dict(dim=128, heads=16, kv_heads=2, page=128,
         q_lengths=(1,), kv_lengths=(1,), used_q=(1,),
         qscale_layout="TND", kv_layout="PA_BBND", schema=0, causal=True, lse=True),
    dict(dim=256, heads=12, kv_heads=2, page=128,
         q_lengths=(3,), kv_lengths=(3,), used_q=None,
         qscale_layout="N2TGD", kv_layout="PA_NZ", schema=1, causal=False, lse=True),
    dict(dim=256, heads=8, kv_heads=1, page=128,
         q_lengths=(22,), kv_lengths=(22,), used_q=(22,),
         qscale_layout="TND", kv_layout="PA_BNBD", schema=2, causal=False, lse=True),
    dict(dim=128, heads=8, kv_heads=1, page=128,
         q_lengths=(1, 3, 22, 129), kv_lengths=(11, 32771, 65, 259), used_q=None,
         qscale_layout="N2TGD", kv_layout="PA_BNBD", schema=2, causal=True, lse=True),
    dict(dim=256, heads=8, kv_heads=1, page=256,
         q_lengths=(1,), kv_lengths=(32768,), used_q=None,
         qscale_layout="N2TGD", kv_layout="PA_NZ", schema=1, causal=False, lse=True),
    dict(dim=128, heads=8, kv_heads=1, page=256,
         q_lengths=(4,), kv_lengths=(32768,), used_q=(4,),
         qscale_layout="TND", kv_layout="PA_BBND", schema=0, causal=True, lse=True),
)
TOLERANCES = ((0.0001, 0.0078125), (0.000025, 0.005))
