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

"""Packed MXFP4 inputs and CPU reference helpers adapted from qfa_mxfp4_test."""

import torch


LEVELS = (0., .5, 1., 1.5, 2., 3., 4., 6.)


def unpack_fp4(payload):
    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2)
    levels = torch.tensor(LEVELS, dtype=torch.float32)
    return levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1., 1.)


def decode_scale(payload):
    return torch.exp2(payload.float() - 127.)


def round_bf16_away(values):
    bits = values.float().contiguous().view(torch.int32)
    return ((bits + 0x8000) & -65536).view(torch.float32)


def quantize_probability(values):
    boundaries = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], dtype=torch.float32)
    levels = torch.tensor(LEVELS, dtype=torch.float32)
    rounded = round_bf16_away(values)
    return levels[torch.bucketize(rounded.contiguous(), boundaries, right=True)]


def generate_data(p):
    gen = torch.Generator().manual_seed(p['seed'])
    total, heads, kv_heads, dimension, page = (p['cu_seqlens_q'][-1], p['N_q'], p['N_kv'], p['D'], p['block_size'])
    counts = [(length + page - 1)//page for length in p['seqused_kv']]
    pages = sum(counts) + 3
    # Randomize unused pages and tails too: an erroneous padding read is observable.
    q = torch.randint(0, 256, (total, heads, dimension//2), dtype=torch.uint8, generator=gen)
    k = torch.randint(0, 256, (pages, page, kv_heads, dimension//2), dtype=torch.uint8, generator=gen)
    v = torch.randint(0, 256, k.shape, dtype=torch.uint8, generator=gen)
    qs = torch.randint(122, 126, (total, heads, dimension//64, 2), dtype=torch.uint8, generator=gen)
    ks = torch.randint(122, 126, (pages, page, kv_heads, dimension//64, 2), dtype=torch.uint8, generator=gen)
    vs = torch.randint(124, 128, (pages, page//64, kv_heads, dimension, 2), dtype=torch.uint8, generator=gen)
    if p['q_scale_layout'] == 'N2TGD':
        qs = qs.reshape(total, kv_heads, heads//kv_heads, dimension//64, 2).permute(1, 0, 2, 3, 4).contiguous()
    table = torch.zeros((p['B'], max(1, max(counts))), dtype=torch.int32)
    permutation = torch.randperm(pages, generator=gen)
    offset = 0
    for batch, count in enumerate(counts):
        table[batch, :count] = permutation[offset:offset+count].to(torch.int32)
        offset += count
    return dict(q_fp4=q, k_fp4=k, v_fp4=v, dequant_scale_q=qs,
                dequant_scale_k=ks, v_descale=vs, block_table_torch=table,
                cu_seqlens_q=torch.tensor(p['cu_seqlens_q'], dtype=torch.int32),
                seqused_q=None if p['seqused_q'] is None else torch.tensor(p['seqused_q'], dtype=torch.int32),
                seqused_kv=torch.tensor(p['seqused_kv'], dtype=torch.int32))


def decode_inputs(data, p):
    total, heads, kv_heads, dimension, page = (p['cu_seqlens_q'][-1], p['N_q'], p['N_kv'], p['D'], p['block_size'])
    qs = data['dequant_scale_q']
    if p['q_scale_layout'] == 'N2TGD':
        qs = qs.permute(1, 0, 2, 3, 4).contiguous()
    qs = qs.reshape(total, heads, dimension//32)
    query = unpack_fp4(data['q_fp4']) * decode_scale(qs).repeat_interleave(32, -1)
    keys, values = [], []
    for batch, length in enumerate(p['seqused_kv']):
        count = (length + page - 1)//page
        ids = data['block_table_torch'][batch, :count].long()
        k = unpack_fp4(data['k_fp4'][ids])
        ks = decode_scale(data['dequant_scale_k'][ids]).reshape(count, page, kv_heads, dimension//32)
        keys.append((k * ks.repeat_interleave(32, -1)).reshape(count*page, kv_heads, dimension)[:length])
        v = unpack_fp4(data['v_fp4'][ids])
        vs = decode_scale(data['v_descale'][ids]).permute(0, 1, 4, 2, 3)
        vs = vs.reshape(count, page//32, kv_heads, dimension).repeat_interleave(32, 1)
        values.append((v * vs).reshape(count*page, kv_heads, dimension)[:length])
    return query, keys, values


def convert_bbnd_tensor(tensor, layout, kind):
    """Convert fully materialized physical BBND uint8 storage without changing values."""
    if layout == 'PA_BBND':
        return tensor
    if kind not in ('payload', 'k_scale', 'v_scale'):
        raise ValueError(f'Unknown KV tensor kind: {kind}')
    if layout == 'PA_BNBD':
        # payload/K scale: [Bn,Bs,N,...]; V scale: [Bn,Bs/64,N,D,2]
        return tensor.transpose(1, 2).contiguous()
    if layout != 'PA_NZ':
        raise ValueError(f'Unsupported KV layout: {layout}')
    bnbd = convert_bbnd_tensor(tensor, 'PA_BNBD', kind)
    bn, heads = bnbd.shape[:2]
    if kind == 'payload':
        page, row_bytes = bnbd.shape[2:4]
        if row_bytes % 32:
            raise ValueError('PA_NZ payload rows must contain whole 32-byte fractals')
        return bnbd.reshape(bn, heads, page, row_bytes//32, 32).permute(0, 1, 3, 2, 4).contiguous()
    if kind == 'k_scale':
        page, groups, pair = bnbd.shape[2:5]
        if page % 16:
            raise ValueError('PA_NZ K scale requires page size divisible by 16')
        return bnbd.reshape(bn, heads, page//16, 16, groups, pair).permute(0, 1, 2, 4, 3, 5).contiguous()
    groups64, dimension, pair = bnbd.shape[2:5]
    if dimension % 16:
        raise ValueError('PA_NZ V scale requires head dimension divisible by 16')
    return bnbd.reshape(bn, heads, groups64, dimension//16, 16, pair).permute(0, 1, 3, 2, 4, 5).contiguous()


def effective_lengths(p):
    return p['seqused_q'] if p['seqused_q'] is not None else [
        b-a for a, b in zip(p['cu_seqlens_q'], p['cu_seqlens_q'][1:])]


def decode_sections(metadata, p):
    meta = metadata[0].tolist()
    if len(meta) < 16:
        raise ValueError('C4 metadata header is missing')
    sections, is_fd, m_base, task_n, cores = meta[:5]
    if sections < 1 or cores < 1 or m_base != 128 or task_n != (128 if p['D'] == 256 else 256):
        raise ValueError(f'Unsupported C4 metadata header: {meta[:5]}')
    if len(meta) < 16 + sections*cores*16:
        raise ValueError('C4 metadata FA records are truncated')
    head_num = p['N_kv'] if p['q_scale_layout'] == 'N2TGD' else p['N_q']
    group = p['N_q']//head_num
    qs, ks = effective_lengths(p), p['seqused_kv']
    counts = [(q*group+m_base-1)//m_base for q in qs]
    ranges = []
    for q, k, count in zip(qs, ks, counts):
        per_batch = []
        for mi in range(count):
            last_q = (min((mi+1)*m_base, q*group)-1)//group
            end = k if p['mask_mode'] == 0 else min(k, max(0, last_q+k-q+1))
            per_batch.append((0, (end+task_n-1)//task_n))
        ranges.append(per_batch)
    segments = [[[] for _ in range(counts[bn//head_num])] for bn in range(p['B']*head_num)]
    def advance(bn, mi):
        mi += 1
        if mi >= counts[bn//head_num]:
            bn, mi = bn+1, 0
        return bn, mi
    for section in range(sections):
        for core in range(cores):
            offset = 16+(section*cores+core)*16
            bn, mi, start, bn_end, mi_end, task_end = meta[offset:offset+6]
            end_state = (bn_end, mi_end, task_end)
            steps = 0
            while (bn, mi, start) != end_state:
                steps += 1
                if bn >= len(segments) or steps > 10_000_000:
                    raise ValueError('C4 metadata traversal exceeded its domain')
                if mi >= counts[bn//head_num]:
                    bn, mi = advance(bn, mi)
                    start = 0
                    continue
                lower, upper = ranges[bn//head_num][mi]
                if upper <= lower or start >= upper:
                    bn, mi = advance(bn, mi)
                    start = 0
                    continue
                stop = min(task_end, upper) if (bn, mi) == (bn_end, mi_end) else upper
                if stop <= start:
                    raise ValueError('C4 metadata contains a reversed segment')
                segments[bn][mi].append((start, stop))
                if stop >= upper:
                    bn, mi = advance(bn, mi)
                    start = 0
                else:
                    start = stop
    for bn, blocks in enumerate(segments):
        for mi, parts in enumerate(blocks):
            current, expected = ranges[bn//head_num][mi]
            for start, stop in parts:
                if start != current:
                    raise ValueError(f'Noncontiguous C4 segments: bn={bn}, mi={mi}')
                current = stop
            if current != expected:
                raise ValueError(f'Incomplete C4 segments: bn={bn}, mi={mi}, {current}!={expected}')
    return dict(m_base=m_base, task_n=task_n, head_num=head_num, group=group,
                segments=segments, is_fd=bool(is_fd))


def iter_tiles(p, schedule):
    qs = effective_lengths(p)
    for bn, blocks in enumerate(schedule['segments']):
        batch, head = divmod(bn, schedule['head_num'])
        for mi, parts in enumerate(blocks):
            if not parts:
                continue
            start = mi*schedule['m_base']
            stop = min(start+schedule['m_base'], qs[batch]*schedule['group'])
            yield batch, head, start, stop, parts
