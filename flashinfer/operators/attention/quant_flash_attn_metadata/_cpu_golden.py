# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python reference for C4A4 TND/PA SectionStreamK metadata on Ascend 950."""

from dataclasses import dataclass


_L2_BYTES = 96 * 1024 * 1024
_RECORD_SIZE = 16
_ZERO_RECORD = (0,) * _RECORD_SIZE
_ORIGIN = (0, 0, 0)


def _ceil_div(value, divisor):
    return (value + divisor - 1) // divisor


def _cost(rows, keys):
    return 6 * _ceil_div(rows, 16) + 10 * _ceil_div(keys, 64)


@dataclass(frozen=True)
class _Row:
    first: int
    stop: int
    cost: int
    normal_cost: int
    last_cost: int


_EMPTY_ROW = _Row(0, 0, 0, 0, 0)


class _Batch:
    def __init__(self, query, kv, group, m_tile, s2_tile, causal):
        self.query = query
        self.kv = kv
        self.group = group
        self.m_tile = m_tile
        self.s2_tile = s2_tile
        self.causal = causal
        self.m_size = query * group
        self.m_count = _ceil_div(self.m_size, m_tile)
        self.s2_count = _ceil_div(kv, s2_tile)
        self.rows = tuple(self._row(index) for index in range(self.m_count))
        self.cost = sum(row.cost for row in self.rows)
        self.blocks = sum(row.stop - row.first for row in self.rows)
        self.last_cost = next((row.last_cost for row in reversed(self.rows) if row.cost), 0)

    def row(self, index):
        return self.rows[index] if index < self.m_count else self._row(index)

    def _row(self, index):
        if not self.query or not self.kv:
            return _EMPTY_ROW
        first, stop = 0, self.s2_count
        m_first = index * self.m_tile
        if self.causal:
            q_first = m_first // self.group
            q_last = (min(m_first + self.m_tile, self.m_size) - 1) // self.group
            first_key = max(0, q_first - self.query)
            last_key = min(self.kv - 1, q_last + self.kv - self.query)
            if last_key < first_key:
                return _EMPTY_ROW
            first = first_key // self.s2_tile
            stop = last_key // self.s2_tile + 1
        m_size = self.m_tile
        if index == self.m_count - 1:
            m_size = self.m_size - m_first
        last_s2_size = self.s2_tile
        if stop == self.s2_count:
            last_s2_size = self.kv - (stop - 1) * self.s2_tile
        normal = _cost(m_size, self.s2_tile)
        last = _cost(m_size, last_s2_size)
        return _Row(first, stop, normal * (stop - first - 1) + last, normal, last)


def _section_ends(batches, dim, heads, kv_heads, m_tile, aic, causal):
    total_heads = len(batches) * heads
    if not any(batch.blocks for batch in batches):
        return [total_heads]
    # Each FP4 token costs D bytes for the Q/O or K/V pair in the L2 estimate.
    head_costs = [(min(batch.query, batch.kv) + batch.kv if causal else batch.query + batch.kv) * dim
                  for batch in batches]
    if max(batch.m_size for batch in batches) <= m_tile or max(head_costs) <= _L2_BYTES // aic:
        return [total_heads]

    ends = []
    section_bytes = 0
    for batch, head_cost in enumerate(head_costs):
        for kv_head in range(kv_heads):
            if section_bytes and section_bytes + head_cost > _L2_BYTES:
                ends.append(batch * heads + kv_head * (heads // kv_heads))
                section_bytes = 0
            section_bytes += head_cost
    return ends + [total_heads]


class _Cursor:
    def __init__(self, batches, heads, first, stop):
        self.batches = batches
        self.heads = heads
        self.head = first
        self.stop = stop
        self._load_head()

    @property
    def done(self):
        return self.head == self.stop

    @property
    def position(self):
        return self.head, self.tile, self.s2

    def _load_head(self):
        self.tile = self.s2 = 0
        if not self.done:
            self.batch = self.batches[self.head // self.heads]
            self.head_cost = self.batch.cost
            self._load_row()

    def _load_row(self):
        self.row = self.batch.row(self.tile)
        self.s2 = self.row.first
        self.row_cost = self.row.cost

    def advance_head(self):
        self.head += 1
        self._load_head()

    def advance_row(self):
        self.head_cost = max(0, self.head_cost - self.row_cost)
        self.tile += 1
        self._load_row()
        while self.row.first == self.row.stop:
            self.tile += 1
            self._load_row()

    def next_block_cost(self):
        return self.row.last_cost if self.s2 == self.row.stop - 1 else self.row.normal_cost


def _assign_fa(batches, heads, first, stop, core_count, total_cost, *, split_kv):
    cursor = _Cursor(batches, heads, first, stop)
    assignments, reductions = [], []
    max_cost = 0
    completed_slots, split_parts = 0, 1
    for core in range(core_count):
        if cursor.done or total_cost <= 0:
            break
        first_slot = completed_slots + split_parts - 1
        minimum = cursor.next_block_cost() if split_kv else cursor.row_cost
        limit = max(total_cost // (core_count - core), minimum)
        cost = 0

        while cursor.head_cost == 0 or cost + cursor.head_cost <= limit + cursor.batch.last_cost // 2:
            cost += cursor.head_cost
            cursor.advance_head()
            if cursor.done:
                break

        if not cursor.done:
            while cost + cursor.row_cost <= limit + cursor.row.last_cost // 2:
                cost += cursor.row_cost
                cursor.advance_row()
            if split_kv:
                block_cost = cursor.next_block_cost()
                # SectionStreamK charges the cost of the first block for this run.
                budget = limit + block_cost // 2 - cost
                if budget >= block_cost > 0:
                    blocks = budget // block_cost
                    cursor.s2 += blocks
                    allocated = blocks * block_cost
                    cost += allocated
                    cursor.head_cost -= allocated
                    cursor.row_cost -= allocated

        position = cursor.position
        if split_kv and assignments and split_parts > 1 and assignments[-1][0][:2] != position[:2]:
            head, tile, _ = assignments[-1][0]
            batch = batches[head // heads]
            rows = min(batch.m_tile, batch.m_size - tile * batch.m_tile)
            reductions.append((head, tile, completed_slots, split_parts, rows))
            completed_slots += split_parts
            split_parts = 1
        if not cursor.done and cursor.row.first < cursor.s2 <= cursor.row.stop:
            split_parts += 1
        assignments.append((position, first_slot))
        total_cost -= cost
        max_cost = max(max_cost, cost)
    return max_cost, assignments, reductions


def _assign_fd(reductions, aiv):
    if not reductions:
        return []
    average = _ceil_div(sum(count * rows for _, _, _, count, rows in reductions), aiv)
    free_cores = aiv - len(reductions)
    records = []
    for head, tile, slot, count, rows in reductions:
        requested = max(1, count * rows // average)
        rows_per_core = _ceil_div(rows, requested)
        cores = min(_ceil_div(rows, rows_per_core), free_cores + 1)
        for core in range(cores):
            start = core * rows_per_core
            length = rows - start if core == cores - 1 else rows_per_core
            records.append((head, tile, slot, count, start, length))
        free_cores -= cores - 1
    return records


def _schedule_section(batches, heads, first, stop, aic, aiv, full_block_cost):
    loads = [batches[head // heads] for head in range(first, stop)]
    blocks = sum(batch.blocks for batch in loads)
    if blocks == 0:
        return [((stop, 0, 0), 0)], []

    # The metadata operator sets the minimum and maximum AIC counts to aic.
    # SectionStreamK caps both by the number of blocks in the section.
    cores = min(aic, blocks)
    total_cost = sum(batch.cost for batch in loads)
    fa_cost, fa, _ = _assign_fa(batches, heads, first, stop, cores, total_cost, split_kv=False)
    split_cost, split_fa, reductions = _assign_fa(batches, heads, first, stop, cores, total_cost, split_kv=True)
    if fa_cost > 3 * full_block_cost and fa_cost - split_cost > 10 * full_block_cost:
        return split_fa, _assign_fd(reductions, aiv)
    return fa, []


def _record(fields):
    return fields + (0,) * (_RECORD_SIZE - len(fields))


def forward_metadata(*, dim, heads, kv_heads, q_lengths, kv_lengths, used_q,
                     qscale_layout, mask_mode, aic, aiv):
    """Compute the initialized forward row from the public metadata inputs."""
    effective_q = q_lengths if used_q is None else used_q
    m_tile = 64 * (aiv // aic)
    s2_tile = 128 if dim == 256 else 256
    scheduled_heads = kv_heads if qscale_layout == "N2TGD" else heads
    group = heads // scheduled_heads
    causal = mask_mode == 3
    batches = [_Batch(q, kv, group, m_tile, s2_tile, causal) for q, kv in zip(effective_q, kv_lengths)]
    ends = _section_ends(batches, dim, scheduled_heads, kv_heads, m_tile, aic, causal)
    full_block_cost = _cost(m_tile, s2_tile)
    sections = [_schedule_section(batches, scheduled_heads, first, stop, aic, aiv, full_block_cost)
                for first, stop in zip([0, *ends], ends)]

    need_init = any(q == 0 or q < stored or kv == 0 or (causal and q > kv)
                    for stored, q, kv in zip(q_lengths, effective_q, kv_lengths))
    header = [len(sections), int(any(fd for _, fd in sections)), m_tile, s2_tile, aic, aiv] + [0] * 10
    header[15] = int(need_init)
    fa_records, fd_records = [], []
    start = _ORIGIN
    for fa, fd in sections:
        for end, slot in fa:
            fa_records.append(_record((*start, *end, slot)))
            start = end
        fa_records.extend([_ZERO_RECORD] * (aic - len(fa)))
        fd_records.extend(_record(record) for record in fd)
        fd_records.extend([_ZERO_RECORD] * (aiv - len(fd)))
    return header + [value for record in (*fa_records, *fd_records) for value in record]
