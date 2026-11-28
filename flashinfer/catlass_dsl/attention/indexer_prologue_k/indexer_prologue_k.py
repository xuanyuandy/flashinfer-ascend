"""Ascend950 上 indexer K 路径 prologue 的 CATLASS Python DSL 实现。

同一条 stream 上执行两个 kernel：

1. ``ipk_projection``（Cube）：BF16 ``latent[T,H] @ wk[D,H]^T``，FP32 L0C 累加，
   回写为 BF16 ``projected[T,D]``（只做一次舍入）。
2. ``ipk_post``（Vector）：destination-owned RMSNorm + 尾部 RoPE + block-32
   MXFP4 打包 + paged cache 散射。

``T`` 在 GEMM 中是运行时标量（host 把 M 补齐到 tile 高度，保证 stride 都是编译期
常量），在 cache 几何中是动态 layout；编译 key 由模型形状 ``(H, D, Dr)`` 和存储
配置共同决定。
"""

from dataclasses import dataclass

import catlass.tla as tla

FP4_GROUP = 32
FP4_MAX = 6.0
E2M1_PER_BYTE = 2
FP32_MANT_MASK = 0x7FFFFF
FP32_EXP_SHIFT = 23
FP32_EXP_MASK = 0xFF


CUBE_CORES = 32


@dataclass(frozen=True)
class ProjTiling:
    bm: tla.Constexpr[int] = 128
    bn: tla.Constexpr[int] = 64
    bk: tla.Constexpr[int] = 128
    sk: tla.Constexpr[int] = 64
    split: tla.Constexpr[int] = 1


def _tiling_override():
    import os

    spec = os.environ.get("IPK_TILING")
    if not spec:
        return None
    parts = [int(x) for x in spec.split("/")]
    bm, bn, bk, sk = parts[:4]
    split = parts[4] if len(parts) > 4 else 1
    return ProjTiling(bm=bm, bn=bn, bk=bk, sk=sk, split=split)


def _ceil_div(value, step):
    return (value + step - 1) // step


def pick_tiling(total_rows, head_pad, index_head_dim):
    """按当前 shape 选择 GEMM tile 形状。

    两种情形：

    * 大 shape 沿用 ``bm/bn`` 阶梯，``split == 1``。
    * 小 shape（A5 indexer 场景为 T=96, D=128, K=512）受延迟约束：一波宽 tile
      （4..16 个 block）比占满 32 个 cube 核更快。该 shape 实测：16x16 的 48 个
      block -> 3.8us / cube 72%，16x64 的 12 个 block -> 2.58us / cube 28%，
      此场景时间优先。``split-K`` 仍可通过 ``IPK_TILING`` 打开（``split > 1``
      即其实现），用于以占用率优先的测量。
    """
    forced = _tiling_override()
    if forced is not None:
        return forced
    bm0 = max(16, _ceil_div(min(128, total_rows), 16) * 16)
    bn0 = index_head_dim if index_head_dim <= 128 else 128
    grid0 = _ceil_div(total_rows, bm0) * _ceil_div(index_head_dim, bn0)
    if grid0 >= CUBE_CORES or head_pad >= 1024 or total_rows > 512:
        if head_pad <= 512:
            bk = head_pad
            sk = 128 if bk >= 256 else (64 if bk >= 128 else bk // 2)
        else:
            bk, sk = 128, 64
        return ProjTiling(bm=bm0, bn=bn0, bk=bk, sk=sk)

    best = None
    for bn in (128, 64, 32, 16):
        if bn > index_head_dim:
            continue
        blocks = _ceil_div(total_rows, 16) * _ceil_div(index_head_dim, bn)
        if 8 <= blocks <= 32:
            best = (bn, blocks)
            break
    if best is not None:
        bn, _blocks = best
        bk, sk = (128, 64) if head_pad >= 128 else (head_pad, head_pad // 2)
        return ProjTiling(bm=16, bn=bn, bk=bk, sk=sk, split=1)
    bk, sk = (128, 64) if head_pad >= 128 else (head_pad, head_pad // 2)
    return ProjTiling(bm=bm0, bn=bn0, bk=bk, sk=sk, split=1)


@tla.kernel
def ipk_projection(
    latent: tla.Tensor,
    wk: tla.Tensor,
    projected: tla.Tensor,
    m_rows: int,
    _tiling: tla.Constexpr[ProjTiling],
) -> None:
    """projected[M, N] = latent[M, K] @ wk[N, K]^T，输入 BF16，L0C 内 FP32 累加。"""
    zero = 0
    one = 1
    bm = _tiling.bm
    bn = _tiling.bn
    bk = _tiling.bk
    sk = _tiling.sk
    split = _tiling.split

    dtype = latent.ptr.dtype
    m = m_rows
    ktot = latent.origin_shape[1]
    n = wk.origin_shape[1]
    nm = (m + bm - 1) // bm
    nn = (n + bn - 1) // bn
    kper = ktot // split
    nk = (kper + bk - 1) // bk
    total = nm * nn * split

    m_l10 = tla.mutex("proj_l1_0", id=0)
    m_l11 = tla.mutex("proj_l1_1", id=1)
    m_l00 = tla.mutex("proj_l0_0", id=2)
    m_l01 = tla.mutex("proj_l0_1", id=3)
    m_l0c = tla.mutex("proj_l0c", id=4)

    l1a0 = tla.allocate(bm * bk, dtype, tla.AddressSpace.l1, 512)
    l1a1 = tla.allocate(bm * bk, dtype, tla.AddressSpace.l1, 512)
    l1b0 = tla.allocate(bk * bn, dtype, tla.AddressSpace.l1, 512)
    l1b1 = tla.allocate(bk * bn, dtype, tla.AddressSpace.l1, 512)
    l0a0 = tla.allocate(bm * sk, dtype, tla.AddressSpace.l0a, 512)
    l0a1 = tla.allocate(bm * sk, dtype, tla.AddressSpace.l0a, 512)
    l0b0 = tla.allocate(sk * bn, dtype, tla.AddressSpace.l0b, 512)
    l0b1 = tla.allocate(sk * bn, dtype, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(bm * bn, tla.Float32, tla.AddressSpace.l0c, 1024)

    with tla.cube():
        for task in tla.range(tla.arch.block_idx(), total, tla.arch.block_num()):
            mt = task // (nn * split)
            nt = (task % (nn * split)) // split
            sp = task % split
            kbase = sp * nk
            og = tla.tile_view(
                projected,
                tla.make_shape(bm, bn),
                tla.make_coord(mt, sp * nn + nt),
            )
            l0_c = tla.make_tensor_like(l0c, og)

            ag_head = tla.tile_view(
                latent, tla.make_shape(bm, bk), tla.make_coord(mt, kbase)
            )
            bg_head = tla.tile_view(
                wk, tla.make_shape(bk, bn), tla.make_coord(kbase, nt)
            )
            al = tla.make_tensor_like(l1a0, ag_head)
            bl = tla.make_tensor_like(l1b0, bg_head)
            m_l10.lock(pipe=tla.arch.MTE2)
            tla.copy(al, ag_head)
            tla.copy(bl, bg_head)
            m_l10.unlock(pipe=tla.arch.MTE2)
            m_l0c.lock(pipe=tla.arch.CUBE)

            l1_phase = tla.as_numeric(zero)
            for kb in tla.range(zero, nk, one):
                if l1_phase == zero:
                    m_l10.lock(pipe=tla.arch.MTE1)
                else:
                    m_l11.lock(pipe=tla.arch.MTE1)

                if kb + one < nk:
                    ag_n = tla.tile_view(
                        latent,
                        tla.make_shape(bm, bk),
                        tla.make_coord(mt, kbase + kb + one),
                    )
                    bg_n = tla.tile_view(
                        wk,
                        tla.make_shape(bk, bn),
                        tla.make_coord(kbase + kb + one, nt),
                    )
                    na = l1a1 if l1_phase == zero else l1a0
                    nb = l1b1 if l1_phase == zero else l1b0
                    al_n = tla.make_tensor_like(na, ag_n)
                    bl_n = tla.make_tensor_like(nb, bg_n)
                    if l1_phase == zero:
                        m_l11.lock(pipe=tla.arch.MTE2)
                    else:
                        m_l10.lock(pipe=tla.arch.MTE2)
                    tla.copy(al_n, ag_n)
                    tla.copy(bl_n, bg_n)
                    if l1_phase == zero:
                        m_l11.unlock(pipe=tla.arch.MTE2)
                    else:
                        m_l10.unlock(pipe=tla.arch.MTE2)

                ca = l1a0 if l1_phase == zero else l1a1
                cb = l1b0 if l1_phase == zero else l1b1
                ag_c = tla.tile_view(
                    latent, tla.make_shape(bm, bk), tla.make_coord(mt, kbase + kb)
                )
                bg_c = tla.tile_view(
                    wk, tla.make_shape(bk, bn), tla.make_coord(kbase + kb, nt)
                )
                al_c = tla.make_tensor_like(ca, ag_c)
                bl_c = tla.make_tensor_like(cb, bg_c)

                aa = tla.tile_view(al_c, tla.make_shape(bm, sk), tla.make_coord(zero, zero))
                bb = tla.tile_view(bl_c, tla.make_shape(sk, bn), tla.make_coord(zero, zero))
                a0 = tla.make_tensor_like(l0a0, aa)
                b0 = tla.make_tensor_like(l0b0, bb)
                m_l00.lock(pipe=tla.arch.MTE1)
                tla.copy(a0, aa)
                tla.copy(b0, bb)
                m_l00.unlock(pipe=tla.arch.MTE1)

                l0_phase = tla.as_numeric(zero)
                for kk in tla.range(zero, bk // sk, one):
                    if l0_phase == zero:
                        m_l00.lock(pipe=tla.arch.CUBE)
                    else:
                        m_l01.lock(pipe=tla.arch.CUBE)

                    if kk + one < bk // sk:
                        aa_n = tla.tile_view(
                            al_c, tla.make_shape(bm, sk), tla.make_coord(zero, kk + one)
                        )
                        bb_n = tla.tile_view(
                            bl_c, tla.make_shape(sk, bn), tla.make_coord(kk + one, zero)
                        )
                        na0 = l0a1 if l0_phase == zero else l0a0
                        nb0 = l0b1 if l0_phase == zero else l0b0
                        a0_n = tla.make_tensor_like(na0, aa_n)
                        b0_n = tla.make_tensor_like(nb0, bb_n)
                        if l0_phase == zero:
                            m_l01.lock(pipe=tla.arch.MTE1)
                        else:
                            m_l00.lock(pipe=tla.arch.MTE1)
                        tla.copy(a0_n, aa_n)
                        tla.copy(b0_n, bb_n)
                        if l0_phase == zero:
                            m_l01.unlock(pipe=tla.arch.MTE1)
                        else:
                            m_l00.unlock(pipe=tla.arch.MTE1)

                    ca0 = l0a0 if l0_phase == zero else l0a1
                    cb0 = l0b0 if l0_phase == zero else l0b1
                    aa_c = tla.tile_view(
                        al_c, tla.make_shape(bm, sk), tla.make_coord(zero, kk)
                    )
                    bb_c = tla.tile_view(
                        bl_c, tla.make_shape(sk, bn), tla.make_coord(kk, zero)
                    )
                    a0_c = tla.make_tensor_like(ca0, aa_c)
                    b0_c = tla.make_tensor_like(cb0, bb_c)
                    is_last = (kb == nk - 1) and (kk == bk // sk - 1)
                    tla.mmad(
                        l0_c,
                        a0_c,
                        b0_c,
                        init_c=(kb == 0 and kk == 0),
                        unit_flag=0b11 if is_last else 0b10,
                    )
                    if l0_phase == zero:
                        m_l00.unlock(pipe=tla.arch.CUBE)
                    else:
                        m_l01.unlock(pipe=tla.arch.CUBE)
                    l0_phase = one - l0_phase

                if l1_phase == zero:
                    m_l10.unlock(pipe=tla.arch.MTE1)
                else:
                    m_l11.unlock(pipe=tla.arch.MTE1)
                l1_phase = one - l1_phase

            m_l0c.unlock(pipe=tla.arch.CUBE)
            m_l0c.lock(pipe=tla.arch.FIX)
            tla.copy(og, l0_c, tla.params.CopyL0C2DstParams(unit_flag=0b11))
            m_l0c.unlock(pipe=tla.arch.FIX)


@tla.kernel
def ipk_post(
    projected: tla.Tensor,
    rope_sin: tla.Tensor,
    rope_cos: tla.Tensor,
    gamma: tla.Tensor,
    cache_index: tla.Tensor,
    k_cache: tla.Tensor,
    k_scale_cache: tla.Tensor,
    epsilon: tla.Float32,
    rows_per_block: int,
    n_rows: int,
    _mode: tla.Constexpr[int],
    _group: tla.Constexpr[int],
    _head_dim: tla.Constexpr[int],
    _rope_dim: tla.Constexpr[int],
    _shift: tla.Constexpr[int],
    _write_scale: tla.Constexpr[bool],
    _split: tla.Constexpr[int],
) -> None:
    """destination-owned RMSNorm + RoPE + MXFP4 打包 + cache 散射。"""
    zero = 0
    one = 1
    dim = _head_dim
    rd = _rope_dim
    shift = _shift
    mode = _mode
    group = _group
    write_scale = _write_scale
    split = _split

    value_bytes = dim // E2M1_PER_BYTE
    scale_bytes = dim // FP4_GROUP
    scale_ub = (value_bytes + 63) // 32 * 32
    nchunk = (dim + 63) // 64
    ngroups = dim // FP4_GROUP
    rope_chunks = (rd + 63) // 64
    rope_start = dim - rd
    row_in = dim * split
    pipe_ub = row_in + 128

    m_row = tla.mutex("post_row", id=0)
    m_pack = tla.mutex("post_pack", id=2)
    m_gamma = tla.mutex("post_gamma", id=3)

    ub_in = tla.allocate(pipe_ub, tla.Float32, tla.AddressSpace.ub, 32)
    ub_key = tla.allocate(pipe_ub, tla.Float32, tla.AddressSpace.ub, 32)
    ub_gamma = tla.allocate(dim + 64, tla.Float32, tla.AddressSpace.ub, 32)
    ub_sin = tla.allocate(rd + 64, tla.Float32, tla.AddressSpace.ub, 32)
    ub_cos = tla.allocate(rd + 64, tla.Float32, tla.AddressSpace.ub, 32)
    ub_pack = tla.allocate(scale_ub + 64, tla.UInt8, tla.AddressSpace.ub, 32)

    ub_in_t = tla.make_tensor(
        ub_in,
        tla.make_layout(tla.make_shape(1, pipe_ub), tla.make_stride(pipe_ub, one)),
    )
    ub_key_t = tla.make_tensor(
        ub_key,
        tla.make_layout(tla.make_shape(1, pipe_ub), tla.make_stride(pipe_ub, one)),
    )
    ub_gamma_t = tla.make_tensor(
        ub_gamma,
        tla.make_layout(
            tla.make_shape(1, dim + 64), tla.make_stride(dim + 64, one)
        ),
    )
    ub_sin_t = tla.make_tensor(
        ub_sin,
        tla.make_layout(tla.make_shape(1, rd + 64), tla.make_stride(rd + 64, one)),
    )
    ub_cos_t = tla.make_tensor(
        ub_cos,
        tla.make_layout(tla.make_shape(1, rd + 64), tla.make_stride(rd + 64, one)),
    )
    ub_pack_t = tla.make_tensor(
        ub_pack,
        tla.make_layout(
            tla.make_shape(1, scale_ub + 64), tla.make_stride(scale_ub + 64, one)
        ),
    )

    total_rows = n_rows
    lane_id = tla.arch.block_idx()
    lane_num = tla.arch.block_num()

    with tla.vector():
        m_gamma.lock(pipe=tla.arch.MTE2)
        tla.copy(
            tla.get_tile(ub_gamma_t, tla.make_coord(zero, zero), tla.make_shape(1, dim)),
            tla.get_tile(gamma, tla.make_coord(zero, zero), tla.make_shape(1, dim)),
        )
        m_gamma.unlock(pipe=tla.arch.MTE2)
        m_gamma.lock(pipe=tla.arch.VECTOR)

        for row in tla.range(zero, total_rows, one):
            slot64 = cache_index[row]
            slot = slot64.to(tla.Int32)
            if slot >= zero:
                owner = tla.as_numeric(zero)
                block_size = rows_per_block * group
                block_id = tla.as_numeric(zero)
                in_block = tla.as_numeric(zero)
                combined = tla.as_numeric(zero)
                if mode == zero:
                    owner = slot % lane_num
                else:
                    block_id = slot // block_size
                    in_block = slot % block_size
                    combined = in_block // group
                    owner = (block_id * rows_per_block + combined) % lane_num
                if owner == lane_id:
                    m_row.lock(pipe=tla.arch.MTE2)
                    tla.copy(
                        tla.get_tile(
                            ub_in_t,
                            tla.make_coord(zero, shift),
                            tla.make_shape(1, row_in),
                        ),
                        tla.get_tile(
                            projected,
                            tla.make_coord(row, zero),
                            tla.make_shape(1, row_in),
                        ),
                    )
                    tla.copy(
                        tla.get_tile(
                            ub_sin_t,
                            tla.make_coord(zero, zero),
                            tla.make_shape(1, rd),
                        ),
                        tla.get_tile(
                            rope_sin,
                            tla.make_coord(row, zero),
                            tla.make_shape(1, rd),
                        ),
                    )
                    tla.copy(
                        tla.get_tile(
                            ub_cos_t,
                            tla.make_coord(zero, zero),
                            tla.make_shape(1, rd),
                        ),
                        tla.get_tile(
                            rope_cos,
                            tla.make_coord(row, zero),
                            tla.make_shape(1, rd),
                        ),
                    )
                    m_row.unlock(pipe=tla.arch.MTE2)

                    m_row.lock(pipe=tla.arch.VECTOR)
                    m_pack.lock(pipe=tla.arch.VECTOR)

                    with tla.vec.func(mode="simd"):
                        preg = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
                        for sp in tla.range_constexpr(1, split, 1):
                            for cwp in tla.range_constexpr(0, nchunk, 1):
                                off_a = shift + cwp * 64
                                acc_a = tla.add(
                                    tla.get_tile(
                                        ub_in_t,
                                        tla.make_coord(zero, off_a),
                                        tla.make_shape(1, 64),
                                    ).load(),
                                    tla.get_tile(
                                        ub_in_t,
                                        tla.make_coord(zero, dim * sp + off_a),
                                        tla.make_shape(1, 64),
                                    ).load(),
                                    mask=preg,
                                )
                                tla.get_tile(
                                    ub_in_t,
                                    tla.make_coord(zero, off_a),
                                    tla.make_shape(1, 64),
                                ).store(acc_a, mask=preg)
                        lane_idx = tla.arange(
                            base=0, order="increase", dtype=tla.Int32
                        )
                        pm_lo = tla.cmp(lane_idx, 32, "lt")
                        pm_hi = tla.cmp(lane_idx, 32, "ge")
                        pm32f, _ = tla.update_mask(FP4_GROUP, dtype=tla.Float32)
                        pm16i, _ = tla.update_mask(
                            FP4_GROUP // E2M1_PER_BYTE, dtype=tla.Int32
                        )
                        pm32i, _ = tla.update_mask(2 * FP4_GROUP, dtype=tla.Int32)
                        cast_rint = tla.params.CastParams(
                            reg_slot=tla.params.RegSlot.ZERO,
                            sat_mode=tla.params.SatMode.NOSAT,
                            round_mode=tla.params.RoundMode.CAST_RINT,
                        )
                        cast_trunc = tla.params.CastParams(
                            reg_slot=tla.params.RegSlot.ZERO,
                            sat_mode=tla.params.SatMode.NOSAT,
                            round_mode=tla.params.RoundMode.CAST_TRUNC,
                        )
                        cast_widen = tla.params.CastParams(
                            reg_slot=tla.params.RegSlot.ZERO,
                            sat_mode=tla.params.SatMode.NOSAT,
                        )
                        pack_b8 = tla.params.NormalStoreParams(
                            store_dist=tla.params.StoreDist.DIST_PACK4_B32
                        )
                        first_b8 = tla.params.NormalStoreParams(
                            store_dist=tla.params.StoreDist.DIST_FIRST_ELEMENT_B8
                        )
                        zero_f = tla.full(0.0, tla.Float32)
                        six_f = tla.full(FP4_MAX, tla.Float32)
                        min_scale = tla.full(FP4_MAX * (2.0**-126), tla.Float32)
                        zero_i = tla.full(zero, tla.Int32)
                        one_i = tla.full(one, tla.Int32)

                        for cw in tla.range_constexpr(0, nchunk, 1):
                            off_w = shift + cw * 64
                            vals_w = tla.get_tile(
                                ub_in_t,
                                tla.make_coord(zero, off_w),
                                tla.make_shape(1, 64),
                            ).load()
                            f_w = vals_w.to(tla.BFloat16, cast_rint, preg).to(
                                tla.Float32, cast_widen
                            )
                            tla.get_tile(
                                ub_key_t,
                                tla.make_coord(zero, off_w),
                                tla.make_shape(1, 64),
                            ).store(f_w, mask=preg)

                        acc = zero_f
                        for ci in tla.range_constexpr(0, nchunk, 1):
                            off_i = shift + ci * 64
                            f_i = tla.get_tile(
                                ub_key_t,
                                tla.make_coord(zero, off_i),
                                tla.make_shape(1, 64),
                            ).load()
                            if ci * 64 + 64 > dim:
                                valid_i = dim - ci * 64
                                pm_i, _ = tla.update_mask(valid_i, dtype=tla.Float32)
                                f_i = tla.where(pm_i, f_i, zero_f)
                            acc = tla.add(acc, tla.mul(f_i, f_i, mask=preg), mask=preg)
                        total = acc.reduce(tla.ReductionOp.ADD, mask=preg)
                        mean_v = tla.mul(
                            tla.full(total, tla.Float32),
                            tla.full(1.0 / dim, tla.Float32),
                            mask=preg,
                        )
                        root_v = tla.sqrt(
                            tla.add(
                                mean_v, tla.full(epsilon, tla.Float32), mask=preg
                            ),
                            mask=preg,
                        )
                        inv_v = tla.div(
                            tla.full(1.0, tla.Float32), root_v, mask=preg
                        )

                        for cn in tla.range_constexpr(0, nchunk, 1):
                            off_n = shift + cn * 64
                            f_n = tla.get_tile(
                                ub_key_t,
                                tla.make_coord(zero, off_n),
                                tla.make_shape(1, 64),
                            ).load()
                            g_n = tla.get_tile(
                                ub_gamma_t,
                                tla.make_coord(zero, cn * 64),
                                tla.make_shape(1, 64),
                            ).load()
                            y_n = tla.mul(
                                tla.mul(f_n, inv_v, mask=preg), g_n, mask=preg
                            )
                            y_bf = y_n.to(tla.BFloat16, cast_rint, preg)
                            tla.get_tile(
                                ub_key_t,
                                tla.make_coord(zero, off_n),
                                tla.make_shape(1, 64),
                            ).store(y_bf.to(tla.Float32, cast_widen), mask=preg)

                        for cr in tla.range_constexpr(0, rope_chunks, 1):
                            off_r = shift + rope_start + cr * 64
                            xf_r = tla.get_tile(
                                ub_key_t,
                                tla.make_coord(zero, off_r),
                                tla.make_shape(1, 64),
                            ).load()
                            x_even, x_odd = tla.deinterleave(xf_r, xf_r)
                            s_r = tla.get_tile(
                                ub_sin_t,
                                tla.make_coord(zero, cr * 64),
                                tla.make_shape(1, 64),
                            ).load()
                            c_r = tla.get_tile(
                                ub_cos_t,
                                tla.make_coord(zero, cr * 64),
                                tla.make_shape(1, 64),
                            ).load()
                            s_even, s_odd = tla.deinterleave(s_r, s_r)
                            c_even, c_odd = tla.deinterleave(c_r, c_r)
                            out_even = tla.sub(
                                tla.mul(x_even, c_even, mask=preg),
                                tla.mul(x_odd, s_even, mask=preg),
                                mask=preg,
                            )
                            out_odd = tla.add(
                                tla.mul(x_odd, c_odd, mask=preg),
                                tla.mul(x_even, s_odd, mask=preg),
                                mask=preg,
                            )
                            rot, rot_hi = tla.interleave(out_even, out_odd)
                            rot_bf = rot.to(tla.BFloat16, cast_rint, preg)
                            tla.get_tile(
                                ub_key_t,
                                tla.make_coord(zero, off_r),
                                tla.make_shape(1, 64),
                            ).store(
                                rot_bf.to(tla.Float32, cast_widen), mask=preg
                            )

                        scale_vec = zero_i
                        for pi in tla.range_constexpr(0, ngroups // 2, 1):
                            off_p = shift + pi * 64
                            f_p = tla.get_tile(
                                ub_key_t,
                                tla.make_coord(zero, off_p),
                                tla.make_shape(1, 64),
                            ).load()
                            abs_p = tla.abs(f_p)
                            amax_lo = tla.where(pm_lo, abs_p, zero_f).reduce(
                                tla.ReductionOp.MAX, mask=preg
                            )
                            amax_hi = tla.where(pm_hi, abs_p, zero_f).reduce(
                                tla.ReductionOp.MAX, mask=preg
                            )
                            safe_lo = tla.max(
                                tla.full(amax_lo, tla.Float32),
                                min_scale,
                                mask=preg,
                            )
                            safe_hi = tla.max(
                                tla.full(amax_hi, tla.Float32),
                                min_scale,
                                mask=preg,
                            )
                            raw_lo = tla.div(safe_lo, six_f, mask=preg)
                            raw_hi = tla.div(safe_hi, six_f, mask=preg)
                            bits_lo = raw_lo.bitcast(tla.Int32)
                            bits_hi = raw_hi.bitcast(tla.Int32)
                            expo_lo = tla.bitwise_and(
                                tla.shift_right(bits_lo, FP32_EXP_SHIFT),
                                tla.full(FP32_EXP_MASK, tla.Int32),
                            )
                            expo_hi = tla.bitwise_and(
                                tla.shift_right(bits_hi, FP32_EXP_SHIFT),
                                tla.full(FP32_EXP_MASK, tla.Int32),
                            )
                            mant_lo = tla.bitwise_and(
                                bits_lo, tla.full(FP32_MANT_MASK, tla.Int32)
                            )
                            mant_hi = tla.bitwise_and(
                                bits_hi, tla.full(FP32_MANT_MASK, tla.Int32)
                            )
                            expo_lo = tla.add(
                                expo_lo,
                                tla.where(
                                    tla.cmp(mant_lo, zero_i, "ne", mask=preg),
                                    one_i,
                                    zero_i,
                                ),
                                mask=preg,
                            )
                            expo_hi = tla.add(
                                expo_hi,
                                tla.where(
                                    tla.cmp(mant_hi, zero_i, "ne", mask=preg),
                                    one_i,
                                    zero_i,
                                ),
                                mask=preg,
                            )
                            expo = tla.where(pm_lo, expo_lo, expo_hi)
                            scale_v = tla.shift_left(
                                expo, FP32_EXP_SHIFT, mask=preg
                            ).bitcast(tla.Float32)
                            mag = tla.min(
                                tla.div(abs_p, scale_v, mask=preg), six_f, mask=preg
                            )
                            code = tla.full(7, tla.Int32)
                            for thr, low in (
                                (5.0, 6),
                                (3.5, 5),
                                (2.5, 4),
                                (1.75, 3),
                                (1.25, 2),
                                (0.75, 1),
                                (0.25, 0),
                            ):
                                hit = tla.cmp(mag, thr, "le", mask=preg)
                                code = tla.where(hit, tla.full(low, tla.Int32), code)
                            fbits = f_p.bitcast(tla.Int32)
                            sign = tla.shift_left(
                                tla.bitwise_and(
                                    tla.shift_right(fbits, 31),
                                    tla.full(one, tla.Int32),
                                ),
                                3,
                            )
                            code = tla.bitwise_or(code, sign)
                            byte_even, byte_odd = tla.deinterleave(code, code)
                            packed = tla.bitwise_or(
                                byte_even, tla.shift_left(byte_odd, 4)
                            )
                            tla.get_tile(
                                ub_pack_t,
                                tla.make_coord(zero, pi * 32),
                                tla.make_shape(1, 32),
                            ).store(packed, pack_b8, mask=pm32i)
                            scale_vec = tla.where(
                                tla.cmp(lane_idx, pi * 2, "eq", mask=preg),
                                expo_lo,
                                scale_vec,
                            )
                            scale_vec = tla.where(
                                tla.cmp(lane_idx, pi * 2 + 1, "eq", mask=preg),
                                expo_hi,
                                scale_vec,
                            )

                        if ngroups % 2 == 1:
                            off_t = shift + (ngroups - 1) * FP4_GROUP
                            f_t = tla.get_tile(
                                ub_key_t,
                                tla.make_coord(zero, off_t),
                                tla.make_shape(1, 64),
                            ).load()
                            abs_t = tla.abs(f_t, mask=pm32f)
                            amax_t = abs_t.reduce(tla.ReductionOp.MAX, mask=pm32f)
                            safe_t = tla.max(
                                tla.full(amax_t, tla.Float32), min_scale, mask=pm32f
                            )
                            raw_t = tla.div(safe_t, six_f, mask=pm32f)
                            bits_t = raw_t.bitcast(tla.Int32)
                            expo_t = tla.bitwise_and(
                                tla.shift_right(bits_t, FP32_EXP_SHIFT),
                                tla.full(FP32_EXP_MASK, tla.Int32),
                            )
                            mant_t = tla.bitwise_and(
                                bits_t, tla.full(FP32_MANT_MASK, tla.Int32)
                            )
                            expo_t = tla.add(
                                expo_t,
                                tla.where(
                                    tla.cmp(mant_t, zero_i, "ne", mask=pm32f),
                                    one_i,
                                    zero_i,
                                ),
                                mask=pm32f,
                            )
                            scale_t = tla.shift_left(
                                expo_t, FP32_EXP_SHIFT, mask=pm32f
                            ).bitcast(tla.Float32)
                            mag_t = tla.min(
                                tla.div(abs_t, scale_t, mask=pm32f),
                                six_f,
                                mask=pm32f,
                            )
                            code_t = tla.full(7, tla.Int32)
                            for thr2, low2 in (
                                (5.0, 6),
                                (3.5, 5),
                                (2.5, 4),
                                (1.75, 3),
                                (1.25, 2),
                                (0.75, 1),
                                (0.25, 0),
                            ):
                                hit_t = tla.cmp(mag_t, thr2, "le", mask=pm32f)
                                code_t = tla.where(
                                    hit_t, tla.full(low2, tla.Int32), code_t
                                )
                            fbits_t = f_t.bitcast(tla.Int32)
                            sign_t = tla.shift_left(
                                tla.bitwise_and(
                                    tla.shift_right(fbits_t, 31),
                                    tla.full(one, tla.Int32),
                                ),
                                3,
                            )
                            code_t = tla.bitwise_or(code_t, sign_t)
                            even_t, odd_t = tla.deinterleave(code_t, code_t)
                            packed_t = tla.bitwise_or(
                                even_t, tla.shift_left(odd_t, 4)
                            )
                            tla.get_tile(
                                ub_pack_t,
                                tla.make_coord(zero, (ngroups // 2) * 32),
                                tla.make_shape(1, 16),
                            ).store(packed_t, pack_b8, mask=pm16i)
                            scale_vec = tla.where(
                                tla.cmp(lane_idx, ngroups - 1, "eq", mask=preg),
                                expo_t,
                                scale_vec,
                            )

                        tla.get_tile(
                            ub_pack_t,
                            tla.make_coord(zero, scale_ub),
                            tla.make_shape(1, 32),
                        ).store(scale_vec, pack_b8, mask=pm32i)

                    m_pack.unlock(pipe=tla.arch.VECTOR)
                    m_row.unlock(pipe=tla.arch.VECTOR)

                    m_pack.lock(pipe=tla.arch.MTE3)
                    data_off = tla.as_numeric(zero)
                    scale_off = tla.as_numeric(zero)
                    if mode == zero:
                        data_off = slot * value_bytes
                        scale_off = slot * scale_bytes
                    else:
                        width = group * (value_bytes + scale_bytes)
                        block_size = rows_per_block * group
                        block_id = slot // block_size
                        in_block = slot % block_size
                        combined = in_block // group
                        sub_slot = in_block % group
                        base = (block_id * rows_per_block + combined) * width
                        data_off = base + sub_slot * value_bytes
                        scale_off = base + group * value_bytes + sub_slot * scale_bytes

                    gm_data = tla.make_tensor(
                        k_cache.ptr + data_off.to(tla.Int64),
                        tla.make_layout(
                            tla.make_shape(1, value_bytes),
                            tla.make_stride(value_bytes, one),
                        ),
                    )
                    gm_scale = tla.make_tensor(
                        k_scale_cache.ptr + scale_off.to(tla.Int64),
                        tla.make_layout(
                            tla.make_shape(1, scale_bytes),
                            tla.make_stride(scale_bytes, one),
                        ),
                    )
                    if mode != zero:
                        gm_scale = tla.make_tensor(
                            k_cache.ptr + scale_off.to(tla.Int64),
                            tla.make_layout(
                                tla.make_shape(1, scale_bytes),
                                tla.make_stride(scale_bytes, one),
                            ),
                        )

                    tla.copy(
                        gm_data,
                        tla.get_tile(
                            ub_pack_t,
                            tla.make_coord(zero, zero),
                            tla.make_shape(1, value_bytes),
                        ),
                    )
                    if mode == zero:
                        if write_scale:
                            tla.copy(
                                gm_scale,
                                tla.get_tile(
                                    ub_pack_t,
                                    tla.make_coord(zero, scale_ub),
                                    tla.make_shape(1, scale_bytes),
                                ),
                            )
                    else:
                        tla.copy(
                            gm_scale,
                            tla.get_tile(
                                ub_pack_t,
                                tla.make_coord(zero, scale_ub),
                                tla.make_shape(1, scale_bytes),
                            ),
                        )
                    m_pack.unlock(pipe=tla.arch.MTE3)
