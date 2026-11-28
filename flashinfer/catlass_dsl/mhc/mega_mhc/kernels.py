"""Fused residual mix, mix-coefficient update and RMSNorm projection for MHC.

The operator is split into four CATLASS DSL kernels:

* K0 casts the FP32 projection matrix ``fn`` to BF16 when it changes.
* K1 mixes the four residual streams, produces ``u`` and the RMS statistics.
* K2 computes the ``[24, 4H] @ [4H, T]`` projection with split-K on the cube.
* K3 finishes the coefficients (sigmoid / softmax / Sinkhorn) and the RMSNorm.

K1 splits a token into two chunks of ``H/2`` columns, so both halves can live in
one core's UB at the same time.  Tokens fully owned by one core ("hot") finish
``y = u * gv * w`` in K1 without a GM round trip; tokens straddling a core
boundary ("cold") leave ``u`` in GM for K3.  The RMS sums of a token are
therefore written as per-core slot partials (``slot = block_idx % NSLOT``) and
reduced by K3.

Invariants required by the DSL:

* every ``tla.copy`` keeps one rank per (dtype, direction, layout), otherwise the
  generated extern copy helpers collide; this file uses rank-2 everywhere;
* static expansion uses ``tla.range_constexpr`` and dynamic loop variables never
  repeat inside one kernel;
* ``tile.load/store`` only touch UB and GM goes through ``tla.copy``;
  SIMD-to-SIMT exchange goes through UB.
"""

from __future__ import annotations

import catlass.tla as tla

# ---------------------------------------------------------------------------
# Constants (kept in sync with the CPU reference)
# ---------------------------------------------------------------------------
MF = 24            # fn rows: 4(pre) + 4(post) + 16(comb)
VL = 64            # float32 vector register lanes
CH1 = 1024         # h elements per K1 transfer (larger DMA granularity)
BM_G = 32          # K2 cube M tile (MF=24 rounded up)
BN_G = 128         # K2 cube N tile
SK_G = 64          # K2 L0 K step
BK_G = 256         # K2 L1 K tile (2x L1A + 2x L1B = 160KB < 256KB budget)
SPLIT_MAX = 32     # maximum split-K parts (one per cube core)
NBLK_VEC = 64      # AIV cores (K1/K3 block_num)
NSLOT = 8          # zz/su partial slots: one token spans at most M cores (M = H/CH <= 7),
                   # and each core writes its own slot (slot = block_idx % NSLOT, adjacent
                   # cores differ), so no atomics and no per-call clear are needed; slots
                   # that are never written are zero from allocation.
TSPLIT = 8         # K3 coefficient transpose: TSPLIT SIMT threads per token (MF % TSPLIT == 0)

_CAST_EVEN = tla.params.CastParams(
    reg_slot=tla.params.RegSlot.ZERO,
    sat_mode=tla.params.SatMode.NOSAT,
    round_mode=tla.params.RoundMode.CAST_ROUND,
)
_CAST_NONE = tla.params.CastParams()
_LOAD_UP16 = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_UNPACK_B16)
_STORE_PK32 = tla.params.NormalStoreParams(
    store_dist=tla.params.StoreDist.DIST_PACK_B32
)
_BRC = tla.params.NormalLoadParams(load_dist=tla.params.LoadDist.DIST_BRC_B32)


def _t2d(ptr, rows, cols):
    return tla.make_tensor(
        ptr, tla.make_layout(tla.make_shape(rows, cols), tla.make_stride(cols, 1))
    )


def _bf(tile, row, grp):
    """Load 64 bf16 values from UB as 64 FP32 lanes."""
    return (
        tla.tile_view(tile, tla.make_shape(1, VL), tla.make_coord(row, grp))
        .load(_LOAD_UP16)
        .to(tla.Float32, _CAST_NONE)
    )


def _st_bf(tile, row, grp, v):
    tla.tile_view(tile, tla.make_shape(1, VL), tla.make_coord(row, grp)).store(
        v.to(tla.BFloat16, _CAST_EVEN), _STORE_PK32
    )


def _ld1(tile, row):
    return tla.tile_view(
        tile, tla.make_shape(1, VL), tla.make_coord(row, 0)
    ).load()


def _brc(tile, idx):
    return tla.tile_view(tile, tla.make_shape(1, 1), tla.make_coord(0, idx)).load(
        params=_BRC
    )


def _mix_n(xsrc, rsrc, ub_r, ub_u, ub_comb, ub_post, ub_pm, gp, uoff, zz, uu, NG):
    """Mix NG adjacent 64-lane groups of one token, reusing the constant broadcasts.

    Keeping the constants alive only inside the NG-group body avoids holding them in
    vector registers across a whole chunk; ``_mix_n(NG=2)`` halves their reloads.

    ``u`` is written into the token-level UB tile ``ub_u`` (1, H) at the dynamic
    column offset ``uoff + gp * NG`` instead of going through GM.
    """
    g0 = gp * NG
    xv = [_bf(xsrc, 0, g0 + q) for q in tla.range_constexpr(NG)]
    r0 = [_bf(rsrc, 0, g0 + q) for q in tla.range_constexpr(NG)]
    r1 = [_bf(rsrc, 1, g0 + q) for q in tla.range_constexpr(NG)]
    r2 = [_bf(rsrc, 2, g0 + q) for q in tla.range_constexpr(NG)]
    r3 = [_bf(rsrc, 3, g0 + q) for q in tla.range_constexpr(NG)]
    ua = [None] * NG
    for j in tla.range_constexpr(4):
        pj = _brc(ub_post, j)
        a = [tla.mul(xv[q], pj) for q in tla.range_constexpr(NG)]
        for q in tla.range_constexpr(NG):
            a[q] = tla.add(a[q], tla.mul(r0[q], _brc(ub_comb, 0 + j)))
        for q in tla.range_constexpr(NG):
            a[q] = tla.add(a[q], tla.mul(r1[q], _brc(ub_comb, 4 + j)))
        for q in tla.range_constexpr(NG):
            a[q] = tla.add(a[q], tla.mul(r2[q], _brc(ub_comb, 8 + j)))
        for q in tla.range_constexpr(NG):
            a[q] = tla.add(a[q], tla.mul(r3[q], _brc(ub_comb, 12 + j)))
        bv = [a[q].to(tla.BFloat16, _CAST_EVEN) for q in tla.range_constexpr(NG)]
        for q in tla.range_constexpr(NG):
            tla.tile_view(
                ub_r, tla.make_shape(1, VL), tla.make_coord(j, g0 + q)
            ).store(bv[q], _STORE_PK32)
        fv = [bv[q].to(tla.Float32, _CAST_NONE) for q in tla.range_constexpr(NG)]
        s = tla.mul(fv[0], fv[0])
        for q in tla.range_constexpr(1, NG):
            s = tla.add(s, tla.mul(fv[q], fv[q]))
        zz = tla.add(zz, s)
        pmj = _brc(ub_pm, j)
        if j == 0:
            for q in tla.range_constexpr(NG):
                ua[q] = tla.mul(fv[q], pmj)
        else:
            for q in tla.range_constexpr(NG):
                ua[q] = tla.add(ua[q], tla.mul(fv[q], pmj))
    for q in tla.range_constexpr(NG):
        av = ua[q].to(tla.BFloat16, _CAST_EVEN)
        tla.tile_view(ub_u, tla.make_shape(1, VL), tla.make_coord(0, uoff + g0 + q)).store(
            av, _STORE_PK32
        )
        fa = av.to(tla.Float32, _CAST_NONE)
        uu = tla.add(uu, tla.mul(fa, fa))
    return zz, uu


def _mix_group(xsrc, rsrc, ub_r, ub_u, ub_comb, ub_post, ub_pm, gpa, uoff, zz, uu):
    """One 64-lane group: four ``r`` outputs plus ``u`` and the ``zz``/``uu`` sums.

    The four ``j`` outputs are independent chains and ``zz``/``uu`` accumulate
    serially across them.  Broadcasting the comb constants once per chunk instead of
    per group is slower because they then occupy vector registers for too long, so
    the reuse is done through ``_mix_n``.  This single-group form is used for the
    odd tail of ``CH // VL`` and for the latency-bound small-T path.
    """
    xv = _bf(xsrc, 0, gpa)
    r0 = _bf(rsrc, 0, gpa)
    r1 = _bf(rsrc, 1, gpa)
    r2 = _bf(rsrc, 2, gpa)
    r3 = _bf(rsrc, 3, gpa)
    for j in tla.range_constexpr(4):
        a = tla.mul(xv, _brc(ub_post, j))
        a = tla.add(a, tla.mul(r0, _brc(ub_comb, 0 + j)))
        a = tla.add(a, tla.mul(r1, _brc(ub_comb, 4 + j)))
        a = tla.add(a, tla.mul(r2, _brc(ub_comb, 8 + j)))
        a = tla.add(a, tla.mul(r3, _brc(ub_comb, 12 + j)))
        b = a.to(tla.BFloat16, _CAST_EVEN)
        tla.tile_view(ub_r, tla.make_shape(1, VL), tla.make_coord(j, gpa)).store(
            b, _STORE_PK32
        )
        f = b.to(tla.Float32, _CAST_NONE)
        zz = tla.add(zz, tla.mul(f, f))
        if j == 0:
            ua = tla.mul(f, _brc(ub_pm, 0))
        else:
            ua = tla.add(ua, tla.mul(f, _brc(ub_pm, j)))
    b = ua.to(tla.BFloat16, _CAST_EVEN)
    tla.tile_view(ub_u, tla.make_shape(1, VL), tla.make_coord(0, uoff + gpa)).store(
        b, _STORE_PK32
    )
    f = b.to(tla.Float32, _CAST_NONE)
    uu = tla.add(uu, tla.mul(f, f))
    return zz, uu


def _red_ss(ub_ssz, ub_ssu, ub_s):
    """Reduce the 64-lane zz/su accumulators to two scalars in ub_s; return su."""
    tla.local_mem_bar(
        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
    )
    allm = tla.create_mask(pattern=tla.mask.ALL, dtype=tla.Float32)
    one_m, _ = tla.update_mask(1, dtype=tla.Float32)
    sz = tla.tile_view(
        ub_ssz, tla.make_shape(1, VL), tla.make_coord(0, 0)
    ).load().reduce(tla.ReductionOp.ADD, mask=allm)
    su = tla.tile_view(
        ub_ssu, tla.make_shape(1, VL), tla.make_coord(0, 0)
    ).load().reduce(tla.ReductionOp.ADD, mask=allm)
    tla.tile_view(ub_s, tla.make_shape(1, 1), tla.make_coord(0, 0)).store(
        tla.full(sz, dtype=tla.Float32),
        tla.params.UnalignStoreParams(),
        mask=one_m,
    )
    tla.tile_view(ub_s, tla.make_shape(1, 1), tla.make_coord(0, 1)).store(
        tla.full(su, dtype=tla.Float32),
        tla.params.UnalignStoreParams(),
        mask=one_m,
    )
    return su


# ---------------------------------------------------------------------------
# K0: fn [24, 4H] f32 to bf16 (only when fn changes)
# ---------------------------------------------------------------------------
@tla.kernel
def cast_kernel(
    gm_src: tla.Tensor,      # [1, N] f32
    gm_dst: tla.Tensor,      # [1, N] bf16
    N: tla.Constexpr[int],
    CHK: tla.Constexpr[int],
) -> None:
    ub_in = _t2d(tla.allocate(CHK, tla.Float32, tla.AddressSpace.ub, 256), 1, CHK)
    ub_out = _t2d(tla.allocate(CHK, tla.BFloat16, tla.AddressSpace.ub, 256), 1, CHK)
    m_c = tla.mutex("k0_c", id=0)
    nblk = tla.arch.block_num()
    bid = tla.arch.block_idx()

    with tla.vector():
        for it in tla.range(bid, N // CHK, nblk):
            m_c.lock(pipe=tla.arch.MTE2)
            tla.copy(
                ub_in,
                tla.tile_view(gm_src, tla.make_shape(1, CHK), tla.make_coord(0, it)),
            )
            m_c.unlock(pipe=tla.arch.MTE2)
            m_c.lock(pipe=tla.arch.VECTOR)
            with tla.vec.func(mode="simd"):
                for gc in tla.range_constexpr(CHK // VL):
                    fv = tla.tile_view(
                        ub_in, tla.make_shape(1, VL), tla.make_coord(0, gc)
                    ).load()
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(0, gc)
                    ).store(fv.to(tla.BFloat16, _CAST_EVEN), _STORE_PK32)
            m_c.unlock(pipe=tla.arch.VECTOR)
            m_c.lock(pipe=tla.arch.MTE3)
            tla.copy(
                tla.tile_view(gm_dst, tla.make_shape(1, CHK), tla.make_coord(0, it)),
                ub_out,
            )
            m_c.unlock(pipe=tla.arch.MTE3)


# ---------------------------------------------------------------------------
# K1: residual mix + u + partial sums (token x h range split evenly)
# ---------------------------------------------------------------------------
@tla.kernel
def mix_kernel(
    gm_x: tla.Tensor,        # [T, H] bf16
    gm_res: tla.Tensor,      # [T*4, H] bf16
    gm_pm: tla.Tensor,       # [T, 4] f32
    gm_post: tla.Tensor,     # [T, 4] f32
    gm_comb: tla.Tensor,     # [T, 16] f32
    gm_new_res: tla.Tensor,  # [T*4, H] bf16
    gm_w: tla.Tensor,        # [1, H] bf16
    gm_y: tla.Tensor,        # [T, H] bf16 (hot tokens get y here, cold tokens get u for K3)
    gm_pzz: tla.Tensor,      # [NSLOT, TPAD] f32 (zz partial sums per core slot)
    gm_puu: tla.Tensor,      # [TPAD, NSLOT] f32 (u squared partial sums per core slot)
    gm_partf: tla.Tensor,    # [1, PN] f32 (K2 atomic accumulation target, cleared here)
    T: tla.Constexpr[int],
    H: tla.Constexpr[int],
    CH: tla.Constexpr[int],
    M: tla.Constexpr[int],
    PN: tla.Constexpr[int],
    SPLIT: tla.Constexpr[int],   # 1 = even chunk split (small T), 0 = whole-token assignment
) -> None:
    nblk = tla.arch.block_num()
    bid = tla.arch.block_idx()

    ub_res = _t2d(tla.allocate(4 * CH, tla.BFloat16, tla.AddressSpace.ub, 256), 4, CH)
    ub_x = _t2d(tla.allocate(CH, tla.BFloat16, tla.AddressSpace.ub, 256), 1, CH)
    ub_res2 = _t2d(tla.allocate(4 * CH, tla.BFloat16, tla.AddressSpace.ub, 256), 4, CH)
    ub_x2 = _t2d(tla.allocate(CH, tla.BFloat16, tla.AddressSpace.ub, 256), 1, CH)
    ub_r = _t2d(tla.allocate(4 * CH, tla.BFloat16, tla.AddressSpace.ub, 256), 4, CH)
    # u lives in a chunk-sized buffer at a static column offset and is double buffered
    # together with r.  A token-sized buffer with a dynamic column offset is not an
    # option: a dynamic dim-1 coordinate makes the packed store take a slow path.
    ub_uv = _t2d(tla.allocate(CH, tla.BFloat16, tla.AddressSpace.ub, 256), 1, CH)
    ub_uv2 = _t2d(tla.allocate(CH, tla.BFloat16, tla.AddressSpace.ub, 256), 1, CH)
    ub_r2 = _t2d(tla.allocate(4 * CH, tla.BFloat16, tla.AddressSpace.ub, 256), 4, CH)
    ub_w = _t2d(tla.allocate(H, tla.BFloat16, tla.AddressSpace.ub, 256), 1, H)
    ub_yall = _t2d(tla.allocate(H, tla.BFloat16, tla.AddressSpace.ub, 256), 1, H)
    ub_comb = _t2d(tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256), 1, 64)
    ub_post = _t2d(tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256), 1, 64)
    ub_pm = _t2d(tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256), 1, 64)
    ub_ssz = _t2d(tla.allocate(VL, tla.Float32, tla.AddressSpace.ub, 256), 1, VL)
    ub_ssu = _t2d(tla.allocate(VL, tla.Float32, tla.AddressSpace.ub, 256), 1, VL)
    ub_s = _t2d(tla.allocate(VL, tla.Float32, tla.AddressSpace.ub, 256), 1, VL)
    ub_zer = _t2d(tla.allocate(VL, tla.Float32, tla.AddressSpace.ub, 256), 1, VL)

    m_in = tla.mutex("k1_in", id=0)
    m_in1 = tla.mutex("k1_in1", id=1)
    m_out = tla.mutex("k1_out", id=2)
    m_out1 = tla.mutex("k1_out1", id=5)
    m_sc = tla.mutex("k1_sc", id=3)
    m_z = tla.mutex("k1_z", id=4)
    m_w = tla.mutex("k1_w", id=6)

    with tla.vector():
        # w is loaded once per block for the hot-token y = u * gv * w.
        m_w.lock(pipe=tla.arch.MTE2)
        tla.copy(
            ub_w,
            tla.tile_view(gm_w, tla.make_shape(1, H), tla.make_coord(0, 0)),
        )
        m_w.unlock(pipe=tla.arch.MTE2)

        # ---- clear part (the split-K atomic accumulation target of K2) ----
        m_z.lock(pipe=tla.arch.VECTOR)
        with tla.vec.func(mode="simd"):
            tla.tile_view(
                ub_zer, tla.make_shape(1, VL), tla.make_coord(0, 0)
            ).store(tla.full(0.0, dtype=tla.Float32))
        m_z.unlock(pipe=tla.arch.VECTOR)
        m_z.lock(pipe=tla.arch.MTE3)
        for gz in tla.range(bid, PN // VL, nblk):
            tla.copy(
                tla.tile_view(
                    gm_partf, tla.make_shape(1, VL), tla.make_coord(0, gz)
                ),
                tla.tile_view(ub_zer, tla.make_shape(1, VL), tla.make_coord(0, 0)),
            )
        m_z.unlock(pipe=tla.arch.MTE3)

        # Tasks are the flattened T*M chunks spread evenly over the cores: each block
        # takes UT/NB of them.  A token can therefore straddle two blocks, so this core
        # only mixes its own chunk range and writes the RMS sums as slot partials that
        # K3 reduces; ``u`` is written per chunk into ``gm_y`` and normalized by K3,
        # which avoids token-level atomics.
        NB = NBLK_VEC
        UT = T * M
        # Task count: when T*M < NB only the first T*M cores are active so that the
        # busy cores are contiguous in bid.  A sparse floor(bid*UT/NB) mapping would
        # place adjacent chunks on cores far apart and slot = bid % NSLOT could then
        # collide (e.g. T=1 sends chunk0 to core 31 and chunk1 to core 63, which share
        # a slot and overwrite each other's partials).
        NT = UT if UT < NB else NB
        slot = bid % NSLOT
        # Default: spread chunks evenly (better balance for small T, critical load
        # ceil(T*M/NT) chunks).
        lo = (bid * UT) // NT
        hi = ((bid + 1) * UT) // NT
        if UT < NB:
            # Static: only the first NT cores are active; the rest clamp to empty.
            if lo > UT:
                lo = UT
            if hi > UT:
                hi = UT
        if SPLIT == 0:
            # Large T (bandwidth bound): assign whole tokens, ceil(T/NB) per core.
            # Every token is hot (y finished in core, u never reaches GM) and K3 only
            # runs the coefficient chain.
            lo = ((bid * T) // NB) * M
            hi = (((bid + 1) * T) // NB) * M
        if hi > lo:
            t0 = lo // M
            t1 = (hi - 1) // M
            for t in tla.range(t0, t1 + 1):
                m_sc.lock(pipe=tla.arch.MTE2)
                tla.copy(
                    ub_comb,
                    tla.tile_view(gm_comb, tla.make_shape(1, 16), tla.make_coord(t, 0)),
                )
                tla.copy(
                    ub_post,
                    tla.tile_view(gm_post, tla.make_shape(1, 4), tla.make_coord(t, 0)),
                )
                tla.copy(
                    ub_pm,
                    tla.tile_view(gm_pm, tla.make_shape(1, 4), tla.make_coord(t, 0)),
                )
                m_sc.unlock(pipe=tla.arch.MTE2)

                m_sc.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                    z0 = tla.full(0.0, dtype=tla.Float32)
                    tla.tile_view(
                        ub_ssz, tla.make_shape(1, VL), tla.make_coord(0, 0)
                    ).store(z0)
                    tla.tile_view(
                        ub_ssu, tla.make_shape(1, VL), tla.make_coord(0, 0)
                    ).store(z0)
                m_sc.unlock(pipe=tla.arch.VECTOR)

                # chunk range [c0, c1) this core covers for the token
                c0 = lo - t * M
                if c0 < 0:
                    c0 = 0
                c1 = hi - t * M
                if c1 > M:
                    c1 = M
                # hot = this core owns every chunk of the token, so gv can be computed
                # here and y finished in place; otherwise this core only contributes u
                # and a slot partial and K3 finishes y.
                # Only M==2 has the hot path: the two chunk buffers then hold the whole
                # token and all addresses are static (M>2 falls back to all-cold).
                hot = 0
                if M == 2:
                    if c0 == 0:
                        if c1 == M:
                            hot = 1

                # First chunk DMA; later chunks are prefetched into the other buffer
                # while this one is being computed.
                m_in.lock(pipe=tla.arch.MTE2)
                tla.copy(
                    ub_res,
                    tla.tile_view(
                        gm_res, tla.make_shape(4, CH), tla.make_coord(t, c0)
                    ),
                )
                tla.copy(
                    ub_x,
                    tla.tile_view(
                        gm_x, tla.make_shape(1, CH), tla.make_coord(t, c0)
                    ),
                )
                m_in.unlock(pipe=tla.arch.MTE2)

                ph = tla.as_numeric(0)
                for hc in tla.range(c0, c1):
                    hci = hc
                    if ph == 0:
                        m_in.lock(pipe=tla.arch.VECTOR)
                    else:
                        m_in1.lock(pipe=tla.arch.VECTOR)
                    rsrc = ub_res if ph == 0 else ub_res2
                    xsrc = ub_x if ph == 0 else ub_x2
                    if hc + 1 < c1:
                        if ph == 0:
                            m_in1.lock(pipe=tla.arch.MTE2)
                            tla.copy(
                                ub_res2,
                                tla.tile_view(
                                    gm_res, tla.make_shape(4, CH),
                                    tla.make_coord(t, hci + 1),
                                ),
                            )
                            tla.copy(
                                ub_x2,
                                tla.tile_view(
                                    gm_x, tla.make_shape(1, CH),
                                    tla.make_coord(t, hci + 1),
                                ),
                            )
                            m_in1.unlock(pipe=tla.arch.MTE2)
                        else:
                            m_in.lock(pipe=tla.arch.MTE2)
                            tla.copy(
                                ub_res,
                                tla.tile_view(
                                    gm_res, tla.make_shape(4, CH),
                                    tla.make_coord(t, hci + 1),
                                ),
                            )
                            tla.copy(
                                ub_x,
                                tla.tile_view(
                                    gm_x, tla.make_shape(1, CH),
                                    tla.make_coord(t, hci + 1),
                                ),
                            )
                            m_in.unlock(pipe=tla.arch.MTE2)

                    # Output double buffer: writing this chunk back does not stall the
                    # next chunk's compute.
                    if ph == 0:
                        m_out.lock(pipe=tla.arch.VECTOR)
                    else:
                        m_out1.lock(pipe=tla.arch.VECTOR)
                    wr = ub_r if ph == 0 else ub_r2
                    wu = ub_uv if ph == 0 else ub_uv2
                    with tla.vec.func(mode="simd"):
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE,
                            tla.params.MemType.VEC_LOAD,
                        )
                        zz = tla.tile_view(
                            ub_ssz, tla.make_shape(1, VL), tla.make_coord(0, 0)
                        ).load()
                        uu = tla.tile_view(
                            ub_ssu, tla.make_shape(1, VL), tla.make_coord(0, 0)
                        ).load()
                        # Group size: four groups at once spill the vector registers,
                        # while the single-token path is latency bound, so only the
                        # two-group body is used there and single groups elsewhere.
                        if T > 4:
                            for gpa in tla.range(0, CH // VL // 2):
                                zz, uu = _mix_n(
                                    xsrc, rsrc, wr, wu, ub_comb, ub_post, ub_pm,
                                    gpa, 0, zz, uu, 2,
                                )
                            if (CH // VL) % 2:
                                zz, uu = _mix_group(
                                    xsrc, rsrc, wr, wu, ub_comb, ub_post, ub_pm,
                                    CH // VL - 1, 0, zz, uu,
                                )
                        else:
                            for gpa in tla.range(0, CH // VL):
                                zz, uu = _mix_group(
                                    xsrc, rsrc, wr, wu, ub_comb, ub_post, ub_pm,
                                    gpa, 0, zz, uu,
                                )
                        tla.tile_view(
                            ub_ssz, tla.make_shape(1, VL), tla.make_coord(0, 0)
                        ).store(zz)
                        tla.tile_view(
                            ub_ssu, tla.make_shape(1, VL), tla.make_coord(0, 0)
                        ).store(uu)
                    if ph == 0:
                        m_out.unlock(pipe=tla.arch.VECTOR)
                    else:
                        m_out1.unlock(pipe=tla.arch.VECTOR)
                    if ph == 0:
                        m_in.unlock(pipe=tla.arch.VECTOR)
                    else:
                        m_in1.unlock(pipe=tla.arch.VECTOR)

                    if ph == 0:
                        m_out.lock(pipe=tla.arch.MTE3)
                    else:
                        m_out1.lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        tla.tile_view(
                            gm_new_res,
                            tla.make_shape(4, CH),
                            tla.make_coord(t, hci),
                        ),
                        wr,
                    )
                    # Only cold tokens write this chunk's u back (K3 turns it into y via
                    # gv * w); a hot token finishes y once from both chunk buffers.
                    if hot == 0:
                        tla.copy(
                            tla.tile_view(
                                gm_y, tla.make_shape(1, CH), tla.make_coord(t, hci)
                            ),
                            wu,
                        )
                    if ph == 0:
                        m_out.unlock(pipe=tla.arch.MTE3)
                    else:
                        m_out1.unlock(pipe=tla.arch.MTE3)
                    ph = 1 - ph

                # Reduce zz/su and write this core's slot partial for K3.
                # A hot token's y = u * gv * w joins the same vector region: gv is
                # broadcast from su in place, which saves the ub_gv staging tile and a
                # second vector region/barrier.  Both chunks live in ub_uv/ub_uv2 and
                # hot implies c0 == 0 and c1 == M, so the final ph is 0 and every
                # address is static.
                if M == 2:
                    if hot == 1:
                        ngc = CH // VL
                        m_w.lock(pipe=tla.arch.VECTOR)
                        with tla.vec.func(mode="simd"):
                            su = _red_ss(ub_ssz, ub_ssu, ub_s)
                            gvv = tla.div(
                                tla.full(1.0, dtype=tla.Float32),
                                tla.sqrt(
                                    tla.add(
                                        tla.mul(tla.full(su, dtype=tla.Float32), 1.0 / H),
                                        7e-6,
                                    )
                                ),
                            )
                            for gp in tla.range(0, ngc):
                                _st_bf(
                                    ub_yall, 0, gp,
                                    tla.mul(tla.mul(_bf(ub_uv, 0, gp), gvv),
                                            _bf(ub_w, 0, gp)),
                                )
                            for gq in tla.range(0, ngc):
                                _st_bf(
                                    ub_yall, 0, ngc + gq,
                                    tla.mul(tla.mul(_bf(ub_uv2, 0, gq), gvv),
                                            _bf(ub_w, 0, ngc + gq)),
                                )
                        m_w.unlock(pipe=tla.arch.VECTOR)
                    else:
                        with tla.vec.func(mode="simd"):
                            _red_ss(ub_ssz, ub_ssu, ub_s)
                else:
                    with tla.vec.func(mode="simd"):
                        _red_ss(ub_ssz, ub_ssu, ub_s)
                # the full barrier also orders the cross-task pipeline
                tla.pipe_barrier(tla.pipes.ALL)
                with tla.vec.func(mode="simt", thread_block_dim=1):
                    gm_pzz[slot, t] = ub_s[0, 0]
                    gm_puu[t, slot] = ub_s[0, 1]
                if hot == 1:
                    m_out.lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        tla.tile_view(
                            gm_y, tla.make_shape(1, H), tla.make_coord(t, 0)
                        ),
                        ub_yall,
                    )
                    m_out.unlock(pipe=tla.arch.MTE3)


# ---------------------------------------------------------------------------
# K2: split-K projection  l[MF, T] = fn[MF, K] @ z[K, T]
# ---------------------------------------------------------------------------
@tla.kernel
def proj_gemm_kernel(
    a: tla.Tensor,      # [MF, K] bf16
    b: tla.Tensor,      # [K, T] (col) bf16
    c: tla.Tensor,      # [SPLIT*32, TPAD] f32
    SPLIT: tla.Constexpr[int],
    NT: tla.Constexpr[int],
    KU: tla.Constexpr[int],
    UPC: tla.Constexpr[int],
    BK: tla.Constexpr[int],
    BN: tla.Constexpr[int],
    SK: tla.Constexpr[int],
) -> None:
    zero = 0
    one = 1
    bm = BM_G
    dtype = a.ptr.dtype
    nsk = BK // SK

    m_l1a0 = tla.mutex("g_l1a0", id=0)
    m_l1b0 = tla.mutex("g_l1b0", id=1)
    m_l1a1 = tla.mutex("g_l1a1", id=2)
    m_l1b1 = tla.mutex("g_l1b1", id=3)
    m_l0a0 = tla.mutex("g_l0a0", id=4)
    m_l0b0 = tla.mutex("g_l0b0", id=5)
    m_l0a1 = tla.mutex("g_l0a1", id=6)
    m_l0b1 = tla.mutex("g_l0b1", id=7)
    m_l0c = tla.mutex("g_l0c", id=8)

    l1a0_ptr = tla.allocate(bm * BK, dtype, tla.AddressSpace.l1, 512)
    l1a1_ptr = tla.allocate(bm * BK, dtype, tla.AddressSpace.l1, 512)
    l1b0_ptr = tla.allocate(BK * BN, dtype, tla.AddressSpace.l1, 512)
    l1b1_ptr = tla.allocate(BK * BN, dtype, tla.AddressSpace.l1, 512)
    l0a0_ptr = tla.allocate(bm * SK, dtype, tla.AddressSpace.l0a, 512)
    l0a1_ptr = tla.allocate(bm * SK, dtype, tla.AddressSpace.l0a, 512)
    l0b0_ptr = tla.allocate(SK * BN, dtype, tla.AddressSpace.l0b, 512)
    l0b1_ptr = tla.allocate(SK * BN, dtype, tla.AddressSpace.l0b, 512)
    l0c = tla.allocate(bm * BN, tla.Float32, tla.AddressSpace.l0c, 1024)

    with tla.cube():
        for task in tla.range(tla.arch.block_idx(), SPLIT * NT, tla.arch.block_num()):
            sb = task // NT
            nt = task % NT
            u_lo = sb * UPC
            u_stop = (sb + 1) * UPC
            if u_stop > KU:
                u_stop = KU
            if u_lo < u_stop:
                og = tla.tile_view(
                    c, tla.make_shape(bm, BN), tla.make_coord(zero, nt)
                )
                l0_c = tla.make_tensor_like(l0c, og)

                ag_head = tla.tile_view(
                    a, tla.make_shape(bm, BK), tla.make_coord(zero, u_lo)
                )
                bg_head = tla.tile_view(
                    b, tla.make_shape(BK, BN), tla.make_coord(u_lo, nt)
                )
                al = tla.make_tensor_like(l1a0_ptr, ag_head, tla.arch.zN)
                bl = tla.make_tensor_like(l1b0_ptr, bg_head, tla.arch.nZ)
                m_l1a0.lock(pipe=tla.arch.MTE2)
                tla.copy(al, ag_head)
                m_l1a0.unlock(pipe=tla.arch.MTE2)
                m_l1b0.lock(pipe=tla.arch.MTE2)
                tla.copy(bl, bg_head)
                m_l1b0.unlock(pipe=tla.arch.MTE2)
                m_l0c.lock(pipe=tla.arch.CUBE)

                l1_phase = tla.as_numeric(zero)
                for ku in tla.range(u_lo, u_stop):
                    if l1_phase == zero:
                        m_l1a0.lock(pipe=tla.arch.MTE1)
                        m_l1b0.lock(pipe=tla.arch.MTE1)
                    else:
                        m_l1a1.lock(pipe=tla.arch.MTE1)
                        m_l1b1.lock(pipe=tla.arch.MTE1)

                    if ku + one < u_stop:
                        ag_n = tla.tile_view(
                            a, tla.make_shape(bm, BK), tla.make_coord(zero, ku + one)
                        )
                        bg_n = tla.tile_view(
                            b,
                            tla.make_shape(BK, BN),
                            tla.make_coord(ku + one, nt),
                        )
                        na_ptr = l1a1_ptr if l1_phase == zero else l1a0_ptr
                        nb_ptr = l1b1_ptr if l1_phase == zero else l1b0_ptr
                        al_n = tla.make_tensor_like(na_ptr, ag_n, tla.arch.zN)
                        bl_n = tla.make_tensor_like(nb_ptr, bg_n, tla.arch.nZ)
                        if l1_phase == zero:
                            m_l1a1.lock(pipe=tla.arch.MTE2)
                            m_l1b1.lock(pipe=tla.arch.MTE2)
                        else:
                            m_l1a0.lock(pipe=tla.arch.MTE2)
                            m_l1b0.lock(pipe=tla.arch.MTE2)
                        tla.copy(al_n, ag_n)
                        tla.copy(bl_n, bg_n)
                        if l1_phase == zero:
                            m_l1a1.unlock(pipe=tla.arch.MTE2)
                            m_l1b1.unlock(pipe=tla.arch.MTE2)
                        else:
                            m_l1a0.unlock(pipe=tla.arch.MTE2)
                            m_l1b0.unlock(pipe=tla.arch.MTE2)

                    ca_ptr = l1a0_ptr if l1_phase == zero else l1a1_ptr
                    cb_ptr = l1b0_ptr if l1_phase == zero else l1b1_ptr
                    ag_c = tla.tile_view(
                        a, tla.make_shape(bm, BK), tla.make_coord(zero, ku)
                    )
                    bg_c = tla.tile_view(
                        b, tla.make_shape(BK, BN), tla.make_coord(ku, nt)
                    )
                    al_c = tla.make_tensor_like(ca_ptr, ag_c, tla.arch.zN)
                    bl_c = tla.make_tensor_like(cb_ptr, bg_c, tla.arch.nZ)

                    aa = tla.tile_view(
                        al_c, tla.make_shape(bm, SK), tla.make_coord(zero, zero)
                    )
                    bb = tla.tile_view(
                        bl_c, tla.make_shape(SK, BN), tla.make_coord(zero, zero)
                    )
                    a0 = tla.make_tensor_like(l0a0_ptr, aa, tla.arch.zN)
                    b0 = tla.make_tensor_like(l0b0_ptr, bb, tla.arch.nZ)
                    m_l0a0.lock(pipe=tla.arch.MTE1)
                    m_l0b0.lock(pipe=tla.arch.MTE1)
                    tla.copy(a0, aa)
                    tla.copy(b0, bb)
                    m_l0a0.unlock(pipe=tla.arch.MTE1)
                    m_l0b0.unlock(pipe=tla.arch.MTE1)

                    l0_phase = tla.as_numeric(zero)
                    for kk in tla.range(zero, nsk, one):
                        if l0_phase == zero:
                            m_l0a0.lock(pipe=tla.arch.CUBE)
                            m_l0b0.lock(pipe=tla.arch.CUBE)
                        else:
                            m_l0a1.lock(pipe=tla.arch.CUBE)
                            m_l0b1.lock(pipe=tla.arch.CUBE)

                        if kk + one < nsk:
                            aa_n = tla.tile_view(
                                al_c,
                                tla.make_shape(bm, SK),
                                tla.make_coord(zero, kk + one),
                            )
                            bb_n = tla.tile_view(
                                bl_c,
                                tla.make_shape(SK, BN),
                                tla.make_coord(kk + one, zero),
                            )
                            na0_ptr = l0a1_ptr if l0_phase == zero else l0a0_ptr
                            nb0_ptr = l0b1_ptr if l0_phase == zero else l0b0_ptr
                            a0_n = tla.make_tensor_like(na0_ptr, aa_n, tla.arch.zN)
                            b0_n = tla.make_tensor_like(nb0_ptr, bb_n, tla.arch.nZ)
                            if l0_phase == zero:
                                m_l0a1.lock(pipe=tla.arch.MTE1)
                                m_l0b1.lock(pipe=tla.arch.MTE1)
                            else:
                                m_l0a0.lock(pipe=tla.arch.MTE1)
                                m_l0b0.lock(pipe=tla.arch.MTE1)
                            tla.copy(a0_n, aa_n)
                            tla.copy(b0_n, bb_n)
                            if l0_phase == zero:
                                m_l0a1.unlock(pipe=tla.arch.MTE1)
                                m_l0b1.unlock(pipe=tla.arch.MTE1)
                            else:
                                m_l0a0.unlock(pipe=tla.arch.MTE1)
                                m_l0b0.unlock(pipe=tla.arch.MTE1)

                        ca0_ptr = l0a0_ptr if l0_phase == zero else l0a1_ptr
                        cb0_ptr = l0b0_ptr if l0_phase == zero else l0b1_ptr
                        aa_c = tla.tile_view(
                            al_c, tla.make_shape(bm, SK), tla.make_coord(zero, kk)
                        )
                        bb_c = tla.tile_view(
                            bl_c, tla.make_shape(SK, BN), tla.make_coord(kk, zero)
                        )
                        a0_c = tla.make_tensor_like(ca0_ptr, aa_c, tla.arch.zN)
                        b0_c = tla.make_tensor_like(cb0_ptr, bb_c, tla.arch.nZ)
                        is_last = (ku == u_stop - 1) and (kk == nsk - 1)
                        tla.mmad(
                            l0_c,
                            a0_c,
                            b0_c,
                            init_c=(ku == u_lo and kk == zero),
                            unit_flag=0b11 if is_last else 0b10,
                        )
                        if l0_phase == zero:
                            m_l0a0.unlock(pipe=tla.arch.CUBE)
                            m_l0b0.unlock(pipe=tla.arch.CUBE)
                        else:
                            m_l0a1.unlock(pipe=tla.arch.CUBE)
                            m_l0b1.unlock(pipe=tla.arch.CUBE)
                        l0_phase = one - l0_phase

                    if l1_phase == zero:
                        m_l1a0.unlock(pipe=tla.arch.MTE1)
                        m_l1b0.unlock(pipe=tla.arch.MTE1)
                    else:
                        m_l1a1.unlock(pipe=tla.arch.MTE1)
                        m_l1b1.unlock(pipe=tla.arch.MTE1)
                    l1_phase = one - l1_phase

                m_l0c.unlock(pipe=tla.arch.CUBE)
                m_l0c.lock(pipe=tla.arch.FIX)
                # split-K partial sums accumulate atomically in GM; K3 reads them by row.
                tla.copy(
                    og,
                    l0_c,
                    tla.params.CopyL0C2DstParams(
                        unit_flag=0b11, atomic_mode=tla.params.AtomicMode.ADD
                    ),
                )
                m_l0c.unlock(pipe=tla.arch.FIX)


# ---------------------------------------------------------------------------
# K3: y (scaled after RMSNorm) + coefficient finishing (sigmoid / softmax / Sinkhorn)
# ---------------------------------------------------------------------------
@tla.kernel
def post_kernel(
    gm_pzz: tla.Tensor,     # [P, TPAD] f32 (P = NSLOT core slots)
    gm_part: tla.Tensor,    # [32, TPAD] f32
    gm_scales: tla.Tensor,  # [3] f32
    gm_bases: tla.Tensor,   # [24] f32
    gm_prev: tla.Tensor,    # [TPAD, 4] f32 (coefficients 0:4)
    gm_post: tla.Tensor,    # [TPAD, 4] f32 (coefficients 4:8)
    gm_comb: tla.Tensor,    # [TPAD, 16] f32 (coefficients 8:24)
    gm_yu: tla.Tensor,      # [T, H] bf16 (u written by K1, overwritten in place by y)
    gm_w: tla.Tensor,       # [1, H] bf16
    gm_puu: tla.Tensor,     # [TPAD, NSLOT] f32 (u squared partial sums per core slot)
    T: tla.Constexpr[int],
    H: tla.Constexpr[int],
    P: tla.Constexpr[int],
    NCOEF: tla.Constexpr[int],
    M: tla.Constexpr[int],
    NYB: tla.Constexpr[int],   # blocks that finish cold tokens (0 = none for this shape)
    NTSK: tla.Constexpr[int],  # K1 chunk tasks = min(T*M, NBLK_VEC)
) -> None:
    K4 = 4 * H
    bid = tla.arch.block_idx()

    ub_pz = _t2d(tla.allocate(P * VL, tla.Float32, tla.AddressSpace.ub, 256), P, VL)
    ub_pp = _t2d(tla.allocate(MF * VL, tla.Float32, tla.AddressSpace.ub, 256), MF, VL)
    ub_out = _t2d(tla.allocate(MF * VL, tla.Float32, tla.AddressSpace.ub, 256), MF, VL)
    # Token-major staging for the coefficient write-back ([token, 24 coefficients]):
    # a 96B row satisfies the 32B MTE3 row granularity, and ub_flat is the (1, VL*MF)
    # view of the same UB block used for one whole-group DMA.
    tk_p = tla.allocate(VL * 32, tla.Float32, tla.AddressSpace.ub, 256)
    ub_tk = _t2d(tk_p, VL, 32)
    ub_flat = _t2d(tk_p, 1, VL * MF)
    ub_sc = _t2d(tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256), 1, 64)
    ub_ba = _t2d(tla.allocate(64, tla.Float32, tla.AddressSpace.ub, 256), 1, 64)
    ub_pu = _t2d(tla.allocate(VL, tla.Float32, tla.AddressSpace.ub, 256), 1, VL)
    ub_u = _t2d(tla.allocate(H, tla.BFloat16, tla.AddressSpace.ub, 256), 1, H)
    ub_w = _t2d(tla.allocate(H, tla.BFloat16, tla.AddressSpace.ub, 256), 1, H)
    ub_y = _t2d(tla.allocate(H, tla.BFloat16, tla.AddressSpace.ub, 256), 1, H)

    m_c = tla.mutex("k3_c", id=0)
    m_in = tla.mutex("k3_in", id=1)
    m_out = tla.mutex("k3_out", id=2)

    with tla.vector():
        # ================= coefficient finishing (first NCOEF blocks) =================
        if bid < NCOEF:
            ngroups = (T + VL - 1) // VL
            for g in tla.range(bid, ngroups, NCOEF):
                m_c.lock(pipe=tla.arch.MTE2)
                tla.copy(
                    ub_pz,
                    tla.tile_view(
                        gm_pzz, tla.make_shape(P, VL), tla.make_coord(0, g)
                    ),
                )
                tla.copy(
                    ub_pp,
                    tla.tile_view(
                        gm_part, tla.make_shape(MF, VL), tla.make_coord(0, g)
                    ),
                )
                tla.copy(
                    ub_sc,
                    tla.tile_view(gm_scales, tla.make_shape(1, 3), tla.make_coord(0, 0)),
                )
                tla.copy(
                    ub_ba,
                    tla.tile_view(gm_bases, tla.make_shape(1, 24), tla.make_coord(0, 0)),
                )
                m_c.unlock(pipe=tla.arch.MTE2)
                m_c.lock(pipe=tla.arch.VECTOR)
                with tla.vec.func(mode="simd"):
                    tla.local_mem_bar(
                        tla.params.MemType.VEC_STORE, tla.params.MemType.VEC_LOAD
                    )
                    one_f = tla.full(1.0, dtype=tla.Float32)
                    zz = _ld1(ub_pz, 0)
                    for p in tla.range_constexpr(1, P):
                        zz = tla.add(zz, _ld1(ub_pz, p))
                    invz = tla.div(
                        one_f, tla.sqrt(tla.add(tla.mul(zz, 1.0 / K4), 2e-5))
                    )
                    s0 = _brc(ub_sc, 0)
                    s1 = _brc(ub_sc, 1)
                    s2 = _brc(ub_sc, 2)
                    l0 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 0), invz), s0), _brc(ub_ba, 0))
                    l1 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 1), invz), s0), _brc(ub_ba, 1))
                    l2 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 2), invz), s0), _brc(ub_ba, 2))
                    l3 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 3), invz), s0), _brc(ub_ba, 3))
                    l4 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 4), invz), s1), _brc(ub_ba, 4))
                    l5 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 5), invz), s1), _brc(ub_ba, 5))
                    l6 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 6), invz), s1), _brc(ub_ba, 6))
                    l7 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 7), invz), s1), _brc(ub_ba, 7))
                    l8 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 8), invz), s2), _brc(ub_ba, 8))
                    l9 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 9), invz), s2), _brc(ub_ba, 9))
                    l10 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 10), invz), s2), _brc(ub_ba, 10))
                    l11 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 11), invz), s2), _brc(ub_ba, 11))
                    l12 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 12), invz), s2), _brc(ub_ba, 12))
                    l13 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 13), invz), s2), _brc(ub_ba, 13))
                    l14 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 14), invz), s2), _brc(ub_ba, 14))
                    l15 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 15), invz), s2), _brc(ub_ba, 15))
                    l16 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 16), invz), s2), _brc(ub_ba, 16))
                    l17 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 17), invz), s2), _brc(ub_ba, 17))
                    l18 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 18), invz), s2), _brc(ub_ba, 18))
                    l19 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 19), invz), s2), _brc(ub_ba, 19))
                    l20 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 20), invz), s2), _brc(ub_ba, 20))
                    l21 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 21), invz), s2), _brc(ub_ba, 21))
                    l22 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 22), invz), s2), _brc(ub_ba, 22))
                    l23 = tla.add(tla.mul(tla.mul(_ld1(ub_pp, 23), invz), s2), _brc(ub_ba, 23))
                    e0 = tla.exp(tla.neg(l0))
                    e1 = tla.exp(tla.neg(l1))
                    e2 = tla.exp(tla.neg(l2))
                    e3 = tla.exp(tla.neg(l3))
                    e4 = tla.exp(tla.neg(l4))
                    e5 = tla.exp(tla.neg(l5))
                    e6 = tla.exp(tla.neg(l6))
                    e7 = tla.exp(tla.neg(l7))
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(0, 0)
                    ).store(tla.add(tla.div(one_f, tla.add(one_f, e0)), 3e-4))
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(1, 0)
                    ).store(tla.add(tla.div(one_f, tla.add(one_f, e1)), 3e-4))
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(2, 0)
                    ).store(tla.add(tla.div(one_f, tla.add(one_f, e2)), 3e-4))
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(3, 0)
                    ).store(tla.add(tla.div(one_f, tla.add(one_f, e3)), 3e-4))
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(4, 0)
                    ).store(tla.mul(tla.div(one_f, tla.add(one_f, e4)), 1.25))
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(5, 0)
                    ).store(tla.mul(tla.div(one_f, tla.add(one_f, e5)), 1.25))
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(6, 0)
                    ).store(tla.mul(tla.div(one_f, tla.add(one_f, e6)), 1.25))
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(7, 0)
                    ).store(tla.mul(tla.div(one_f, tla.add(one_f, e7)), 1.25))
                    # comb: softmax over j
                    mx0 = tla.max(tla.max(l8, l9), tla.max(l10, l11))
                    f0 = tla.exp(tla.sub(l8, mx0))
                    f1 = tla.exp(tla.sub(l9, mx0))
                    f2 = tla.exp(tla.sub(l10, mx0))
                    f3 = tla.exp(tla.sub(l11, mx0))
                    r0 = tla.add(tla.add(f0, f1), tla.add(f2, f3))
                    c00 = tla.add(tla.div(f0, r0), 2e-6)
                    c01 = tla.add(tla.div(f1, r0), 2e-6)
                    c02 = tla.add(tla.div(f2, r0), 2e-6)
                    c03 = tla.add(tla.div(f3, r0), 2e-6)
                    mx1 = tla.max(tla.max(l12, l13), tla.max(l14, l15))
                    f4 = tla.exp(tla.sub(l12, mx1))
                    f5 = tla.exp(tla.sub(l13, mx1))
                    f6 = tla.exp(tla.sub(l14, mx1))
                    f7 = tla.exp(tla.sub(l15, mx1))
                    r1 = tla.add(tla.add(f4, f5), tla.add(f6, f7))
                    c10 = tla.add(tla.div(f4, r1), 2e-6)
                    c11 = tla.add(tla.div(f5, r1), 2e-6)
                    c12 = tla.add(tla.div(f6, r1), 2e-6)
                    c13 = tla.add(tla.div(f7, r1), 2e-6)
                    mx2 = tla.max(tla.max(l16, l17), tla.max(l18, l19))
                    f8 = tla.exp(tla.sub(l16, mx2))
                    f9 = tla.exp(tla.sub(l17, mx2))
                    f10 = tla.exp(tla.sub(l18, mx2))
                    f11 = tla.exp(tla.sub(l19, mx2))
                    r2 = tla.add(tla.add(f8, f9), tla.add(f10, f11))
                    c20 = tla.add(tla.div(f8, r2), 2e-6)
                    c21 = tla.add(tla.div(f9, r2), 2e-6)
                    c22 = tla.add(tla.div(f10, r2), 2e-6)
                    c23 = tla.add(tla.div(f11, r2), 2e-6)
                    mx3 = tla.max(tla.max(l20, l21), tla.max(l22, l23))
                    f12 = tla.exp(tla.sub(l20, mx3))
                    f13 = tla.exp(tla.sub(l21, mx3))
                    f14 = tla.exp(tla.sub(l22, mx3))
                    f15 = tla.exp(tla.sub(l23, mx3))
                    r3 = tla.add(tla.add(f12, f13), tla.add(f14, f15))
                    c30 = tla.add(tla.div(f12, r3), 2e-6)
                    c31 = tla.add(tla.div(f13, r3), 2e-6)
                    c32 = tla.add(tla.div(f14, r3), 2e-6)
                    c33 = tla.add(tla.div(f15, r3), 2e-6)
                    # Column normalization (one column pass right after softmax).
                    k0 = tla.add(tla.add(c00, c10), tla.add(c20, c30))
                    k1 = tla.add(tla.add(c01, c11), tla.add(c21, c31))
                    k2 = tla.add(tla.add(c02, c12), tla.add(c22, c32))
                    k3 = tla.add(tla.add(c03, c13), tla.add(c23, c33))
                    k0 = tla.div(one_f, tla.add(k0, 2e-6))
                    k1 = tla.div(one_f, tla.add(k1, 2e-6))
                    k2 = tla.div(one_f, tla.add(k2, 2e-6))
                    k3 = tla.div(one_f, tla.add(k3, 2e-6))
                    c00 = tla.mul(c00, k0)
                    c01 = tla.mul(c01, k1)
                    c02 = tla.mul(c02, k2)
                    c03 = tla.mul(c03, k3)
                    c10 = tla.mul(c10, k0)
                    c11 = tla.mul(c11, k1)
                    c12 = tla.mul(c12, k2)
                    c13 = tla.mul(c13, k3)
                    c20 = tla.mul(c20, k0)
                    c21 = tla.mul(c21, k1)
                    c22 = tla.mul(c22, k2)
                    c23 = tla.mul(c23, k3)
                    c30 = tla.mul(c30, k0)
                    c31 = tla.mul(c31, k1)
                    c32 = tla.mul(c32, k2)
                    c33 = tla.mul(c33, k3)
                    # Sinkhorn alternates the normalizations: the reference does one
                    # column pass followed by N-1 row/column rounds.  Fewer rounds do
                    # not reach the required accuracy, so all nine are kept.
                    for it in tla.range_constexpr(9):
                        d0 = tla.add(tla.add(c00, c01), tla.add(c02, c03))
                        d1 = tla.add(tla.add(c10, c11), tla.add(c12, c13))
                        d2 = tla.add(tla.add(c20, c21), tla.add(c22, c23))
                        d3 = tla.add(tla.add(c30, c31), tla.add(c32, c33))
                        d0 = tla.div(one_f, tla.add(d0, 2e-6))
                        d1 = tla.div(one_f, tla.add(d1, 2e-6))
                        d2 = tla.div(one_f, tla.add(d2, 2e-6))
                        d3 = tla.div(one_f, tla.add(d3, 2e-6))
                        c00 = tla.mul(c00, d0)
                        c01 = tla.mul(c01, d0)
                        c02 = tla.mul(c02, d0)
                        c03 = tla.mul(c03, d0)
                        c10 = tla.mul(c10, d1)
                        c11 = tla.mul(c11, d1)
                        c12 = tla.mul(c12, d1)
                        c13 = tla.mul(c13, d1)
                        c20 = tla.mul(c20, d2)
                        c21 = tla.mul(c21, d2)
                        c22 = tla.mul(c22, d2)
                        c23 = tla.mul(c23, d2)
                        c30 = tla.mul(c30, d3)
                        c31 = tla.mul(c31, d3)
                        c32 = tla.mul(c32, d3)
                        c33 = tla.mul(c33, d3)
                        e0 = tla.add(tla.add(c00, c10), tla.add(c20, c30))
                        e1 = tla.add(tla.add(c01, c11), tla.add(c21, c31))
                        e2 = tla.add(tla.add(c02, c12), tla.add(c22, c32))
                        e3 = tla.add(tla.add(c03, c13), tla.add(c23, c33))
                        e0 = tla.div(one_f, tla.add(e0, 2e-6))
                        e1 = tla.div(one_f, tla.add(e1, 2e-6))
                        e2 = tla.div(one_f, tla.add(e2, 2e-6))
                        e3 = tla.div(one_f, tla.add(e3, 2e-6))
                        c00 = tla.mul(c00, e0)
                        c01 = tla.mul(c01, e1)
                        c02 = tla.mul(c02, e2)
                        c03 = tla.mul(c03, e3)
                        c10 = tla.mul(c10, e0)
                        c11 = tla.mul(c11, e1)
                        c12 = tla.mul(c12, e2)
                        c13 = tla.mul(c13, e3)
                        c20 = tla.mul(c20, e0)
                        c21 = tla.mul(c21, e1)
                        c22 = tla.mul(c22, e2)
                        c23 = tla.mul(c23, e3)
                        c30 = tla.mul(c30, e0)
                        c31 = tla.mul(c31, e1)
                        c32 = tla.mul(c32, e2)
                        c33 = tla.mul(c33, e3)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(8, 0)
                    ).store(c00)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(9, 0)
                    ).store(c01)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(10, 0)
                    ).store(c02)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(11, 0)
                    ).store(c03)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(12, 0)
                    ).store(c10)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(13, 0)
                    ).store(c11)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(14, 0)
                    ).store(c12)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(15, 0)
                    ).store(c13)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(16, 0)
                    ).store(c20)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(17, 0)
                    ).store(c21)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(18, 0)
                    ).store(c22)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(19, 0)
                    ).store(c23)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(20, 0)
                    ).store(c30)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(21, 0)
                    ).store(c31)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(22, 0)
                    ).store(c32)
                    tla.tile_view(
                        ub_out, tla.make_shape(1, VL), tla.make_coord(23, 0)
                    ).store(c33)
                m_c.unlock(pipe=tla.arch.VECTOR)
                tla.pipe_barrier(tla.pipes.ALL)
                # Coefficient write-back: transpose [coeff, token] to [token, coeff] in
                # UB with SIMT, then issue one flat MTE3 DMA.  The transpose is a
                # per-thread scalar latency, so each token is split over TSPLIT threads
                # (thread_block_dim = VL*TSPLIT) and each thread keeps MF/TSPLIT values.
                with tla.vec.func(mode="simt", thread_block_dim=VL * TSPLIT):
                    tid, _, _ = tla.arch.thread_idx()
                    tk = tid // TSPLIT
                    tp = tid % TSPLIT
                    for k in tla.range_constexpr(MF // TSPLIT):
                        jq = tp * (MF // TSPLIT) + k
                        ub_tk[tk, jq + (0 if jq < 4 else (4 if jq < 8 else 8))] = ub_out[jq, tk]
                tla.pipe_barrier(tla.pipes.ALL)
                m_c.lock(pipe=tla.arch.MTE3)
                tla.copy(
                    tla.tile_view(gm_prev, tla.make_shape(VL, 4), tla.make_coord(g, 0)),
                    tla.make_tensor(
                        tk_p + 0,
                        tla.make_layout(tla.make_shape(VL, 4), tla.make_stride(32, 1)),
                    ),
                )
                tla.copy(
                    tla.tile_view(gm_post, tla.make_shape(VL, 4), tla.make_coord(g, 0)),
                    tla.make_tensor(
                        tk_p + 8,
                        tla.make_layout(tla.make_shape(VL, 4), tla.make_stride(32, 1)),
                    ),
                )
                tla.copy(
                    tla.tile_view(gm_comb, tla.make_shape(VL, 16), tla.make_coord(g, 0)),
                    tla.make_tensor(
                        tk_p + 16,
                        tla.make_layout(tla.make_shape(VL, 16), tla.make_stride(32, 1)),
                    ),
                )
                m_c.unlock(pipe=tla.arch.MTE3)

        # y finish for the remaining blocks: y = u * gv * w, overwriting u in place.
        # Only tokens split over several cores ("cold") arrive here, because the gv
        # denominator (the whole token's u^2) can only be reduced in K3; hot tokens
        # already finished y in K1.  Cold tokens are handed to NYB blocks as
        # contiguous token ranges (host exact count <= 62) and hot ones are skipped.
        # With NYB == 0 this branch never runs, so K3 only launches NCOEF blocks and
        # no block loads w without work.
        if bid >= NCOEF:
            m_in.lock(pipe=tla.arch.MTE2)
            tla.copy(
                ub_w,
                tla.tile_view(gm_w, tla.make_shape(1, H), tla.make_coord(0, 0)),
            )
            m_in.unlock(pipe=tla.arch.MTE2)
            # NYB is a constexpr; NYB == 0 makes this branch unreachable
            # (block_num = NCOEF), and 1 only keeps the constant division valid.
            nyb = NYB if NYB > 0 else 1
            ylo = ((bid - NCOEF) * T) // nyb
            yhi = ((bid - NCOEF + 1) * T) // nyb
            for t in tla.range(ylo, yhi):
                # cold <=> the two chunks are on different cores, the complement of
                # K1's hot test; the core owning unit u is ((u+1)*NTSK-1)//(T*M) and
                # NTSK matches K1's NT.  With M != 2 K1 has no hot path, so every
                # token is handled here.
                cold = 1
                if M == 2:
                    h0 = (((t * M + 1) * NTSK) - 1) // (T * M)
                    h1 = (((t + 1) * M * NTSK) - 1) // (T * M)
                    cold = 0
                    if h0 < h1:
                        cold = 1
                if cold == 1:
                    m_in.lock(pipe=tla.arch.MTE2)
                    tla.copy(
                        tla.tile_view(ub_pu, tla.make_shape(1, NSLOT), tla.make_coord(0, 0)),
                        tla.tile_view(
                            gm_puu, tla.make_shape(1, NSLOT), tla.make_coord(t, 0)
                        ),
                    )
                    tla.copy(
                        ub_u,
                        tla.tile_view(gm_yu, tla.make_shape(1, H), tla.make_coord(t, 0)),
                    )
                    m_in.unlock(pipe=tla.arch.MTE2)
                    m_in.lock(pipe=tla.arch.VECTOR)
                    m_c.lock(pipe=tla.arch.VECTOR)
                    m_out.lock(pipe=tla.arch.VECTOR)
                    with tla.vec.func(mode="simd"):
                        tla.local_mem_bar(
                            tla.params.MemType.VEC_STORE,
                            tla.params.MemType.VEC_LOAD,
                        )
                        sm, _ = tla.update_mask(NSLOT, dtype=tla.Float32)
                        acc = _ld1(ub_pu, 0)
                        su = acc.reduce(tla.ReductionOp.ADD, mask=sm)
                        gv = tla.div(
                            tla.full(1.0, dtype=tla.Float32),
                            tla.sqrt(
                                tla.add(
                                    tla.mul(tla.full(su, dtype=tla.Float32), 1.0 / H),
                                    7e-6,
                                )
                            ),
                        )
                        # Dynamic group loop: inlining the H/VL groups overflows ICache.
                        for gp in tla.range(0, H // VL):
                            uf = _bf(ub_u, 0, gp)
                            wf = _bf(ub_w, 0, gp)
                            _st_bf(ub_y, 0, gp, tla.mul(tla.mul(uf, gv), wf))
                    m_out.unlock(pipe=tla.arch.VECTOR)
                    m_c.unlock(pipe=tla.arch.VECTOR)
                    m_in.unlock(pipe=tla.arch.VECTOR)
                    m_out.lock(pipe=tla.arch.MTE3)
                    tla.copy(
                        tla.tile_view(gm_yu, tla.make_shape(1, H), tla.make_coord(t, 0)),
                        ub_y,
                    )
                    m_out.unlock(pipe=tla.arch.MTE3)


# ---------------------------------------------------------------------------
# Host planning
# ---------------------------------------------------------------------------


def _k1_ub_fits(ch, h):
    """UB budget of K1 in bytes: double-buffered res/x/r/u plus w and the y tile."""
    return 40 * ch + 4 * h + 2048 <= 184 * 1024


def _ncold(t, m, nb):
    """Tokens whose two chunks land on different cores when chunks are spread evenly."""
    ut = t * m
    n = 0
    for tok in range(t):
        h0 = (((tok * m + 1) * nb) - 1) // ut
        h1 = (((tok + 1) * m * nb) - 1) // ut
        if h0 < h1:
            n += 1
    return n


def _plan(t, h):
    # CH is fixed at h/2 (M=2) so one token is exactly two chunks and both can live
    # in one core's UB; that is what makes the hot path possible.  When the UB budget
    # or the 64-element alignment does not allow it, fall back to a cost model where
    # M != 2 has no hot path and K3 finishes every token.
    # Chunk assignment is a static choice on T: small T spreads chunks evenly (better
    # balance, at the cost of a few cold tokens), large T assigns whole tokens.
    k = 4 * h
    ch = None
    if h % 2 == 0 and (h // 2) % 64 == 0 and _k1_ub_fits(h // 2, h):
        ch = h // 2
    if ch is None:
        EFF = 1180
        cap = 2560 if h > 2560 else h
        cands = [c for c in range(1024, cap + 1, 64) if h % c == 0]
        if not cands:
            cands = [h if h <= 1024 else 1024]
        best = None
        for cand in cands:
            mm = h // cand
            nch = (t * mm + NBLK_VEC - 1) // NBLK_VEC
            ntok = (nch + mm - 1) // mm + 1
            load = nch * cand + ntok * EFF
            key = (load, 0 if mm >= 2 else 1, -cand)
            if best is None or key < best[0]:
                best = (key, cand, mm)
        _, ch, gpt = best
    else:
        gpt = h // ch
    m = h // ch
    pdiv = NSLOT
    bk = BK_G if k % BK_G == 0 else 128
    ku = k // bk
    bn = BN_G
    if t < BN_G:
        # Narrow the N tile for few tokens (16-aligned) so the padding does not
        # multiply the z read traffic.
        bn = max(16, ((t + 15) // 16) * 16)
    nt = (t + bn - 1) // bn
    splitk = max(1, SPLIT_MAX // nt)
    if splitk > ku:
        splitk = ku
    upc = (ku + splitk - 1) // splitk
    tpad = nt * bn
    ncoef = (t + VL - 1) // VL
    if ncoef > max(1, NBLK_VEC // 8):
        ncoef = max(1, NBLK_VEC // 8)
    # Task mapping, cold-token count and K3 block count.
    bal = 1 if (m == 2 and t <= 128) else 0
    ntsk = min(t * m, NBLK_VEC)
    if m != 2:
        ncold = t                      # no hot path: K3 finishes every token
    elif bal:
        ncold = _ncold(t, m, ntsk)
    else:
        ncold = 0                      # whole-token assignment makes every token hot
    nyb = min(ncold, NBLK_VEC - ncoef)
    # K3 reads the RMS partials one whole token group (P, VL) at a time, so the
    # column count must be rounded up to VL.
    pzpad = max(tpad, ((t + VL - 1) // VL) * VL)
    return {
        "k": k, "bk": bk, "ku": ku, "bn": bn, "nt": nt, "split": splitk,
        "upc": upc, "tpad": tpad, "ch": ch, "m": m, "p": pdiv, "gpt": gpt,
        "ncoef": ncoef, "pn": BM_G * tpad, "pzpad": pzpad,
        "bal": bal, "nyb": nyb, "k3blk": ncoef + nyb, "ntsk": ntsk,
    }


def _fn_len(p):
    return MF * p["k"]


def _cast_chunk(n):
    for chk in (2048, 1024, 512, 256, 128, 64, 32):
        if n % chk == 0:
            return chk
    raise RuntimeError("cast chunk not found for N={}".format(n))


def _desc(t, h, p, d):
    """Build the three kernel argument tuples shared by compile-time dummies and runtime."""
    from catlass.tla.runtime import from_dlpack as fdp

    rm = tla.arch.RowMajor
    cm = tla.arch.ColumnMajor

    k1 = (
        fdp(d["x"], layout_tag=rm),
        fdp(d["res"].view(t * 4, h), layout_tag=rm),
        fdp(d["pm"].view(t, 4), layout_tag=rm),
        fdp(d["post"].view(t, 4), layout_tag=rm),
        fdp(d["comb"].view(t, 16), layout_tag=rm),
        fdp(d["new_res"][:t].view(t * 4, h), layout_tag=rm),
        fdp(d["w"].view(1, h), layout_tag=rm),
        fdp(d["y"].view(t, h), layout_tag=rm),
        fdp(d["pzz"].view(p["p"], p["pzpad"]), layout_tag=rm),
        fdp(d["puu"].view(p["pzpad"], NSLOT), layout_tag=rm),
        fdp(d["part"].view(1, p["pn"]), layout_tag=rm),
        t, h, p["ch"], p["m"], p["pn"], p["bal"],
    )
    k2 = (
        fdp(d["fn_bf"].view(MF, p["k"]), layout_tag=rm),
        fdp(d["new_res"].view(p["tpad"], p["k"]), layout_tag=cm,
            origin_shape=(p["k"], p["tpad"])),
        fdp(d["part"].view(BM_G, p["tpad"]), layout_tag=rm),
        p["split"], p["nt"], p["ku"], p["upc"], p["bk"], p["bn"], SK_G,
    )
    k3 = (
        fdp(d["pzz"].view(p["p"], p["pzpad"]), layout_tag=rm),
        fdp(d["part"].view(BM_G, p["tpad"]), layout_tag=rm),
        fdp(d["scales"].view(1, 3), layout_tag=rm),
        fdp(d["bases"].view(1, 24), layout_tag=rm),
        fdp(d["new_prev"].view(p["tpad"], 4), layout_tag=rm),
        fdp(d["new_post"].view(p["tpad"], 4), layout_tag=rm),
        fdp(d["new_comb"].view(p["tpad"], 16), layout_tag=rm),
        fdp(d["y"].view(t, h), layout_tag=rm),
        fdp(d["w"].view(1, h), layout_tag=rm),
        fdp(d["puu"].view(p["pzpad"], NSLOT), layout_tag=rm),
        t, h, p["p"], p["ncoef"], p["m"], p["nyb"], p["ntsk"],
    )
    return k1, k2, k3


def _workspace(t, h, p, dev):
    """Allocate the per-shape GM workspace and the compile-time dummy tensors."""
    import torch

    bf = torch.bfloat16
    f32 = torch.float32
    n = _fn_len(p)
    hold = {
        "x": torch.empty(t, h, dtype=bf, device=dev),
        "res": torch.empty(t, 4, h, dtype=bf, device=dev),
        "pm": torch.empty(t, 4, dtype=f32, device=dev),
        "post": torch.empty(t, 4, 1, dtype=f32, device=dev),
        "comb": torch.empty(t, 4, 4, dtype=f32, device=dev),
        "new_res": torch.zeros(p["tpad"], 4, h, dtype=bf, device=dev),
        "w": torch.empty(h, dtype=bf, device=dev),
        "scales": torch.empty(3, dtype=f32, device=dev),
        "bases": torch.empty(24, dtype=f32, device=dev),
        "y": torch.empty(t, h, dtype=bf, device=dev),
        "new_coef": torch.empty(p["pzpad"], MF, dtype=f32, device=dev),
        "new_prev": torch.zeros(p["tpad"], 4, dtype=f32, device=dev),
        "new_post": torch.zeros(p["tpad"], 4, dtype=f32, device=dev),
        "new_comb": torch.zeros(p["tpad"], 16, dtype=f32, device=dev),
        "fn_bf": torch.empty(1, n, dtype=bf, device=dev),
        "fn_src": torch.empty(1, n, dtype=f32, device=dev),
        "pzz": torch.zeros(p["p"], p["pzpad"], dtype=f32, device=dev),
        "puu": torch.zeros(p["pzpad"], NSLOT, dtype=f32, device=dev),
        "part": torch.zeros(BM_G, p["tpad"], dtype=f32, device=dev),
    }
    from catlass.tla.runtime import from_dlpack as fdp

    rm = tla.arch.RowMajor
    chk = _cast_chunk(n)
    k0 = (
        fdp(hold["fn_src"].view(1, n), layout_tag=rm),
        fdp(hold["fn_bf"].view(1, n), layout_tag=rm),
        n, chk,
    )
    k1, k2, k3 = _desc(t, h, p, hold)
    return hold, chk, k0, k1, k2, k3

