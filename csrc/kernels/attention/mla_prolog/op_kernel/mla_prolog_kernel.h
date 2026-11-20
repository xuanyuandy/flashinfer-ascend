/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mla_prolog_kernel.h
 * \brief Device-side part of the A+E3 kernel, extracted verbatim from the
 *        standalone source (SHA256 443d75c7...). Only the two host-only
 *        includes were removed; the namespace body is byte-identical except
 *        for CATLASS_HOST_DEVICE annotations on ToUnderlyingArguments, which
 *        let the device entry rebuild Params without changing any math.
 *        Dsv41:: is a frozen implementation namespace, not the public repo
 *        operator name (public name: mla_prolog).
 */

#ifndef MLA_PROLOG_KERNEL_H
#define MLA_PROLOG_KERNEL_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "catlass/arch/arch.hpp"
#include "catlass/catlass.hpp"
#include "catlass/epilogue/block/block_epilogue.hpp"
#include "catlass/epilogue/dispatch_policy.hpp"
#include "catlass/epilogue/tile/tile_broadcast_mul.hpp"
#include "catlass/epilogue/tile/tile_broadcast_one_blk.hpp"
#include "catlass/epilogue/tile/tile_swizzle.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
#include "catlass/gemm/device/device_gemm.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/kernel/quant_matmul_multistage_workspace.hpp"
#include "catlass/layout/layout.hpp"
#include "catlass/status.hpp"
#include "kernel_operator.h"

#include <cstdint>

using namespace Catlass;

namespace Dsv41 {

constexpr uint32_t kHidden = 5120;
constexpr uint32_t kQa = 1280;
constexpr uint32_t kKv = 512;
// Local Q head count is a runtime value (1..64, one per TP shard); the fixed
// 64 is only the staging/capacity bound. Q-B output channels N are
// headNum * kHeadDim, never a compile-time constant.
constexpr uint32_t kMaxHeads = 64;
constexpr uint32_t kMinHeads = 1;
constexpr uint32_t kHeadDim = 512;
constexpr uint32_t kNopeDim = 448;
constexpr uint32_t kRopeDim = 64;
constexpr uint32_t kCacheRowsPerPage = 128;
constexpr uint32_t kMaxT = 4096;
constexpr uint32_t kWorkspaceStages = 2;
// Split-K of the S0 dual projection: K=5120 is divided into per-split extents that
// are multiples of kProjectionSplitKTile (e.g. {1792, 1792, 1536} for 3 splits), so
// every GM offset stays 32B aligned and every BlockMmad K step stays full-tile.
// Tier mapping: T <= kSplitProjectionMaxT -> kProjectionSplitsSmall;
//   kSplitProjectionMaxT < T <= kSplitProjectionMidMaxT -> kProjectionSplitsMid;
//   otherwise 1 split. Mid tier is disabled when kSplitProjectionMidMaxT <= kSplitProjectionMaxT.
constexpr uint32_t kProjectionSplitKTile = 256U;
constexpr uint32_t kMaxProjectionSplits = 3;
constexpr uint32_t kSplitProjectionMaxT = 128;
constexpr uint32_t kProjectionSplitsSmall = 3;
constexpr uint32_t kSplitProjectionMidMaxT = 128;
constexpr uint32_t kProjectionSplitsMid = 2;
// Narrow-M rebalance of the split=1 tier: for kSplitProjectionMaxT < T <=
// kNarrowProjectionMaxT the BM=128 grid only yields ceil(T/128) M-tiles x 7
// projection units (14 tiles on 24 AIC for T in (128, 256]) and the busiest
// tile carries a full 128-row M extent, even when the tail M-tile is mostly
// empty (T=191/192 -> tiles {128, 63/64}). Switching those T to a BM=64
// L1/L0 tile yields ceil(T/64) = 3 M-tiles x 7 = 21 tiles <= 24 AIC, so every
// AIC still runs at most one tile while the busiest M extent halves
// (64 rows). Split stays 1, so no extra S1 merge cost and no workspace
// change. T <= kSplitProjectionMaxT (incl. the T=96 main case) keeps the
// proven BM=128 + split=3 grid bit-identically: CATLASS already crops MMAD
// and the A-tile copy to the actual M extent (block_mmad_pingpong.hpp uses
// mPartActual / GetTileLayout(actualShape.m(), ...)), so shrinking BM cannot
// speed up the saturated 21-tile grid; it would only double the B-tile
// traffic. T > kNarrowProjectionMaxT keeps BM=128: BM=64 there produces 4x7
// = 28 tiles > 24 AIC and the busiest AIC would chain two tiles (same M
// work as one BM=128 tile) plus twice the B traffic.
constexpr uint32_t kNarrowProjectionMaxT = 192;
constexpr uint32_t kNarrowProjectionMTile = 64;
static_assert(kHidden % kProjectionSplitKTile == 0, "split-K tile must divide kHidden");
static_assert(kProjectionSplitsSmall >= 1U && kProjectionSplitsSmall <= kMaxProjectionSplits, "bad small tier splits");
static_assert(kProjectionSplitsMid >= 1U && kProjectionSplitsMid <= kMaxProjectionSplits, "bad mid tier splits");
static_assert(kProjectionSplitsMid <= kProjectionSplitsSmall || kSplitProjectionMidMaxT <= kSplitProjectionMaxT,
    "split count should not increase with T");
constexpr bool kEnableRopeScatter = true;
constexpr float kProjectionUlpGuard = 1.000006F;
constexpr float kProjectionGeneralGuard = 0.99999994F;
constexpr float kRmsNormUlpGuard = 1.0000055F;

CATLASS_HOST_DEVICE constexpr uint64_t Align512(uint64_t value)
{
    return (value + 511U) & ~uint64_t(511U);
}

// K base (start offset in the K dimension) of split `splitId` when kHidden is split
// into `splits` K-tile aligned extents. First (kTiles % splits) splits get one extra
// K tile. splits=2 -> {2560, 2560} (identical to the historic even split).
CATLASS_HOST_DEVICE inline uint32_t ProjectionSplitKBase(uint32_t splits, uint32_t splitId)
{
    const uint32_t kTiles = kHidden / kProjectionSplitKTile;
    const uint32_t base = kTiles / splits;
    const uint32_t extra = kTiles % splits;
    const uint32_t tilesBefore = (splitId < extra)
        ? splitId * (base + 1U)
        : extra * (base + 1U) + (splitId - extra) * base;
    return tilesBefore * kProjectionSplitKTile;
}

// K extent of split `splitId`; always a multiple of kProjectionSplitKTile.
CATLASS_HOST_DEVICE inline uint32_t ProjectionSplitKSize(uint32_t splits, uint32_t splitId)
{
    const uint32_t kTiles = kHidden / kProjectionSplitKTile;
    const uint32_t base = kTiles / splits;
    const uint32_t extra = kTiles % splits;
    return (splitId < extra ? base + 1U : base) * kProjectionSplitKTile;
}

struct WorkspaceLayout {
    uint64_t qaOffset;
    uint64_t qaBytes;
    uint64_t kvOffset;
    uint64_t kvBytes;
    uint64_t kvNormOffset;
    uint64_t kvNormBytes;
    uint64_t qInt8Offset;
    uint64_t qInt8Bytes;
    uint64_t qScaleOffset;
    uint64_t qScaleBytes;
    uint64_t wScaleFpOffset;
    uint64_t wScaleFpBytes;
    uint64_t qmmOffset;
    uint64_t qmmBytes;
    uint64_t totalBytes;
};

inline WorkspaceLayout MakeWorkspaceLayout(uint32_t t, uint32_t aicCount,
                                           uint32_t headNum)
{
    WorkspaceLayout layout{};
    layout.qaOffset = 0;
    // The qa/kv projection regions are sized exactly as before
    // (kMaxProjectionSplits * t * dim floats); only their interpretation
    // changed: they now hold the interleaved [t][split][dim] row layout
    // (row stride = kMaxProjectionSplits * dim) instead of the old
    // split-major [split][t][dim] blocks. Total bytes are identical.
    layout.qaBytes = Align512(
        static_cast<uint64_t>(kMaxProjectionSplits) * t * kQa * sizeof(float));
    layout.kvOffset = layout.qaOffset + layout.qaBytes;
    layout.kvBytes = Align512(
        static_cast<uint64_t>(kMaxProjectionSplits) * t * kKv * sizeof(float));
    layout.kvNormOffset = layout.kvOffset + layout.kvBytes;
    layout.kvNormBytes = Align512(static_cast<uint64_t>(t) * kKv * sizeof(uint16_t));
    layout.qInt8Offset = layout.kvNormOffset + layout.kvNormBytes;
    layout.qInt8Bytes = Align512(static_cast<uint64_t>(t) * kQa * sizeof(int8_t));
    layout.qScaleOffset = layout.qInt8Offset + layout.qInt8Bytes;
    layout.qScaleBytes = Align512(static_cast<uint64_t>(t) * sizeof(float));
    layout.wScaleFpOffset = layout.qScaleOffset + layout.qScaleBytes;
    // One FP32 weight-dequant scale per Q-B output channel (headNum * 512).
    layout.wScaleFpBytes = Align512(
        static_cast<uint64_t>(headNum) * kHeadDim * sizeof(float));
    layout.qmmOffset = layout.wScaleFpOffset + layout.wScaleFpBytes;
    layout.qmmBytes = Align512(
        static_cast<uint64_t>(aicCount) * kWorkspaceStages * 128U * 256U * sizeof(int32_t));
    layout.totalBytes = Align512(layout.qmmOffset + layout.qmmBytes);
    return layout;
}

using Bf16MatmulDispatch = Gemm::MmadAtlasA2Pingpong<true>;
using Bf16L1Tile = GemmShape<128, 256, 256>;
using Bf16L0Tile = GemmShape<128, 256, 64>;
using Bf16A = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
using Bf16B = Gemm::GemmType<bfloat16_t, layout::ColumnMajor>;
using Bf16C = Gemm::GemmType<float, layout::RowMajor>;
using Bf16BlockMmad = Gemm::Block::BlockMmad<
    Bf16MatmulDispatch, Bf16L1Tile, Bf16L0Tile, Bf16A, Bf16B, Bf16C>;
using Bf16Scheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
// Narrow-M rebalance granularity for the T in (kSplitProjectionMaxT,
// kNarrowProjectionMaxT] tier: the single BM=128 BlockMmad instance above is
// reused; only the scheduler grid granularity changes (MakeCoord(64, 256)).
// CATLASS crops the A-tile GM copy, the MMAD m extent and the L0C epilogue to
// the actual block shape (block_mmad_pingpong.hpp: GetTileLayout(actual m, k),
// mPartActual), which is exactly how the M=96 tail tile already runs on the
// BM=128 grid, so an M=64 scheduled tile is legal with the full 128-row L1/L0
// tile as the static capacity bound. No second BlockMmad instantiation:
// a dual instantiation was measured to inflate the kernel image (~+35KiB on
// the .o) and add a per-launch fixed cost that regressed every T by 2.5-15%.

struct DualProjectionKernel {
    using ElementA = typename Bf16BlockMmad::ElementA;
    using LayoutA = typename Bf16BlockMmad::LayoutA;
    using ElementB = typename Bf16BlockMmad::ElementB;
    using LayoutB = typename Bf16BlockMmad::LayoutB;
    using ElementC = typename Bf16BlockMmad::ElementC;
    using LayoutC = typename Bf16BlockMmad::LayoutC;

    struct Params {
        GemmCoord qaShape;
        GemmCoord kvShape;
        GM_ADDR x;
        LayoutA layoutA;
        GM_ADDR wQa;
        LayoutB layoutQaB;
        GM_ADDR wKv;
        LayoutB layoutKvB;
        GM_ADDR qa;
        LayoutC layoutQaC;
        GM_ADDR kv;
        LayoutC layoutKvC;
        uint32_t projectionSplits;
        uint32_t narrowM;
    };

    struct Arguments {
        GemmCoord qaShape;
        GemmCoord kvShape;
        GM_ADDR x;
        GM_ADDR wQa;
        GM_ADDR wKv;
        GM_ADDR qa;
        GM_ADDR kv;
        uint32_t projectionSplits;
        uint32_t narrowM;
    };

    static bool CanImplement(const Arguments &args)
    {
        return args.x != nullptr && args.wQa != nullptr && args.wKv != nullptr &&
            args.qa != nullptr && args.kv != nullptr &&
            args.projectionSplits > 0 &&
            args.projectionSplits <= kMaxProjectionSplits &&
            kHidden / kProjectionSplitKTile >= args.projectionSplits &&
            args.narrowM <= 1U &&
            (args.narrowM == 0U || args.projectionSplits == 1U);
    }

    static size_t GetWorkspaceSize(const Arguments &)
    {
        return 0;
    }

    CATLASS_HOST_DEVICE
    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *)
    {
        return Params{
            args.qaShape,
            args.kvShape,
            args.x,
            LayoutA::template MakeLayout<ElementA>(args.qaShape.m(), args.qaShape.k()),
            args.wQa,
            LayoutB::template MakeLayout<ElementB>(args.qaShape.k(), args.qaShape.n()),
            args.wKv,
            LayoutB::template MakeLayout<ElementB>(args.kvShape.k(), args.kvShape.n()),
            args.qa,
            // Interleaved partial layout [t][split][dim]: one token row owns
            // kMaxProjectionSplits * n contiguous floats and split `s` of that
            // row starts at an in-row offset of s * n (element units), so the
            // row stride (ldm) is kMaxProjectionSplits * n. Byte-for-byte the
            // same region size as the old [split][t][dim] layout.
            LayoutC(
                args.qaShape.m(), args.qaShape.n() * kMaxProjectionSplits,
                static_cast<typename LayoutC::LongIndex>(args.qaShape.n()) *
                    kMaxProjectionSplits),
            args.kv,
            LayoutC(
                args.kvShape.m(), args.kvShape.n() * kMaxProjectionSplits,
                static_cast<typename LayoutC::LongIndex>(args.kvShape.n()) *
                    kMaxProjectionSplits),
            args.projectionSplits,
            args.narrowM};
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(const Params &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(const Params &params)
    {
        // Scheduler M granularity: 64 on the narrow-M rebalance tier
        // (T in (128, 192], split=1), otherwise the native L1 tile M. The
        // BlockMmad instance and its static L1/L0 capacity are unchanged;
        // CATLASS crops A copy / MMAD / epilogue to the actual M extent.
        const uint32_t mTile =
            (params.narrowM != 0U) ? kNarrowProjectionMTile : Bf16L1Tile::M;
        // Loop counts only depend on the M/N tile grid; K extent per split varies
        // (K-tile aligned), so the outer schedulers use the split-0 extent.
        const GemmCoord qaSplitShape{
            params.qaShape.m(), params.qaShape.n(),
            ProjectionSplitKSize(params.projectionSplits, 0U)};
        const GemmCoord kvSplitShape{
            params.kvShape.m(), params.kvShape.n(),
            ProjectionSplitKSize(params.projectionSplits, 0U)};
        Bf16Scheduler qaScheduler(
            qaSplitShape, MakeCoord(mTile, Bf16L1Tile::N));
        Bf16Scheduler kvScheduler(
            kvSplitShape, MakeCoord(mTile, Bf16L1Tile::N));
        const uint32_t qaLoops = qaScheduler.GetCoreLoops();
        const uint32_t kvLoops = kvScheduler.GetCoreLoops();
        const uint32_t totalLoops =
            (qaLoops + kvLoops) * params.projectionSplits;

        Arch::Resource<Arch::AtlasA2> resource;
        Bf16BlockMmad blockMmad(resource);
        AscendC::GlobalTensor<ElementA> gmX;
        AscendC::GlobalTensor<ElementB> gmWQa;
        AscendC::GlobalTensor<ElementB> gmWKv;
        AscendC::GlobalTensor<ElementC> gmQa;
        AscendC::GlobalTensor<ElementC> gmKv;
        gmX.SetGlobalBuffer(reinterpret_cast<__gm__ ElementA *>(params.x));
        gmWQa.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.wQa));
        gmWKv.SetGlobalBuffer(reinterpret_cast<__gm__ ElementB *>(params.wKv));
        gmQa.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.qa));
        gmKv.SetGlobalBuffer(reinterpret_cast<__gm__ ElementC *>(params.kv));

        for (uint32_t loopIdx = AscendC::GetBlockIdx(); loopIdx < totalLoops;
             loopIdx += AscendC::GetBlockNum()) {
            const uint32_t splitId = loopIdx % params.projectionSplits;
            const uint32_t combinedLoop = loopIdx / params.projectionSplits;
            const bool isQa = combinedLoop < qaLoops;
            const uint32_t localLoop = isQa ? combinedLoop : combinedLoop - qaLoops;
            const uint32_t splitKBase =
                ProjectionSplitKBase(params.projectionSplits, splitId);
            const uint32_t splitK =
                ProjectionSplitKSize(params.projectionSplits, splitId);
            const GemmCoord qaProblem{params.qaShape.m(), params.qaShape.n(), splitK};
            const GemmCoord kvProblem{params.kvShape.m(), params.kvShape.n(), splitK};
            const GemmCoord problemShape = isQa ? qaProblem : kvProblem;
            Bf16Scheduler scheduler(
                problemShape, MakeCoord(mTile, Bf16L1Tile::N));
            const GemmCoord blockCoord = scheduler.GetBlockCoord(localLoop);
            const GemmCoord actualBlockShape =
                scheduler.GetActualBlockShape(blockCoord);
            const MatrixCoord offsetA{
                blockCoord.m() * mTile,
                splitKBase + blockCoord.k() * Bf16L1Tile::K};
            const MatrixCoord offsetB{
                splitKBase + blockCoord.k() * Bf16L1Tile::K,
                blockCoord.n() * Bf16L1Tile::N};
            const MatrixCoord offsetC{
                blockCoord.m() * mTile,
                blockCoord.n() * Bf16L1Tile::N};
            if (isQa) {
                // In-row split offset of the [t][split][dim] layout (elements).
                const uint64_t splitOffset =
                    static_cast<uint64_t>(splitId) * params.qaShape.n();
                blockMmad(
                    gmX[params.layoutA.GetOffset(offsetA)], params.layoutA,
                    gmWQa[params.layoutQaB.GetOffset(offsetB)], params.layoutQaB,
                    gmQa[splitOffset + params.layoutQaC.GetOffset(offsetC)], params.layoutQaC,
                    actualBlockShape);
            } else {
                // In-row split offset of the [t][split][dim] layout (elements).
                const uint64_t splitOffset =
                    static_cast<uint64_t>(splitId) * params.kvShape.n();
                blockMmad(
                    gmX[params.layoutA.GetOffset(offsetA)], params.layoutA,
                    gmWKv[params.layoutKvB.GetOffset(offsetB)], params.layoutKvB,
                    gmKv[splitOffset + params.layoutKvC.GetOffset(offsetC)], params.layoutKvC,
                    actualBlockShape);
            }
        }
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(const Params &)
    {}
};

struct NormQuantKernel {
    struct Params {
        GM_ADDR qa;
        GM_ADDR kv;
        GM_ADDR kvNorm;
        GM_ADDR gammaQa;
        GM_ADDR gammaKv;
        GM_ADDR qInt8;
        GM_ADDR qScale;
        GM_ADDR wScale;
        GM_ADDR wScaleFp;
        uint32_t t;
        uint32_t projectionSplits;
        // Required BF16 [t,1280] output: the same normalized BF16 row that
        // feeds dynamic quantization, i.e. q_norm(wq_a(x)).
        GM_ADDR qr;
        // FP32 RMSNorm epsilon applied to both the Q and the KV branch.
        float rmsNormEps;
        // Local Q head count; w_qb_scale has headNum * kHeadDim entries.
        uint32_t headNum;
    };
    using Arguments = Params;

    static bool CanImplement(const Arguments &args)
    {
        return args.qa != nullptr && args.kv != nullptr && args.kvNorm != nullptr &&
            args.gammaQa != nullptr &&
            args.gammaKv != nullptr && args.qInt8 != nullptr && args.qScale != nullptr &&
            args.wScale != nullptr && args.wScaleFp != nullptr && args.qr != nullptr &&
            args.rmsNormEps > 0.0F &&
            args.t > 0 && args.t <= kMaxT &&
            args.headNum >= kMinHeads && args.headNum <= kMaxHeads &&
            args.projectionSplits > 0 &&
            args.projectionSplits <= kMaxProjectionSplits;
    }

    static size_t GetWorkspaceSize(const Arguments &)
    {
        return 0;
    }

    CATLASS_HOST_DEVICE
    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *)
    {
        return args;
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(const Params &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(const Params &)
    {}

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(const Params &params)
    {
        constexpr uint32_t kGammaQaBf16 = 0;
        constexpr uint32_t kGammaKvBf16 = 2560;
        constexpr uint32_t kGammaQaFp = 3584;
        constexpr uint32_t kGammaKvFp = 8704;
        constexpr uint32_t kRowBf16 = 10752;
        constexpr uint32_t kNormBf16 = 13312;
        constexpr uint32_t kFp = 15872;
        constexpr uint32_t kTmp = 20992;
        constexpr uint32_t kScratch = 26112;
        constexpr uint32_t kQuantInt32 = 42496;
        constexpr uint32_t kQuantHalf = 47616;
        constexpr uint32_t kQuantInt8 = 50176;
        constexpr uint32_t kScalar = 51456;
        constexpr uint32_t kScaleOut = 51712;
        // Landing zone for the interleaved partial row [split0|split1|split2]:
        // kMaxProjectionSplits * kQa floats = 15360 B, 32B aligned, tail of the
        // 192KB VECCALC buffer (previous high-water mark was kScaleOut + 4 B).
        constexpr uint32_t kMerge = 51776;

        Arch::Resource<Arch::AtlasA2> resource;
        auto gammaQaBf16 = resource.ubBuf.GetBufferByByte<bfloat16_t>(kGammaQaBf16);
        auto gammaKvBf16 = resource.ubBuf.GetBufferByByte<bfloat16_t>(kGammaKvBf16);
        auto gammaQaFp = resource.ubBuf.GetBufferByByte<float>(kGammaQaFp);
        auto gammaKvFp = resource.ubBuf.GetBufferByByte<float>(kGammaKvFp);
        auto rowBf16 = resource.ubBuf.GetBufferByByte<bfloat16_t>(kRowBf16);
        auto normBf16 = resource.ubBuf.GetBufferByByte<bfloat16_t>(kNormBf16);
        auto fp = resource.ubBuf.GetBufferByByte<float>(kFp);
        auto tmp = resource.ubBuf.GetBufferByByte<float>(kTmp);
        auto scratch = resource.ubBuf.GetBufferByByte<float>(kScratch);
        auto quantInt32 = resource.ubBuf.GetBufferByByte<int32_t>(kQuantInt32);
        auto quantHalf = resource.ubBuf.GetBufferByByte<half>(kQuantHalf);
        auto quantInt8 = resource.ubBuf.GetBufferByByte<int8_t>(kQuantInt8);
        auto scalar = resource.ubBuf.GetBufferByByte<float>(kScalar);
        auto scaleOut = resource.ubBuf.GetBufferByByte<float>(kScaleOut);
        auto mergeBuf = resource.ubBuf.GetBufferByByte<float>(kMerge);

        AscendC::GlobalTensor<float> gmQa;
        AscendC::GlobalTensor<float> gmKv;
        AscendC::GlobalTensor<bfloat16_t> gmKvNorm;
        AscendC::GlobalTensor<bfloat16_t> gmGammaQa;
        AscendC::GlobalTensor<bfloat16_t> gmGammaKv;
        AscendC::GlobalTensor<int8_t> gmQInt8;
        AscendC::GlobalTensor<float> gmQScale;
        AscendC::GlobalTensor<bfloat16_t> gmWScale;
        AscendC::GlobalTensor<float> gmWScaleFp;
        AscendC::GlobalTensor<bfloat16_t> gmQr;
        gmQa.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.qa));
        gmKv.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.kv));
        gmKvNorm.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.kvNorm));
        gmGammaQa.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.gammaQa));
        gmGammaKv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.gammaKv));
        gmQInt8.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(params.qInt8));
        gmQScale.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.qScale));
        gmWScale.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.wScale));
        gmWScaleFp.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.wScaleFp));
        gmQr.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.qr));

        auto *pipe = GetTPipePtr();
        const int32_t mte2V = pipe->AllocEventID<AscendC::HardEvent::MTE2_V>();
        const int32_t vMte3 = pipe->AllocEventID<AscendC::HardEvent::V_MTE3>();
        const int32_t mte3V = pipe->AllocEventID<AscendC::HardEvent::MTE3_V>();
        const int32_t vMte2 = pipe->AllocEventID<AscendC::HardEvent::V_MTE2>();
        const int32_t vS = pipe->AllocEventID<AscendC::HardEvent::V_S>();
        const int32_t sV = pipe->AllocEventID<AscendC::HardEvent::S_V>();
        const int32_t sMte3 = pipe->AllocEventID<AscendC::HardEvent::S_MTE3>();

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum =
            AscendC::GetBlockNum() * AscendC::GetSubBlockNum();

        // This conversion is independent of the projection output. Run it while
        // the AIC side is still producing the projection to hide its latency.
        // headNum * kHeadDim is always a multiple of kScaleChunk.
        constexpr uint32_t kScaleChunk = 256;
        const uint32_t qbOut = params.headNum * kHeadDim;
        for (uint32_t chunk = blockIdx; chunk < qbOut / kScaleChunk;
             chunk += blockNum) {
            const uint32_t start = chunk * kScaleChunk;
            AscendC::DataCopy(gammaQaBf16, gmWScale[start], kScaleChunk);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
            AscendC::Cast(
                gammaQaFp, gammaQaBf16,
                AscendC::RoundMode::CAST_NONE, kScaleChunk);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3);
            AscendC::DataCopy(gmWScaleFp[start], gammaQaFp, kScaleChunk);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3V);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3V);
            AscendC::PipeBarrier<PIPE_ALL>();
        }

        AscendC::DataCopy(gammaQaBf16, gmGammaQa, 512);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::DataCopy(gammaQaBf16[512], gmGammaQa[512], 512);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::DataCopy(gammaQaBf16[1024], gmGammaQa[1024], 256);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::DataCopy(gammaKvBf16, gmGammaKv, kKv);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
        AscendC::Cast(gammaQaFp, gammaQaBf16, AscendC::RoundMode::CAST_NONE, 512);
        AscendC::Cast(gammaQaFp[512], gammaQaBf16[512], AscendC::RoundMode::CAST_NONE, 512);
        AscendC::Cast(gammaQaFp[1024], gammaQaBf16[1024], AscendC::RoundMode::CAST_NONE, 256);
        AscendC::Cast(gammaKvFp, gammaKvBf16, AscendC::RoundMode::CAST_NONE, kKv);
        AscendC::PipeBarrier<PIPE_V>();

        Arch::CrossCoreFlag projectionReady{4};
        Arch::CrossCoreWaitFlag(projectionReady);
        Arch::CrossCoreBarrier<0x0, PIPE_MTE2>();

        const uint32_t workCount = params.t * 2U;
        for (uint32_t work = blockIdx; work < workCount; work += blockNum) {
            const bool isQ = work < params.t;
            const uint32_t row = isQ ? work : work - params.t;
            const uint32_t dim = isQ ? kQa : kKv;
            const uint64_t offset = static_cast<uint64_t>(row) * dim;
            auto gmProjection = isQ ? gmQa : gmKv;
            // Interleaved partial layout: token `row` owns
            // kMaxProjectionSplits * dim contiguous floats and its split `s`
            // starts at rowBase + s * dim. The P live partials are therefore
            // P * dim *contiguous* floats, fetched with a SINGLE DataCopy
            // (one MTE2 descriptor, one MTE2->V wait) instead of P copies.
            const uint64_t rowBase =
                static_cast<uint64_t>(row) * kMaxProjectionSplits * dim;
            const bool multiSplit = params.projectionSplits > 1U;
            const uint32_t vectorChunks = (dim + 511U) / 512U;
            if (multiSplit) {
                AscendC::DataCopy(
                    mergeBuf, gmProjection[rowBase],
                    params.projectionSplits * dim);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
            }
            for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                const uint32_t start = chunkId * 512U;
                const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                if (!multiSplit) {
                    // P == 1: exactly one partial, plain contiguous row copy
                    // into `fp` (bit-identical to the pre-interleave path).
                    AscendC::DataCopy(
                        fp[start], gmProjection[rowBase + start], count);
                    AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(mte2V);
                    AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(mte2V);
                } else {
                    // Fixed reduction order split0 -> split1 -> split2, i.e.
                    // ((p0 + p1) + p2) in FP32: identical operands and
                    // identical addition order as the serial per-split path.
                    AscendC::Add(
                        fp[start], mergeBuf[start], mergeBuf[dim + start], count);
                    if (params.projectionSplits > 2U) {
                        AscendC::Add(
                            fp[start], fp[start], mergeBuf[2U * dim + start], count);
                    }
                }
            }
            if (multiSplit) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vMte2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vMte2);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (params.t == 1U) {
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Muls(
                        fp[start], fp[start], kProjectionUlpGuard, count);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                const uint32_t start = chunkId * 512U;
                const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                AscendC::Cast(
                    rowBf16[start], fp[start], AscendC::RoundMode::CAST_RINT, count);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (isQ && params.t > 1U) {
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Muls(
                        fp[start], fp[start], kProjectionGeneralGuard, count);
                    AscendC::Cast(
                        normBf16[start], fp[start], AscendC::RoundMode::CAST_RINT, count);
                }
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Cast(
                        fp[start], normBf16[start], AscendC::RoundMode::CAST_NONE, count);
                    AscendC::Mul(tmp[start], fp[start], fp[start], count);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                const uint32_t start = chunkId * 512U;
                const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                AscendC::Cast(
                    fp[start], rowBf16[start], AscendC::RoundMode::CAST_NONE, count);
            }
            AscendC::PipeBarrier<PIPE_V>();
            if (!isQ || params.t == 1U) {
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Mul(tmp[start], fp[start], fp[start], count);
                }
                AscendC::PipeBarrier<PIPE_V>();
            }
            AscendC::ReduceSum(scalar, tmp, scratch, static_cast<int32_t>(dim));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vS);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vS);
            const float invDim = isQ ? (1.0F / 1280.0F) : (1.0F / 512.0F);
            AscendC::Muls(scalar, scalar, invDim, 1);
            AscendC::Adds(scalar, scalar, params.rmsNormEps, 1);
            AscendC::Sqrt(scalar[8], scalar, 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(scalar[16], 1.0F, 1);
            AscendC::Div(scalar, scalar[16], scalar[8], 1);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::SetFlag<AscendC::HardEvent::V_S>(vS);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(vS);
            const float rstd = scalar.GetValue(0) *
                (params.t == 1U ? kRmsNormUlpGuard : 1.0F);
            AscendC::SetFlag<AscendC::HardEvent::S_V>(sV);
            AscendC::WaitFlag<AscendC::HardEvent::S_V>(sV);
            for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                const uint32_t start = chunkId * 512U;
                const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                AscendC::Muls(fp[start], fp[start], rstd, count);
                AscendC::Mul(
                    fp[start], fp[start],
                    (isQ ? gammaQaFp : gammaKvFp)[start], count);
            }
            AscendC::PipeBarrier<PIPE_V>();
            for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                const uint32_t start = chunkId * 512U;
                const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                AscendC::Cast(
                    normBf16[start], fp[start], AscendC::RoundMode::CAST_RINT, count);
            }
            AscendC::PipeBarrier<PIPE_V>();

            if (isQ) {
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Cast(
                        fp[start], normBf16[start], AscendC::RoundMode::CAST_NONE, count);
                }
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Abs(tmp[start], fp[start], count);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::ReduceMax(
                    scalar, tmp, scratch, static_cast<int32_t>(dim), false);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetFlag<AscendC::HardEvent::V_S>(vS);
                AscendC::WaitFlag<AscendC::HardEvent::V_S>(vS);
                const float rowMax = scalar.GetValue(0);
                float outputScale = 0.0F;
                float quantMultiplier = 0.0F;
                if (rowMax != 0.0F) {
                    AscendC::Muls(scaleOut, scalar, 1.0F / 127.0F, 1);
                    AscendC::Duplicate(scalar[8], 127.0F, 1);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Div(scalar[16], scalar[8], scalar, 1);
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::SetFlag<AscendC::HardEvent::V_S>(vS);
                    AscendC::WaitFlag<AscendC::HardEvent::V_S>(vS);
                    outputScale = scaleOut.GetValue(0);
                    quantMultiplier = scalar.GetValue(16);
                }
                AscendC::SetFlag<AscendC::HardEvent::S_V>(sV);
                AscendC::WaitFlag<AscendC::HardEvent::S_V>(sV);
                if (rowMax == 0.0F) {
                    for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                        const uint32_t start = chunkId * 512U;
                        const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                        AscendC::Duplicate(fp[start], 0.0F, count);
                    }
                } else {
                    for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                        const uint32_t start = chunkId * 512U;
                        const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                        AscendC::Muls(fp[start], fp[start], quantMultiplier, count);
                    }
                }
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Cast(
                        quantInt32[start], fp[start],
                        AscendC::RoundMode::CAST_RINT, count);
                }
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::SetDeqScale(half(1.0F));
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Cast(
                        quantHalf[start], quantInt32[start],
                        AscendC::RoundMode::CAST_ROUND, count);
                }
                AscendC::PipeBarrier<PIPE_V>();
                for (uint32_t chunkId = 0; chunkId < vectorChunks; ++chunkId) {
                    const uint32_t start = chunkId * 512U;
                    const uint32_t count = (start + 512U <= dim) ? 512U : dim - start;
                    AscendC::Cast(
                        quantInt8[start], quantHalf[start],
                        AscendC::RoundMode::CAST_TRUNC, count);
                }
                scaleOut.SetValue(0, outputScale);
                AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(sMte3);
                AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(sMte3);
                AscendC::DataCopyParams scaleCopy{1, sizeof(float), 0, 0};
                AscendC::DataCopyPad(gmQScale[row], scaleOut, scaleCopy);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3);
                AscendC::DataCopy(gmQInt8[offset], quantInt8, 512);
                AscendC::DataCopy(gmQInt8[offset + 512], quantInt8[512], 512);
                AscendC::DataCopy(gmQInt8[offset + 1024], quantInt8[1024], 256);
                // qr = q_norm(wq_a(x)): reuse the `normBf16` row that this
                // iteration already materialized for dynamic quantization, so
                // the required second output adds GM write traffic only (no
                // compute). The V_MTE3 flag above already orders every prior V
                // write (including normBf16) before these MTE3 copies, and the
                // MTE3_V wait below covers all six stores before UB reuse.
                AscendC::DataCopy(gmQr[offset], normBf16, 512);
                AscendC::DataCopy(gmQr[offset + 512], normBf16[512], 512);
                AscendC::DataCopy(gmQr[offset + 1024], normBf16[1024], 256);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3V);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3);
                AscendC::DataCopy(gmKvNorm[offset], normBf16, dim);
                AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(mte3V);
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(mte3V);
            }
            // P == 1: the next row iteration overwrites `fp` with an MTE2
            // DataCopy, while the V pipe may still be consuming the current
            // row. Close the V -> MTE2 dependency once per row, after the last
            // V access to `fp` and before the next row's DataCopy.
            if (!multiSplit) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vMte2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vMte2);
            }
        }

        pipe->ReleaseEventID<AscendC::HardEvent::MTE2_V>(mte2V);
        pipe->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vMte3);
        pipe->ReleaseEventID<AscendC::HardEvent::MTE3_V>(mte3V);
        pipe->ReleaseEventID<AscendC::HardEvent::V_MTE2>(vMte2);
        pipe->ReleaseEventID<AscendC::HardEvent::V_S>(vS);
        pipe->ReleaseEventID<AscendC::HardEvent::S_V>(sV);
        pipe->ReleaseEventID<AscendC::HardEvent::S_MTE3>(sMte3);
    }
};

using QuantDispatch = Gemm::MmadAtlasA2PreloadAsyncWithCallback<1, 2, 2, 2, 1, true, true>;
using QuantL1Tile = GemmShape<128, 256, 512>;
using QuantL0Tile = GemmShape<128, 256, 128>;
using QuantA = Gemm::GemmType<int8_t, layout::RowMajor>;
using QuantB = Gemm::GemmType<int8_t, layout::zN>;
using QuantC = Gemm::GemmType<int32_t, layout::RowMajor>;
using QuantBlockMmad = Gemm::Block::BlockMmad<
    QuantDispatch, QuantL1Tile, QuantL0Tile, QuantA, QuantB, QuantC>;
using QuantEpilogueDispatch = Epilogue::EpilogueAtlasA2PerTokenDequant<2>;
using QuantScale = Gemm::GemmType<float, layout::VectorLayout>;
using QuantTokenScale = Gemm::GemmType<float, layout::VectorLayout>;
using QuantD = Gemm::GemmType<bfloat16_t, layout::RowMajor>;
using QuantRowMul = Gemm::GemmType<float, layout::RowMajor>;
using QuantOneBlock = Gemm::GemmType<float, layout::RowMajor>;
using QuantColumnMul = Gemm::GemmType<float, layout::RowMajor>;
using QuantEpilogueTile = MatrixShape<32, 256>;
using QuantTileRowMul = Epilogue::Tile::TileRowBroadcastMul<
    Arch::AtlasA2, QuantRowMul, QuantEpilogueTile>;
using QuantTileOneBlock = Epilogue::Tile::TileBroadcastOneBlk<
    Arch::AtlasA2, QuantOneBlock, QuantEpilogueTile::ROW>;
using QuantTileColumnMul = Epilogue::Tile::TileOneBlkColumnBroadcastMul<
    Arch::AtlasA2, QuantColumnMul, QuantEpilogueTile>;
using QuantEpilogueCopy = Epilogue::Tile::TileCopy<
    Arch::AtlasA2, QuantC, QuantScale, QuantTokenScale, QuantD>;
using QuantTileScheduler = Epilogue::Tile::EpilogueHorizontalTileSwizzle;
using QuantBlockEpilogue = Epilogue::Block::BlockEpilogue<
    QuantEpilogueDispatch, QuantC, QuantScale, QuantTokenScale, QuantD,
    QuantTileRowMul, QuantTileOneBlock, QuantTileColumnMul, QuantEpilogueCopy,
    QuantTileScheduler>;
using QuantScheduler = Gemm::Block::GemmIdentityBlockSwizzle<3, 0>;
using QuantMatmul = Gemm::Kernel::QuantMatmulMultiStageWorkspace<
    QuantBlockMmad, QuantBlockEpilogue, QuantScheduler, kWorkspaceStages>;

struct RopeScatterKernel {
    struct Params {
        GM_ADDR qOutput;
        GM_ADDR kv;
        GM_ADDR ropeCos;
        GM_ADDR ropeSin;
        GM_ADDR slots;
        GM_ADDR cache;
        uint32_t t;
        // Local Q head count: qOutput is [t, headNum, kHeadDim].
        uint32_t headNum;
        // true (default): scatter the computed row into `cache` at the row
        // named by `slots` (fused behavior; `kvOut` is never written).
        // false: write the same [t,kHeadDim] BF16 row to the fresh `kvOut`
        // output and NEVER read `slots` or read/write `cache`. The Q branch is
        // unchanged in both modes.
        bool updateKvCache;
        GM_ADDR kvOut;
    };
    using Arguments = Params;

    static bool CanImplement(const Arguments &args)
    {
        return args.qOutput != nullptr && args.kv != nullptr && args.ropeCos != nullptr &&
            args.ropeSin != nullptr &&
            (args.updateKvCache
                 ? args.slots != nullptr && args.cache != nullptr
                 : args.kvOut != nullptr) &&
            args.t > 0 && args.t <= kMaxT &&
            args.headNum >= kMinHeads && args.headNum <= kMaxHeads;
    }

    static size_t GetWorkspaceSize(const Arguments &)
    {
        return 0;
    }

    CATLASS_HOST_DEVICE
    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *)
    {
        return args;
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(const Params &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(const Params &)
    {}

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(const Params &params)
    {
        constexpr uint32_t kHeadBatch = 4;
        constexpr uint32_t kBatchRope = kHeadBatch * kRopeDim;
        constexpr uint32_t kRow0 = 0;
        constexpr uint32_t kRow1 = 1024;
        constexpr uint32_t kX0 = 2048;
        constexpr uint32_t kX1 = 3072;
        constexpr uint32_t kRotated0 = 4096;
        constexpr uint32_t kRotated1 = 5120;
        constexpr uint32_t kOut0 = 6144;
        constexpr uint32_t kOut1 = 7168;
        constexpr uint32_t kCos = 8192;
        constexpr uint32_t kSin = 9216;
        constexpr uint32_t kSign = 10240;
        constexpr uint32_t kOffsets = 11264;
        // Whole-token staging for the headNum x kRopeDim rope slices: one
        // strided descriptor per token instead of one per 4-head batch. The
        // staging capacity stays at the 64-head bound even when headNum < 64:
        // kMaxHeads * kRopeDim * sizeof(bfloat16_t) = 8192 B.
        constexpr uint32_t kQRows = 12288;

        Arch::Resource<Arch::AtlasA2> resource;
        AscendC::LocalTensor<bfloat16_t> rowLocal[2] = {
            resource.ubBuf.GetBufferByByte<bfloat16_t>(kRow0),
            resource.ubBuf.GetBufferByByte<bfloat16_t>(kRow1)};
        AscendC::LocalTensor<float> x[2] = {
            resource.ubBuf.GetBufferByByte<float>(kX0),
            resource.ubBuf.GetBufferByByte<float>(kX1)};
        AscendC::LocalTensor<float> rotated[2] = {
            resource.ubBuf.GetBufferByByte<float>(kRotated0),
            resource.ubBuf.GetBufferByByte<float>(kRotated1)};
        AscendC::LocalTensor<float> output[2] = {
            resource.ubBuf.GetBufferByByte<float>(kOut0),
            resource.ubBuf.GetBufferByByte<float>(kOut1)};
        auto cos = resource.ubBuf.GetBufferByByte<float>(kCos);
        auto sin = resource.ubBuf.GetBufferByByte<float>(kSin);
        auto sign = resource.ubBuf.GetBufferByByte<float>(kSign);
        auto offsets = resource.ubBuf.GetBufferByByte<uint32_t>(kOffsets);
        auto qRows = resource.ubBuf.GetBufferByByte<bfloat16_t>(kQRows);

        AscendC::GlobalTensor<bfloat16_t> gmQ;
        AscendC::GlobalTensor<bfloat16_t> gmKv;
        AscendC::GlobalTensor<float> gmCos;
        AscendC::GlobalTensor<float> gmSin;
        AscendC::GlobalTensor<int32_t> gmSlots;
        AscendC::GlobalTensor<bfloat16_t> gmCache;
        AscendC::GlobalTensor<bfloat16_t> gmKvOut;
        gmQ.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.qOutput));
        gmKv.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.kv));
        gmCos.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.ropeCos));
        gmSin.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(params.ropeSin));
        if (params.updateKvCache) {
            // Fused mode: `slots`/`cache` are the only KV write target.
            gmSlots.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(params.slots));
            gmCache.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.cache));
        } else {
            // Diagnostic mode: `slots`/`cache` are never bound, so neither the
            // slot scalars nor any cache row can be read or written.
            gmKvOut.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(params.kvOut));
        }

        auto *pipe = GetTPipePtr();
        const int32_t inputReady[2] = {
            pipe->AllocEventID<AscendC::HardEvent::MTE2_V>(),
            pipe->AllocEventID<AscendC::HardEvent::MTE2_V>()};
        const int32_t vMte3 = pipe->AllocEventID<AscendC::HardEvent::V_MTE3>();
        const int32_t vMte2 = pipe->AllocEventID<AscendC::HardEvent::V_MTE2>();
        const int32_t slotFree[2] = {
            pipe->AllocEventID<AscendC::HardEvent::MTE3_MTE2>(),
            pipe->AllocEventID<AscendC::HardEvent::MTE3_MTE2>()};
        bool pending[2] = {false, false};

        for (uint32_t batch = 0; batch < kHeadBatch; ++batch) {
            for (uint32_t i = 0; i < kRopeDim / 2; ++i) {
                const uint32_t base = batch * kRopeDim;
                offsets.SetValue(base + 2U * i,
                    (base + 2U * i + 1U) * sizeof(float));
                offsets.SetValue(base + 2U * i + 1U,
                    (base + 2U * i) * sizeof(float));
                sign.SetValue(base + 2U * i, -1.0F);
                sign.SetValue(base + 2U * i + 1U, 1.0F);
            }
        }

        const uint32_t blockIdx = AscendC::GetBlockIdx();
        const uint32_t blockNum =
            AscendC::GetBlockNum() * AscendC::GetSubBlockNum();
        for (uint32_t token = blockIdx; token < params.t; token += blockNum) {
            const uint32_t headCount = params.headNum;
            const uint64_t ropeOffset = static_cast<uint64_t>(token) * kRopeDim;
            for (uint32_t batch = 0; batch < kHeadBatch; ++batch) {
                AscendC::DataCopy(
                    cos[batch * kRopeDim], gmCos[ropeOffset], kRopeDim);
                AscendC::DataCopy(
                    sin[batch * kRopeDim], gmSin[ropeOffset], kRopeDim);
            }
            // One strided descriptor per token for all headCount rope slices
            // (was: headCount/kHeadBatch descriptors of 4 rows each). The
            // descriptor shrinks with headCount; staging capacity is unchanged.
            // slotFree[1]/pending[1] guard this staging buffer (slot 1 is no
            // longer used by the per-batch row buffer).
            if (pending[1]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slotFree[1]);
                pending[1] = false;
            }
            const uint64_t qRopeOffset =
                static_cast<uint64_t>(token) * headCount * kHeadDim + kNopeDim;
            AscendC::DataCopy(
                qRows, gmQ[qRopeOffset],
                AscendC::DataCopyParams(headCount, 4U, 28U, 0U));
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady[0]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady[0]);
            AscendC::Mul(sin, sin, sign, kBatchRope);
            AscendC::PipeBarrier<PIPE_V>();

            for (uint32_t head = 0; head < headCount; head += kHeadBatch) {
                const uint32_t slot = (head / kHeadBatch) & 1U;
                AscendC::LocalTensor<bfloat16_t> rows =
                    qRows[(head / kHeadBatch) * kBatchRope];
                // Partial final batch for headNum not divisible by 4: only the
                // existing heads are cast/rotated, so the uninitialized tail of
                // the staging buffer is never touched.
                const uint32_t batchHeads =
                    (headCount - head < kHeadBatch) ? headCount - head : kHeadBatch;
                const uint32_t count = batchHeads * kRopeDim;

                AscendC::Cast(
                    x[slot], rows,
                    AscendC::RoundMode::CAST_NONE, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Mul(output[slot], x[slot], cos, count);
                AscendC::Gather(rotated[slot], x[slot], offsets, 0U, count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::FusedMulAdd(
                    rotated[slot], sin, output[slot], count);
                AscendC::PipeBarrier<PIPE_V>();
                AscendC::Cast(
                    rows, rotated[slot],
                    AscendC::RoundMode::CAST_RINT, count);
            }
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3);
            AscendC::DataCopy(
                gmQ[qRopeOffset], qRows,
                AscendC::DataCopyParams(headCount, 4U, 0U, 28U));
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slotFree[1]);
            pending[1] = true;

            constexpr uint32_t kvSlot = 0;
            if (pending[kvSlot]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slotFree[kvSlot]);
                pending[kvSlot] = false;
            }
            const uint64_t kvOffset = static_cast<uint64_t>(token) * kHeadDim;
            AscendC::DataCopy(rowLocal[kvSlot], gmKv[kvOffset], kHeadDim);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(inputReady[kvSlot]);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(inputReady[kvSlot]);
            AscendC::Cast(
                x[kvSlot], rowLocal[kvSlot][kNopeDim],
                AscendC::RoundMode::CAST_NONE, kRopeDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Mul(output[kvSlot], x[kvSlot], cos, kRopeDim);
            AscendC::Gather(rotated[kvSlot], x[kvSlot], offsets, 0U, kRopeDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::FusedMulAdd(
                rotated[kvSlot], sin, output[kvSlot], kRopeDim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Cast(
                rowLocal[kvSlot][kNopeDim], rotated[kvSlot],
                AscendC::RoundMode::CAST_RINT, kRopeDim);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(vMte3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(vMte3);
            if (params.updateKvCache) {
                const int32_t page =
                    gmSlots.GetValue(static_cast<uint64_t>(token) * 2U);
                const int32_t row =
                    gmSlots.GetValue(static_cast<uint64_t>(token) * 2U + 1U);
                const uint64_t cacheRow =
                    static_cast<uint64_t>(page) * kCacheRowsPerPage +
                    static_cast<uint32_t>(row);
                AscendC::DataCopy(
                    gmCache[cacheRow * kHeadDim], rowLocal[kvSlot], kHeadDim);
            } else {
                // Same bytes the fused path scatters into the cache row,
                // written to the fresh [t, kHeadDim] output instead. No slot
                // read and no cache access happen in this branch.
                AscendC::DataCopy(gmKvOut[kvOffset], rowLocal[kvSlot], kHeadDim);
            }
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(slotFree[kvSlot]);
            pending[kvSlot] = true;
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(vMte2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(vMte2);
        }

        for (uint32_t slot = 0; slot < 2; ++slot) {
            if (pending[slot]) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(slotFree[slot]);
            }
            pipe->ReleaseEventID<AscendC::HardEvent::MTE2_V>(inputReady[slot]);
            pipe->ReleaseEventID<AscendC::HardEvent::MTE3_MTE2>(slotFree[slot]);
        }
        pipe->ReleaseEventID<AscendC::HardEvent::V_MTE3>(vMte3);
        pipe->ReleaseEventID<AscendC::HardEvent::V_MTE2>(vMte2);
    }
};

struct UnifiedMlaPrologKernel {
    struct Arguments {
        DualProjectionKernel::Arguments projection;
        NormQuantKernel::Arguments normQuant;
        QuantMatmul::Arguments quantMatmul;
        RopeScatterKernel::Arguments ropeScatter;
    };

    struct Params {
        DualProjectionKernel::Params projection;
        NormQuantKernel::Params normQuant;
        QuantMatmul::Params quantMatmul;
        RopeScatterKernel::Params ropeScatter;
    };

    static bool CanImplement(const Arguments &args)
    {
        return DualProjectionKernel::CanImplement(args.projection) &&
            NormQuantKernel::CanImplement(args.normQuant) &&
            QuantMatmul::CanImplement(args.quantMatmul) &&
            RopeScatterKernel::CanImplement(args.ropeScatter);
    }

    static size_t GetWorkspaceSize(const Arguments &args)
    {
        return QuantMatmul::GetWorkspaceSize(args.quantMatmul);
    }

    CATLASS_HOST_DEVICE
    static Params ToUnderlyingArguments(const Arguments &args, uint8_t *workspace)
    {
        return Params{
            DualProjectionKernel::ToUnderlyingArguments(args.projection, nullptr),
            NormQuantKernel::ToUnderlyingArguments(args.normQuant, nullptr),
            QuantMatmul::ToUnderlyingArguments(args.quantMatmul, workspace),
            RopeScatterKernel::ToUnderlyingArguments(args.ropeScatter, nullptr)};
    }

    template <int32_t CORE_TYPE = g_coreType>
    CATLASS_DEVICE void operator()(const Params &params);

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIC>(const Params &params)
    {
        DualProjectionKernel projection;
        projection.template operator()<AscendC::AIC>(params.projection);
        Arch::CrossCoreSetFlag<0x2, PIPE_FIX>(projectionReady);

        Arch::CrossCoreWaitFlag(normReady);
        quantMatmul.template operator()<AscendC::AIC>(params.quantMatmul);
    }

    template <>
    CATLASS_DEVICE void operator()<AscendC::AIV>(const Params &params)
    {
        NormQuantKernel normQuant;
        normQuant.template operator()<AscendC::AIV>(params.normQuant);
        Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
        Arch::CrossCoreSetFlag<0x2, PIPE_MTE3>(normReady);

        quantMatmul.template operator()<AscendC::AIV>(params.quantMatmul);
        if constexpr (kEnableRopeScatter) {
            Arch::CrossCoreBarrier<0x0, PIPE_MTE3>();
            RopeScatterKernel ropeScatter;
            ropeScatter.template operator()<AscendC::AIV>(params.ropeScatter);
        }
    }

private:
    static constexpr Arch::FlagID kProjectionReadyFlag = 4;
    static constexpr Arch::FlagID kNormReadyFlag = 5;
    Arch::CrossCoreFlag projectionReady{kProjectionReadyFlag};
    Arch::CrossCoreFlag normReady{kNormReadyFlag};
    QuantMatmul quantMatmul;
};

} // namespace Dsv41

#endif // MLA_PROLOG_KERNEL_H
