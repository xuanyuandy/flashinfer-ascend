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
 * \file mla_epilogue_gemm.h
 * \brief mla_epilogue AIC 侧 CATLASS 大 S 模板实现：GEMM1 / GEMM2
 *
 * 约束：
 *  - 仅在 ASCEND_IS_AIC 分支内构造/调用；Arch::Resource<Arch::AtlasA2> 由 Kernel 入口在 AIC 分支提供；
 *  - GEMM1：A/B RowMajor，C layout::zN（Cube NZ），供 GEMM2 直接作为 A 读取；
 *  - GEMM2：A layout::zN，B ColumnMajor（w2 物理 [N,K]），C RowMajor 直接写 y；
 *  - 每核 N 分核与 Host tiling 的 n0/curN 保持一致；M 方向按编译期 L1 tile 循环；
 *  - 本文件不含 SyncAll；AIV→AIC 的唯一屏障在 Kernel 入口。
 */

#ifndef __MLA_EPILOGUE_GEMM_LARGE_H__
#define __MLA_EPILOGUE_GEMM_LARGE_H__

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mla_epilogue_tiling_data.h"

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/arch/resource.hpp"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/dispatch_policy.hpp"
#include "catlass/gemm/gemm_type.hpp"
#include "catlass/gemm/tile/tile_copy.hpp"
#include "catlass/layout/layout.hpp"
#include "tla/layout.hpp"
#include "tla/tensor.hpp"

namespace NsMlaEpilogue {
using namespace AscendC;
using Catlass::GemmCoord;

using CatlassElement = bfloat16_t;
using CatlassArch = Catlass::Arch::AtlasA2;

// ---------------------------------------------------------------------------
// GEMM1: r[g][M,K1] @ w1[g][K1,N1] -> c[g][M,N1], C 为 Cube NZ
// ---------------------------------------------------------------------------
using G1PolicyLarge = Catlass::Gemm::MmadPingpong<CatlassArch, true, false, 1, false, 2, 2, 3, 2>;
using G1L1ShapeLarge = tla::Shape<tla::Int<160>, tla::Int<192>, tla::Int<256>>;
using G1L0ShapeLarge = tla::Shape<tla::Int<160>, tla::Int<192>, tla::Int<64>>;
using G1TileCopyLarge = Catlass::Gemm::Tile::PackedTileCopyTla<
    CatlassArch, CatlassElement, Catlass::layout::RowMajor, CatlassElement, Catlass::layout::RowMajor,
    CatlassElement, Catlass::layout::zN>;
using G1BlockLarge = Catlass::Gemm::Block::BlockMmadTla<
    G1PolicyLarge, G1L1ShapeLarge, G1L0ShapeLarge, CatlassElement, CatlassElement, CatlassElement, void, G1TileCopyLarge>;
// nzConv 路径：w1 预置为 zN（[N1/16][K1][16]），B 的 GM 布局直接是 zN，省掉每 tile 的 ND->zN 重排。
using G1TileCopyNzLarge = Catlass::Gemm::Tile::PackedTileCopyTla<
    CatlassArch, CatlassElement, Catlass::layout::RowMajor, CatlassElement, Catlass::layout::zN,
    CatlassElement, Catlass::layout::zN>;
using G1BlockNzLarge = Catlass::Gemm::Block::BlockMmadTla<
    G1PolicyLarge, G1L1ShapeLarge, G1L0ShapeLarge, CatlassElement, CatlassElement, CatlassElement, void,
    G1TileCopyNzLarge>;

// ---------------------------------------------------------------------------
// GEMM2: c[M,K2] @ w2[N2,K2]^T -> y[M,N2], A 为 Cube NZ，B 用 ColumnMajor 表达转置
// ---------------------------------------------------------------------------
using G2PolicyLarge = Catlass::Gemm::MmadPingpong<CatlassArch, true, false, 1, false, 2, 2, 3, 2>;
using G2L1ShapeLarge = tla::Shape<tla::Int<128>, tla::Int<224>, tla::Int<320>>;
using G2L0ShapeLarge = tla::Shape<tla::Int<128>, tla::Int<224>, tla::Int<64>>;
using G2TileCopyLarge = Catlass::Gemm::Tile::PackedTileCopyTla<
    CatlassArch, CatlassElement, Catlass::layout::zN, CatlassElement, Catlass::layout::ColumnMajor,
    CatlassElement, Catlass::layout::RowMajor>;
using G2BlockLarge = Catlass::Gemm::Block::BlockMmadTla<
    G2PolicyLarge, G2L1ShapeLarge, G2L0ShapeLarge, CatlassElement, CatlassElement, CatlassElement, void, G2TileCopyLarge>;
// nzConv 路径：w2 预置为 nZ（[K2/16][N2][16]），B 的 GM 布局直接是 nZ。
using G2TileCopyNzLarge = Catlass::Gemm::Tile::PackedTileCopyTla<
    CatlassArch, CatlassElement, Catlass::layout::zN, CatlassElement, Catlass::layout::nZ,
    CatlassElement, Catlass::layout::RowMajor>;
using G2BlockNzLarge = Catlass::Gemm::Block::BlockMmadTla<
    G2PolicyLarge, G2L1ShapeLarge, G2L0ShapeLarge, CatlassElement, CatlassElement, CatlassElement, void,
    G2TileCopyNzLarge>;

/**
 * @brief AIC 侧 CATLASS 计算上下文
 *
 * 由 Kernel 入口在 ASCEND_IS_AIC 分支内构造；resource 的生命周期覆盖两次 GEMM。
 * BlockMmadTla 使用局部作用域依次构造/析构，确保 L0/L0C 与 event ID 不会同时占用。
 */
class AicContextLarge {
public:
    __aicore__ inline explicit AicContextLarge(Catlass::Arch::Resource<CatlassArch>& resource) : resource_(resource) {}

    __aicore__ inline void Init(const MlaEpilogueTilingData& td, GM_ADDR userWs, GM_ADDR w1, GM_ADDR w2, GM_ADDR y)
    {
        rGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWs + td.rOffset));
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWs + td.cOffset));
        w1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w1));
        w2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w2));
        w1NzGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWs + td.w1NzOffset));
        w2NzGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWs + td.w2NzOffset));
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(y));
    }

    /**
     * @brief GEMM1：每核固定 (g, n0, curN)，循环 M tile 与 N tile。
     *        C 为 layout::zN，物理布局 [N/16][M/16][16][16]。
     */
    template <bool Nz>
    __aicore__ inline void Gemm1(const MlaEpilogueTilingData& td)
    {
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        if (coreIdx >= td.gemm1UsedCores) {
            return;
        }
        const uint32_t nSplit = td.gemm1CoresPerBatch;
        const uint32_t g = coreIdx / nSplit;
        const uint32_t nIdx = coreIdx - g * nSplit;
        const bool isLast = ((nIdx + 1U) >= nSplit);
        const uint32_t nUnits = isLast ? td.gemm1NUnitsLast : td.gemm1NUnitsBase;
        const uint32_t n0 = MLA_ALIGN16 * (isLast ? ((nSplit - 1U) * td.gemm1NUnitsBase) :
                                                     (nIdx * td.gemm1NUnitsBase));
        const uint32_t curN = MLA_ALIGN16 * nUnits;

        auto gA = rGm_[static_cast<uint64_t>(g) * td.paddedM * MLA_K1];
        auto gC = cGm_[static_cast<uint64_t>(g) * MLA_N1 * td.paddedM];
        auto layoutA = tla::MakeLayout<CatlassElement, Catlass::layout::RowMajor>(td.paddedM, MLA_K1);
        auto layoutC = tla::MakeLayout<CatlassElement, Catlass::layout::zN>(td.paddedM, MLA_N1);
        auto tensorA = tla::MakeTensor(gA, layoutA, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gC, layoutC, Catlass::Arch::PositionGM{});
        constexpr uint32_t kM = tla::get<0>(G1L1ShapeLarge{});
        constexpr uint32_t kN = tla::get<1>(G1L1ShapeLarge{});

        if constexpr (Nz) {
            // B 已由 AIV 预转换为 zN，直接按 zN 布局读取。
            auto gB = w1NzGm_[static_cast<uint64_t>(g) * MLA_K1 * MLA_N1];
            auto layoutB = tla::MakeLayout<CatlassElement, Catlass::layout::zN>(MLA_K1, MLA_N1);
            auto tensorB = tla::MakeTensor(gB, layoutB, Catlass::Arch::PositionGM{});
            G1BlockNzLarge blockMmad(resource_);
            for (uint32_t m0 = 0U; m0 < td.totalS; m0 += kM) {
                const uint32_t mAct = min(kM, td.totalS - m0);
                auto tileA = tla::GetTile(tensorA, tla::MakeCoord(m0, 0U), tla::MakeShape(mAct, MLA_K1));
                for (uint32_t nOff = 0U; nOff < curN; nOff += kN) {
                    const uint32_t nAct = min(kN, curN - nOff);
                    auto tileB = tla::GetTile(tensorB, tla::MakeCoord(0U, n0 + nOff),
                                              tla::MakeShape(MLA_K1, nAct));
                    auto tileC = tla::GetTile(tensorC, tla::MakeCoord(m0, n0 + nOff),
                                              tla::MakeShape(mAct, nAct));
                    blockMmad(tileA, tileB, tileC, GemmCoord{mAct, nAct, MLA_K1});
                }
            }
        } else {
            auto gB = w1Gm_[static_cast<uint64_t>(g) * MLA_K1 * MLA_N1];
            auto layoutB = tla::MakeLayout<CatlassElement, Catlass::layout::RowMajor>(MLA_K1, MLA_N1);
            auto tensorB = tla::MakeTensor(gB, layoutB, Catlass::Arch::PositionGM{});
            G1BlockLarge blockMmad(resource_);
            for (uint32_t m0 = 0U; m0 < td.totalS; m0 += kM) {
                const uint32_t mAct = min(kM, td.totalS - m0);
                auto tileA = tla::GetTile(tensorA, tla::MakeCoord(m0, 0U), tla::MakeShape(mAct, MLA_K1));
                for (uint32_t nOff = 0U; nOff < curN; nOff += kN) {
                    const uint32_t nAct = min(kN, curN - nOff);
                    auto tileB = tla::GetTile(tensorB, tla::MakeCoord(0U, n0 + nOff),
                                              tla::MakeShape(MLA_K1, nAct));
                    auto tileC = tla::GetTile(tensorC, tla::MakeCoord(m0, n0 + nOff),
                                              tla::MakeShape(mAct, nAct));
                    blockMmad(tileA, tileB, tileC, GemmCoord{mAct, nAct, MLA_K1});
                }
            }
        }
    }

    /**
     * @brief GEMM2：每核固定 n0/curN，M 方向按 96 行循环；A 直接读 GEMM1 的 zN 输出。
     */
    template <bool Nz, bool Stage1Partition>
    __aicore__ inline void Gemm2(const MlaEpilogueTilingData& td)
    {
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        if (coreIdx >= td.gemm2UsedCores) {
            return;
        }
        uint32_t n0;
        uint32_t curN;
        if (Stage1Partition && (td.aicUsedCores == 24U) && (td.gemm2UsedCores == 24U)) {
            // Stage001 only on ND paths; the full-NZ path keeps its original partition.
            // 22*224 + 2*96 = 5120; only the two N96 cores use the K/16 fallback.
            constexpr uint32_t FAST_CORES = 22U;
            constexpr uint32_t FAST_N = 224U;
            constexpr uint32_t TAIL_N = 96U;
            static_assert(FAST_CORES * FAST_N + 2U * TAIL_N == MLA_N2, "G2 N partition must cover N2");
            if (coreIdx < FAST_CORES) {
                n0 = coreIdx * FAST_N;
                curN = FAST_N;
            } else {
                n0 = FAST_CORES * FAST_N + (coreIdx - FAST_CORES) * TAIL_N;
                curN = TAIL_N;
            }
        } else {
            const uint32_t hasRem = (coreIdx < td.gemm2NUnitsRem) ? 1U : 0U;
            const uint32_t nUnits = td.gemm2NUnitsBase + hasRem;
            n0 = MLA_ALIGN16 * (coreIdx * td.gemm2NUnitsBase +
                                ((coreIdx < td.gemm2NUnitsRem) ? coreIdx : td.gemm2NUnitsRem));
            curN = MLA_ALIGN16 * nUnits;
        }

        auto gA = cGm_[0];
        auto gC = yGm_[0];
        auto layoutA = tla::MakeLayout<CatlassElement, Catlass::layout::zN>(td.paddedM, MLA_K2);
        auto layoutC = tla::MakeLayout<CatlassElement, Catlass::layout::RowMajor>(td.totalS, MLA_N2);
        auto tensorA = tla::MakeTensor(gA, layoutA, Catlass::Arch::PositionGM{});
        auto tensorC = tla::MakeTensor(gC, layoutC, Catlass::Arch::PositionGM{});
        constexpr uint32_t kM = tla::get<0>(G2L1ShapeLarge{});

        if constexpr (Nz) {
            // B 已由 AIV 预转换为 nZ，直接按 nZ 布局读取。
            auto gB = w2NzGm_[0];
            auto layoutB = tla::MakeLayout<CatlassElement, Catlass::layout::nZ>(MLA_K2, MLA_N2);
            auto tensorB = tla::MakeTensor(gB, layoutB, Catlass::Arch::PositionGM{});
            G2BlockNzLarge blockMmad(resource_);
            for (uint32_t m0 = 0U; m0 < td.totalS; m0 += kM) {
                const uint32_t mAct = min(kM, td.totalS - m0);
                auto tileA = tla::GetTile(tensorA, tla::MakeCoord(m0, 0U), tla::MakeShape(mAct, MLA_K2));
                auto tileB = tla::GetTile(tensorB, tla::MakeCoord(0U, n0), tla::MakeShape(MLA_K2, curN));
                auto tileC = tla::GetTile(tensorC, tla::MakeCoord(m0, n0), tla::MakeShape(mAct, curN));
                blockMmad(tileA, tileB, tileC, GemmCoord{mAct, curN, MLA_K2});
            }
        } else {
            auto gB = w2Gm_[0];
            auto layoutB = tla::MakeLayout<CatlassElement, Catlass::layout::ColumnMajor>(MLA_K2, MLA_N2);
            auto tensorB = tla::MakeTensor(gB, layoutB, Catlass::Arch::PositionGM{});
            G2BlockLarge blockMmad(resource_);
            for (uint32_t m0 = 0U; m0 < td.totalS; m0 += kM) {
                const uint32_t mAct = min(kM, td.totalS - m0);
                auto tileA = tla::GetTile(tensorA, tla::MakeCoord(m0, 0U), tla::MakeShape(mAct, MLA_K2));
                auto tileB = tla::GetTile(tensorB, tla::MakeCoord(0U, n0), tla::MakeShape(MLA_K2, curN));
                auto tileC = tla::GetTile(tensorC, tla::MakeCoord(m0, n0), tla::MakeShape(mAct, curN));
                blockMmad(tileA, tileB, tileC, GemmCoord{mAct, curN, MLA_K2});
            }
        }
    }

private:
    Catlass::Arch::Resource<CatlassArch>& resource_;

    GlobalTensor<bfloat16_t> rGm_;  // [8, paddedM, 4096]
    GlobalTensor<bfloat16_t> cGm_;  // zN: [8192/16][paddedM/16][16][16]
    GlobalTensor<bfloat16_t> w1Gm_;   // [8, 4096, 1024]
    GlobalTensor<bfloat16_t> w2Gm_;   // [5120, 8192]
    GlobalTensor<bfloat16_t> w1NzGm_; // zN: [N1/16][K1][16]（nzConv 时有效）
    GlobalTensor<bfloat16_t> w2NzGm_; // nZ: [K2/16][N2][16]（nzConv 时有效）
    GlobalTensor<bfloat16_t> yGm_;  // [S, 5120]
};

} // namespace NsMlaEpilogue

#endif // __MLA_EPILOGUE_GEMM_LARGE_H__
