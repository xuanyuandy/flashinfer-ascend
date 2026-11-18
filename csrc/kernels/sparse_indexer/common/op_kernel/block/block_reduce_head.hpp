/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_REDUCE_HEAD_HPP
#define FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_REDUCE_HEAD_HPP

#include <cstdint>
#include <type_traits>

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "kernel_operator.h"

#include "mask_policy.hpp"
#include "mask_utils.hpp"
#include "sparse_indexer_common.hpp"

namespace SparseIndexer::Common::Block {

/// Copy a UB tile to GM. A different destination type supplies its own store.
template <class DtypeDst_, class DtypeScore_>
struct DirectWorkspaceStore {
    static_assert(std::is_same_v<DtypeDst_, DtypeScore_>, "DirectWorkspaceStore requires matching element types");

    using DtypeDst = DtypeDst_;
    using DtypeScore = DtypeScore_;

    /// Write rows of a packed UB tile. srcRowStride / dstRowStride are element counts.
    CATLASS_DEVICE static void StoreTile(
        AscendC::GlobalTensor<DtypeDst> gmDst, AscendC::LocalTensor<DtypeScore>& ubTile, uint32_t rows, uint32_t cols,
        uint32_t srcRowStride, uint32_t dstRowStride)
    {
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(EVENT_ID0);
        uint32_t copyBytes = cols * sizeof(DtypeDst);
        uint32_t srcRowBytes = srcRowStride * sizeof(DtypeDst);
        uint32_t copyAligned = (copyBytes + Catlass::BYTE_PER_BLK - 1) / Catlass::BYTE_PER_BLK * Catlass::BYTE_PER_BLK;
        uint32_t srcStrideBlocks = (srcRowBytes - copyAligned) / Catlass::BYTE_PER_BLK;
        uint32_t dstStrideBytes = (dstRowStride - cols) * sizeof(DtypeDst);
        AscendC::DataCopyExtParams params(rows, copyBytes, srcStrideBlocks, dstStrideBytes, 0);
        AscendC::DataCopyPad(gmDst, ubTile, params);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
    }
};

/// Sum_h W[s1, h] * ReLU(Q @ K^T)[s1, h, :] over every head, then write this Vector's half of S2.
template <
    class DtypeScore_, class DtypeW_, class DtypeDst_ = DtypeScore_,
    class MaskPolicy_ = LiV2MaskPolicy<DtypeScore_>,
    class Store_ = DirectWorkspaceStore<DtypeDst_, DtypeScore_>>
class BlockReduceHead {
public:
    static constexpr uint32_t kMaxBatch = SparseIndexer::Common::kMaxBatch;

    struct Params {
        uint32_t n1{0};
        uint32_t s2{0};
        uint32_t s1Tile{0};
        uint32_t s2Tile{0};
        int32_t maskMode{0};
        int32_t cmpRatio{1};
    };

    using DtypeScore = DtypeScore_;
    using DtypeW = DtypeW_;
    using DtypeDst = DtypeDst_;
    using Store = Store_;
    using MaskPolicy = MaskPolicy_;

    using ArchTag = Catlass::Arch::Ascend950;

    static constexpr uint32_t kFloatBlock = Catlass::BYTE_PER_BLK / sizeof(DtypeScore);
    static constexpr uint32_t kFloatVec = 256 / sizeof(DtypeScore);
    /// AscendC repeat and repStride fields are uint8_t.
    static constexpr uint32_t kMaxRepeat = 255;

    CATLASS_DEVICE
    BlockReduceHead(
        AscendC::LocalTensor<DtypeScore>& ubScore, Catlass::Arch::Resource<ArchTag>& resource,
        AscendC::GlobalTensor<DtypeW> const& gmW, AscendC::GlobalTensor<DtypeDst> const& gmDst,
        Params const& params)
        : ubScore_(ubScore),
          gmW_(gmW),
          gmDst_(gmDst),
          params_(params),
          n1_(params.n1),
          s2_(params.s2),
          s1Tile_(params.s1Tile),
          s2Tile_(params.s2Tile),
          maskPolicy_(ubOut_, ubWBrcb_, ubTmp_)
    {
        constexpr uint32_t kWAlign = Catlass::BYTE_PER_BLK / sizeof(DtypeW);
        wStride_ = (n1_ + kWAlign - 1) / kWAlign * kWAlign;
        uint32_t aivPerAic = AscendC::GetSubBlockNum();
        constexpr uint32_t l0cFractalN = 16;
        uint32_t splitNAlign = l0cFractalN * aivPerAic;
        uint32_t colPerAiv = SparseIndexer::Common::AlignUp(s2Tile_, splitNAlign) / aivPerAic;
        uint32_t offset = s1Tile_ * n1_ * colPerAiv * sizeof(DtypeScore);
        ubOut_ = resource.ubBuf.template GetBufferByByte<DtypeScore>(offset);
        offset += s1Tile_ * colPerAiv * sizeof(DtypeScore);
        ubW_ = resource.ubBuf.template GetBufferByByte<DtypeScore>(offset);
        offset += s1Tile_ * wStride_ * sizeof(DtypeScore);
        ubWIn_ = resource.ubBuf.template GetBufferByByte<DtypeW>(offset);
        offset += s1Tile_ * wStride_ * sizeof(DtypeW);
        ubWCol_ = resource.ubBuf.template GetBufferByByte<DtypeScore>(offset);
        offset += s1Tile_ * sizeof(DtypeScore);
        ubWBrcb_ = resource.ubBuf.template GetBufferByByte<DtypeScore>(offset);
        offset += s1Tile_ * kFloatBlock * sizeof(DtypeScore);
        ubTmp_ = resource.ubBuf.template GetBufferByByte<DtypeScore>(offset);
    }

    /// One repeat is one token. src1 is that token's weight, broadcast to a 32B block.
    CATLASS_DEVICE
    void ScaleHead(uint32_t head, uint32_t actualS1, uint32_t colPerAiv, uint32_t cols, bool first)
    {
        uint32_t dstRep = colPerAiv / kFloatBlock;
        uint32_t srcRep = n1_ * colPerAiv / kFloatBlock;
        if (dstRep > kMaxRepeat || srcRep > kMaxRepeat) {
            for (uint32_t token = 0; token < actualS1; ++token) {
                auto score = ubScore_[(token * n1_ + head) * colPerAiv];
                auto out = ubOut_[token * colPerAiv];
                DtypeScore weight = ubW_.GetValue(token * wStride_ + head);
                if (first) {
                    AscendC::Muls(out, score, weight, cols);
                } else {
                    AscendC::Axpy(out, score, weight, cols);
                }
            }
            return;
        }

        uint32_t packed = (actualS1 + kFloatBlock - 1) / kFloatBlock * kFloatBlock;
        for (uint32_t token = 0; token < actualS1; ++token) {
            ubWCol_.SetValue(token, ubW_.GetValue(token * wStride_ + head));
        }
        for (uint32_t token = actualS1; token < packed; ++token) {
            ubWCol_.SetValue(token, 0);
        }
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Brcb(
            ubWBrcb_.template ReinterpretCast<uint32_t>(), ubWCol_.template ReinterpretCast<uint32_t>(),
            packed / kFloatBlock, AscendC::BrcbRepeatParams(1, 8));
        AscendC::PipeBarrier<PIPE_V>();

        AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
        for (uint32_t rowBase = 0; rowBase < actualS1; rowBase += kMaxRepeat) {
            uint32_t rows = AscendC::Std::min(actualS1 - rowBase, kMaxRepeat);
            auto out = ubOut_[rowBase * colPerAiv];
            auto score = ubScore_[(rowBase * n1_ + head) * colPerAiv];
            auto scale = ubWBrcb_[rowBase * kFloatBlock];
            uint32_t full = cols / kFloatVec;
            uint32_t remain = cols % kFloatVec;
            for (uint32_t chunk = 0; chunk < full; ++chunk) {
                uint32_t off = chunk * kFloatVec;
                if (first) {
                    AscendC::Mul<DtypeScore, false>(
                        out[off], score[off], scale, (uint64_t)0, rows,
                        AscendC::BinaryRepeatParams(1, 1, 0, dstRep, srcRep, 1));
                } else {
                    AscendC::Mul<DtypeScore, false>(
                        ubTmp_, score[off], scale, (uint64_t)0, rows,
                        AscendC::BinaryRepeatParams(1, 1, 0, kFloatVec / kFloatBlock, srcRep, 1));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<DtypeScore, false>(
                        out[off], out[off], ubTmp_, (uint64_t)0, rows,
                        AscendC::BinaryRepeatParams(1, 1, 1, dstRep, dstRep, kFloatVec / kFloatBlock));
                }
            }
            if (remain != 0) {
                uint64_t mask = 0;
                for (uint32_t bit = 0; bit < remain; ++bit) {
                    mask |= (uint64_t)1 << bit;
                }
                AscendC::SetVectorMask<int8_t>(0x0, mask);
                uint32_t off = full * kFloatVec;
                if (first) {
                    AscendC::Mul<DtypeScore, false>(
                        out[off], score[off], scale, (uint64_t)0, rows,
                        AscendC::BinaryRepeatParams(1, 1, 0, dstRep, srcRep, 1));
                } else {
                    AscendC::Mul<DtypeScore, false>(
                        ubTmp_, score[off], scale, (uint64_t)0, rows,
                        AscendC::BinaryRepeatParams(1, 1, 0, kFloatVec / kFloatBlock, srcRep, 1));
                    AscendC::PipeBarrier<PIPE_V>();
                    AscendC::Add<DtypeScore, false>(
                        out[off], out[off], ubTmp_, (uint64_t)0, rows,
                        AscendC::BinaryRepeatParams(1, 1, 1, dstRep, dstRep, kFloatVec / kFloatBlock));
                }
                AscendC::SetVectorMask<int8_t>((uint64_t)-1, (uint64_t)-1);
            }
        }
    }

    CATLASS_DEVICE
    void operator()(
        uint32_t s1Start, uint32_t actualS1, uint32_t s2Start, uint32_t actualS2, uint32_t localQStart,
        uint32_t qAct, uint32_t kAct, uint32_t reqKStart, int32_t residual, bool maskActive)
    {
        uint32_t aivPerAic = AscendC::GetSubBlockNum();
        constexpr uint32_t l0cFractalN = 16;
        uint32_t splitNAlign = l0cFractalN * aivPerAic;
        uint32_t alignedN = SparseIndexer::Common::AlignUp(actualS2, splitNAlign);
        uint32_t colPerAiv = alignedN / aivPerAic;
        uint32_t colBegin = AscendC::GetSubBlockIdx() * colPerAiv;
        if (colBegin >= actualS2) {
            return;
        }
        uint32_t cols = AscendC::Std::min(colPerAiv, actualS2 - colBegin);

        // D1: UB DataCopy / Cast / Relu need 32B (often 256B) aligned element counts.
        // Pad S1 to FP32 block (8); only actualS1 rows are scaled / stored.
        uint32_t alignedS1 = SparseIndexer::Common::AlignUp(actualS1, kFloatBlock);
        if (alignedS1 > s1Tile_) {
            alignedS1 = s1Tile_;
        }

        AscendC::Relu(ubScore_, ubScore_, alignedS1 * n1_ * colPerAiv);
        AscendC::PipeBarrier<PIPE_V>();

        uint32_t rightPad = wStride_ - n1_;
        AscendC::DataCopyExtParams loadParams(actualS1, n1_ * sizeof(DtypeW), 0, 0, 0);
        AscendC::DataCopyPadExtParams<DtypeW> padParams(rightPad != 0, 0, rightPad, static_cast<DtypeW>(0));
        AscendC::DataCopyPad(ubWIn_, gmW_[s1Start * n1_], loadParams, padParams);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
        if (alignedS1 > actualS1) {
            AscendC::Duplicate(
                ubWIn_[actualS1 * wStride_], static_cast<DtypeW>(0),
                (alignedS1 - actualS1) * wStride_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        if constexpr (std::is_same_v<DtypeW, DtypeScore>) {
            AscendC::DataCopy(ubW_, ubWIn_, alignedS1 * wStride_);
        } else {
            AscendC::Cast(ubW_, ubWIn_, AscendC::RoundMode::CAST_NONE, alignedS1 * wStride_);
        }
        AscendC::PipeBarrier<PIPE_V>();

        for (uint32_t h = 0; h < n1_; ++h) {
            ScaleHead(h, actualS1, colPerAiv, cols, h == 0);
        }
        maskPolicy_.Apply(
            actualS1, cols, colPerAiv, colBegin, s2Start, localQStart, qAct, kAct, reqKStart, residual,
            maskActive, params_.maskMode, params_.cmpRatio);
        Store::StoreTile(gmDst_[s1Start * s2_ + s2Start + colBegin], ubOut_, actualS1, cols, colPerAiv, s2_);
    }

private:
    AscendC::LocalTensor<DtypeScore>& ubScore_;
    AscendC::LocalTensor<DtypeScore> ubOut_;
    AscendC::LocalTensor<DtypeScore> ubW_;
    AscendC::LocalTensor<DtypeW> ubWIn_;
    AscendC::LocalTensor<DtypeScore> ubWCol_;
    AscendC::LocalTensor<DtypeScore> ubWBrcb_;
    AscendC::LocalTensor<DtypeScore> ubTmp_;
    AscendC::GlobalTensor<DtypeW> const& gmW_;
    AscendC::GlobalTensor<DtypeDst> const& gmDst_;
    Params params_{};
    uint32_t n1_{0};
    uint32_t s2_{0};
    uint32_t s1Tile_{0};
    uint32_t s2Tile_{0};
    uint32_t wStride_{0};
    MaskPolicy maskPolicy_;
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_REDUCE_HEAD_HPP
