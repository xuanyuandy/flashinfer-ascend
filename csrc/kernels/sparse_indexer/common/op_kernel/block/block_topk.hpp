/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 */

#ifndef FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_TOPK_HPP
#define FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_TOPK_HPP

#include "catlass/arch/resource.hpp"
#include "catlass/catlass.hpp"
#include "catlass/detail/alignment.hpp"
#include "kernel_operator.h"

#include "mask_utils.hpp"
#include "sparse_indexer_common.hpp"

namespace SparseIndexer::Common::Block {

namespace detail {

CATLASS_DEVICE uint32_t FloatToSortableU32(float v)
{
    union {
        float f;
        uint32_t u;
    } bits;
    bits.f = v;
    uint32_t u = bits.u;
    if (u & 0x80000000U) {
        u = ~u;
    } else {
        u ^= 0x80000000U;
    }
    return u;
}

CATLASS_DEVICE float SortableU32ToFloat(uint32_t u)
{
    if (u & 0x80000000U) {
        u ^= 0x80000000U;
    } else {
        u = ~u;
    }
    union {
        float f;
        uint32_t u;
    } bits;
    bits.u = u;
    return bits.f;
}

CATLASS_DEVICE void FloatToSortableU32Row(
    AscendC::LocalTensor<float>& scoreF, AscendC::LocalTensor<uint32_t>& scoreU32, uint32_t validLen)
{
    for (uint32_t j = 0; j < validLen; ++j) {
        scoreU32.SetValue(j, FloatToSortableU32(scoreF.GetValue(j)));
    }
    AscendC::PipeBarrier<PIPE_V>();
}

CATLASS_DEVICE uint32_t SortableByte(uint32_t key, uint32_t pass)
{
    if (pass == 0) {
        return (key >> 24U) & 0xffU;
    }
    if (pass == 1) {
        return (key >> 16U) & 0xffU;
    }
    if (pass == 2) {
        return (key >> 8U) & 0xffU;
    }
    return key & 0xffU;
}

CATLASS_DEVICE bool KeyMatchesPrefix(uint32_t key, uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3)
{
    return SortableByte(key, 0) == b0 && SortableByte(key, 1) == b1 && SortableByte(key, 2) == b2 &&
           SortableByte(key, 3) == b3;
}

/// LIV2-style multi-pass 8-bit histogram (scalar); returns k-th largest sortable threshold.
CATLASS_DEVICE uint32_t RadixThresholdK(
    AscendC::LocalTensor<uint32_t>& keys, uint32_t validLen, uint32_t topK, AscendC::LocalTensor<uint32_t>& hist)
{
    uint32_t bottomK = validLen - topK + 1U;
    uint32_t b0 = 0;
    uint32_t b1 = 0;
    uint32_t b2 = 0;
    uint32_t b3 = 0;

    for (uint32_t pass = 0; pass < 4U; ++pass) {
        for (uint32_t i = 0; i < 256U; ++i) {
            hist.SetValue(i, 0);
        }
        for (uint32_t i = 0; i < validLen; ++i) {
            const uint32_t key = keys.GetValue(i);
            if (pass >= 1U && SortableByte(key, 0) != b0) {
                continue;
            }
            if (pass >= 2U && SortableByte(key, 1) != b1) {
                continue;
            }
            if (pass >= 3U && SortableByte(key, 2) != b2) {
                continue;
            }
            const uint32_t bin = SortableByte(key, pass);
            hist.SetValue(bin, hist.GetValue(bin) + 1U);
        }

        uint32_t cum = 0;
        uint32_t target = 0;
        for (uint32_t bin = 0; bin < 256U; ++bin) {
            cum += hist.GetValue(bin);
            if (cum >= bottomK) {
                target = bin;
                break;
            }
        }
        if (pass == 0U) {
            b0 = target;
        } else if (pass == 1U) {
            b1 = target;
        } else if (pass == 2U) {
            b2 = target;
        } else {
            b3 = target;
        }
        bottomK = bottomK - (cum - hist.GetValue(target));
    }

    uint32_t nk = 0xffffffffU;
    for (uint32_t i = 0; i < validLen; ++i) {
        const uint32_t key = keys.GetValue(i);
        if (KeyMatchesPrefix(key, b0, b1, b2, b3)) {
            nk = AscendC::Std::min(nk, key);
        }
    }
    return nk;
}

/// LI V2 order: higher sortable key first; on ties prefer smaller global index.
CATLASS_DEVICE bool IsBetterTopKCandidate(uint32_t keyA, uint32_t idxA, uint32_t keyB, uint32_t idxB)
{
    return keyA > keyB || (keyA == keyB && idxA < idxB);
}

CATLASS_DEVICE void MergeTopKLists(AscendC::LocalTensor<uint32_t>& stateKey,
    AscendC::LocalTensor<uint32_t>& stateIdx, uint32_t& stateCount, AscendC::LocalTensor<uint32_t>& localKey,
    AscendC::LocalTensor<uint32_t>& localIdx, uint32_t localCount, uint32_t topK)
{
    if (localCount == 0) {
        return;
    }

    uint32_t merged = stateCount + localCount;
    if (merged > topK * 2U) {
        merged = topK * 2U;
    }

    AscendC::LocalTensor<uint32_t> tmpKey = stateKey[topK];
    AscendC::LocalTensor<uint32_t> tmpIdx = stateIdx[topK];

    uint32_t n = 0;
    for (uint32_t i = 0; i < stateCount && n < merged; ++i) {
        tmpKey.SetValue(n, stateKey.GetValue(i));
        tmpIdx.SetValue(n, stateIdx.GetValue(i));
        ++n;
    }
    for (uint32_t i = 0; i < localCount && n < merged; ++i) {
        tmpKey.SetValue(n, localKey.GetValue(i));
        tmpIdx.SetValue(n, localIdx.GetValue(i));
        ++n;
    }

    // Always sort (including stateCount==0) so single-tile rows are LI-stable ordered.
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t best = i;
        for (uint32_t j = i + 1; j < n; ++j) {
            if (IsBetterTopKCandidate(
                    tmpKey.GetValue(j), tmpIdx.GetValue(j), tmpKey.GetValue(best), tmpIdx.GetValue(best))) {
                best = j;
            }
        }
        if (best != i) {
            const uint32_t kSwap = tmpKey.GetValue(i);
            const uint32_t iSwap = tmpIdx.GetValue(i);
            tmpKey.SetValue(i, tmpKey.GetValue(best));
            tmpIdx.SetValue(i, tmpIdx.GetValue(best));
            tmpKey.SetValue(best, kSwap);
            tmpIdx.SetValue(best, iSwap);
        }
    }

    stateCount = AscendC::Std::min(topK, n);
    for (uint32_t i = 0; i < stateCount; ++i) {
        stateKey.SetValue(i, tmpKey.GetValue(i));
        stateIdx.SetValue(i, tmpIdx.GetValue(i));
    }
}

CATLASS_DEVICE void SegmentTopKOnRow(AscendC::LocalTensor<float>& scoreUb,
    AscendC::LocalTensor<uint32_t>& scoreU32, AscendC::LocalTensor<uint32_t>& localKey,
    AscendC::LocalTensor<uint32_t>& localIdx, AscendC::LocalTensor<uint32_t>& histUb, uint32_t validLen,
    uint32_t topK, uint32_t idxOffset, uint32_t& localCount)
{
    localCount = 0;
    if (validLen == 0) {
        return;
    }

    // Compact away -inf so masked columns are never selected.
    uint32_t finiteLen = 0;
    for (uint32_t i = 0; i < validLen; ++i) {
        const float v = scoreUb.GetValue(i);
        if (IsNegInfFloat(v)) {
            continue;
        }
        scoreUb.SetValue(finiteLen, v);
        histUb.SetValue(finiteLen, idxOffset + i); // temporary global-id map (hist reused after)
        ++finiteLen;
    }
    if (finiteLen == 0) {
        return;
    }
    FloatToSortableU32Row(scoreUb, scoreU32, finiteLen);

    if (finiteLen <= topK) {
        for (uint32_t i = 0; i < finiteLen; ++i) {
            localKey.SetValue(i, scoreU32.GetValue(i));
            localIdx.SetValue(i, histUb.GetValue(i));
        }
        localCount = finiteLen;
        return;
    }

    // Save global-id map into localIdx before hist is reused by RadixThresholdK.
    for (uint32_t i = 0; i < finiteLen; ++i) {
        localIdx.SetValue(i, histUb.GetValue(i));
    }
    const uint32_t nk = RadixThresholdK(scoreU32, finiteLen, topK, histUb);

    uint32_t count = 0;
    for (uint32_t i = 0; i < finiteLen && count < topK; ++i) {
        if (scoreU32.GetValue(i) > nk) {
            localKey.SetValue(count, scoreU32.GetValue(i));
            // localIdx[i] still holds global id for compacted slot i until we compact outs.
            histUb.SetValue(count, localIdx.GetValue(i));
            ++count;
        }
    }
    for (uint32_t i = 0; i < finiteLen && count < topK; ++i) {
        if (scoreU32.GetValue(i) == nk) {
            localKey.SetValue(count, scoreU32.GetValue(i));
            histUb.SetValue(count, localIdx.GetValue(i));
            ++count;
        }
    }
    localCount = count;
    for (uint32_t i = 0; i < localCount; ++i) {
        localIdx.SetValue(i, histUb.GetValue(i));
    }
}

} // namespace detail

/// TopK along S2 after full score workspace is ready (SyncAll). Scans S2 in s2Tile blocks.
template <class DtypeScore_, class DtypeOut_>
class BlockTopK {
public:
    struct Params {
        uint32_t topK{0};
        uint32_t s2Stride{0};
        uint32_t s2Tile{0};
        bool hasOutputIdxOffset{false};
    };

    using DtypeScore = DtypeScore_;
    using DtypeOut = DtypeOut_;
    using DtypeS = DtypeScore_;
    using ArchTag = Catlass::Arch::Ascend950;

    static constexpr uint32_t kSplitS1 = 2;
    static constexpr uint32_t kFloatBlock = Catlass::BYTE_PER_BLK / sizeof(DtypeScore);

    CATLASS_DEVICE BlockTopK(Catlass::Arch::Resource<ArchTag>& resource,
        AscendC::GlobalTensor<DtypeScore> const& workspace, AscendC::GlobalTensor<DtypeOut> const& outIndices,
        Params const& params, AscendC::GlobalTensor<int32_t> const& outputIdxOffset,
        AscendC::GlobalTensor<DtypeScore> const& outValues)
        : gmScores_(workspace),
          gmOut_(outIndices),
          gmValues_(outValues),
          gmOutputIdxOffset_(outputIdxOffset),
          topK_(params.topK),
          s2Stride_(params.s2Stride),
          s2Tile_(params.s2Tile),
          hasOutputIdxOffset_(params.hasOutputIdxOffset),
          tileColAlign_(SparseIndexer::Common::AlignUp(params.s2Tile, kFloatBlock))
    {
        const uint32_t kTopAlign = SparseIndexer::Common::AlignUp(params.topK, 256u);
        const uint32_t kMergeAlign = SparseIndexer::Common::AlignUp(params.topK * 2U, 256u);
        uint32_t offset = 0;
        scoreUb_ = resource.ubBuf.template GetBufferByByte<DtypeScore>(offset);
        offset += tileColAlign_ * sizeof(DtypeScore);
        scoreU32_ = scoreUb_.template ReinterpretCast<uint32_t>();
        stateKeyUb_ = resource.ubBuf.template GetBufferByByte<uint32_t>(offset);
        offset += kMergeAlign * sizeof(uint32_t);
        stateIdxUb_ = resource.ubBuf.template GetBufferByByte<uint32_t>(offset);
        offset += kMergeAlign * sizeof(uint32_t);
        localKeyUb_ = resource.ubBuf.template GetBufferByByte<uint32_t>(offset);
        offset += kTopAlign * sizeof(uint32_t);
        localIdxUb_ = resource.ubBuf.template GetBufferByByte<uint32_t>(offset);
        offset += kTopAlign * sizeof(uint32_t);
        histUb_ = resource.ubBuf.template GetBufferByByte<uint32_t>(offset);
        offset += 256 * sizeof(uint32_t);
        (void)offset;
        // Reuse scoreUb_ after S2 scan to stage sparse values (no extra UB).
        valueUb_ = scoreUb_;
    }

    /// TopK over [kStart, kStart + kLen) for rows [s1Start, s1Start + actualS1).
    /// Emitted indices are global TND K token ids (kStart + local), plus optional per-row offset.
    CATLASS_DEVICE void operator()(uint32_t s1Start, uint32_t actualS1, uint32_t kStart, uint32_t kLen)
    {
        if (actualS1 == 0) {
            return;
        }
        const uint32_t s2Dim = SparseIndexer::Common::CeilDiv(kLen, s2Tile_);
        const uint32_t rowsPerSub = SparseIndexer::Common::CeilDiv(actualS1, kSplitS1);
        const uint32_t sub = AscendC::GetSubBlockIdx();
        const uint32_t rowBegin = sub * rowsPerSub;
        if (rowBegin >= actualS1) {
            return;
        }
        const uint32_t rowCount = AscendC::Std::min(rowsPerSub, actualS1 - rowBegin);

        AscendC::DataCopyExtParams copyIn{1, 0, 0, 0, 0};
        AscendC::DataCopyExtParams copyOut{1, static_cast<uint32_t>(topK_ * sizeof(DtypeOut)), 0, 0, 0};
        AscendC::LocalTensor<DtypeOut> outIdxUb = stateIdxUb_.template ReinterpretCast<DtypeOut>();

        for (uint32_t ri = 0; ri < rowCount; ++ri) {
            const uint32_t s1Row = s1Start + rowBegin + ri;
            uint32_t stateCount = 0;

            for (uint32_t s2Block = 0; s2Block < s2Dim; ++s2Block) {
                const uint32_t localStart = s2Block * s2Tile_;
                const uint32_t actualS2 =
                    (s2Block == s2Dim - 1U) ? (kLen - localStart) : s2Tile_;
                const uint32_t ubColAlign = SparseIndexer::Common::AlignUp(actualS2, kFloatBlock);
                const uint32_t rightPadCols = ubColAlign - actualS2;
                const uint32_t idxOffset = kStart + localStart;

                copyIn.blockLen = actualS2 * sizeof(DtypeScore);
                AscendC::DataCopyPadExtParams<DtypeScore> padParams{
                    rightPadCols != 0, 0, static_cast<uint8_t>(rightPadCols), static_cast<DtypeScore>(-1e30f)};
                AscendC::DataCopyPad(
                    scoreUb_, gmScores_[s1Row * s2Stride_ + kStart + localStart], copyIn, padParams);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::PipeBarrier<PIPE_V>();

                uint32_t localCount = 0;
                detail::SegmentTopKOnRow(scoreUb_, scoreU32_, localKeyUb_, localIdxUb_, histUb_, actualS2, topK_,
                    idxOffset, localCount);
                detail::MergeTopKLists(stateKeyUb_, stateIdxUb_, stateCount, localKeyUb_, localIdxUb_, localCount,
                    topK_);
            }

            int32_t rowOff = 0;
            if (hasOutputIdxOffset_) {
                rowOff = gmOutputIdxOffset_.GetValue(s1Row);
            }
            // Pad missing slots with -1 (LI invalid index); apply offset only to valid slots.
            // D2b: sparse values from sortable TopK keys (invalid → -inf).
            union {
                float f;
                uint32_t u;
            } negInf;
            negInf.u = kFloatNegInfBits;
            for (uint32_t i = 0; i < topK_; ++i) {
                if (i >= stateCount) {
                    outIdxUb.SetValue(i, static_cast<DtypeOut>(-1));
                    valueUb_.SetValue(i, negInf.f);
                } else {
                    outIdxUb.SetValue(
                        i, static_cast<DtypeOut>(static_cast<int32_t>(stateIdxUb_.GetValue(i)) + rowOff));
                    valueUb_.SetValue(i, detail::SortableU32ToFloat(stateKeyUb_.GetValue(i)));
                }
            }
            AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
            AscendC::DataCopyPad(gmOut_[s1Row * topK_], outIdxUb, copyOut);
            AscendC::DataCopyExtParams copyVal{
                1, static_cast<uint32_t>(topK_ * sizeof(DtypeScore)), 0, 0, 0};
            AscendC::DataCopyPad(gmValues_[s1Row * topK_], valueUb_, copyVal);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_V>(EVENT_ID0);
        }
    }

private:
    AscendC::GlobalTensor<DtypeScore> const& gmScores_;
    AscendC::GlobalTensor<DtypeOut> const& gmOut_;
    AscendC::GlobalTensor<DtypeScore> gmValues_;
    AscendC::GlobalTensor<int32_t> gmOutputIdxOffset_;
    AscendC::LocalTensor<DtypeScore> scoreUb_;
    AscendC::LocalTensor<uint32_t> scoreU32_;
    AscendC::LocalTensor<uint32_t> stateKeyUb_;
    AscendC::LocalTensor<uint32_t> stateIdxUb_;
    AscendC::LocalTensor<uint32_t> localKeyUb_;
    AscendC::LocalTensor<uint32_t> localIdxUb_;
    AscendC::LocalTensor<uint32_t> histUb_;
    AscendC::LocalTensor<DtypeScore> valueUb_;
    uint32_t topK_{0};
    uint32_t s2Stride_{0};
    uint32_t s2Tile_{0};
    bool hasOutputIdxOffset_{false};
    uint32_t tileColAlign_{0};
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_BLOCK_TOPK_HPP
