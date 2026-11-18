/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

#ifndef FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_MASK_POLICY_HPP
#define FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_MASK_POLICY_HPP

#include <cstdint>

#include "catlass/catlass.hpp"
#include "kernel_operator.h"

#include "mask_utils.hpp"
#include "sparse_indexer_common.hpp"

namespace SparseIndexer::Common::Block {

template <class DtypeScore>
class NoMaskPolicy {
public:
    CATLASS_DEVICE
    NoMaskPolicy(
        AscendC::LocalTensor<DtypeScore> const&, AscendC::LocalTensor<DtypeScore> const&,
        AscendC::LocalTensor<DtypeScore> const&)
    {}

    CATLASS_DEVICE
    void Apply(
        uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, int32_t,
        bool, int32_t, int32_t)
    {}
};

/// LI V2 causal/seqused/cmp mask applied on the score tile in UB.
template <class DtypeScore_>
class LiV2MaskPolicy {
public:
    using DtypeScore = DtypeScore_;
    static constexpr uint32_t kFloatBlock = Catlass::BYTE_PER_BLK / sizeof(DtypeScore);

    CATLASS_DEVICE
    LiV2MaskPolicy(
        AscendC::LocalTensor<DtypeScore> const& ubOut, AscendC::LocalTensor<DtypeScore> const& ubMaskScratch,
        AscendC::LocalTensor<DtypeScore> const& ubTmp)
        : ubOut_(ubOut), ubMaskScratch_(ubMaskScratch), ubTmp_(ubTmp)
    {}

    CATLASS_DEVICE
    void Apply(
        uint32_t actualS1, uint32_t cols, uint32_t colPerAiv, uint32_t colBegin, uint32_t s2Start,
        uint32_t localQStart, uint32_t qAct, uint32_t kAct, uint32_t reqKStart, int32_t residual,
        bool maskActive, int32_t maskMode, int32_t cmpRatio)
    {
        if (!maskActive) {
            return;
        }

        const uint32_t kBase = s2Start + colBegin - reqKStart;
        static_assert(sizeof(DtypeScore) == sizeof(uint32_t), "mask VF assumes FP32 scores");
        uint32_t calcLen = SparseIndexer::Common::AlignUp(cols, kFloatBlock);
        if (calcLen > colPerAiv) {
            calcLen = colPerAiv;
        }
        AscendC::LocalTensor<uint8_t> maskUb = ubMaskScratch_.template ReinterpretCast<uint8_t>();
        AscendC::LocalTensor<uint32_t> ubTmpU32 = ubTmp_.template ReinterpretCast<uint32_t>();

        AscendC::PipeBarrier<PIPE_V>();
        for (uint32_t ri = 0; ri < actualS1; ++ri) {
            const uint32_t localQ = localQStart + ri;
            uint32_t vis = kAct;
            if (localQ >= qAct) {
                vis = 0;
            } else if (maskMode == 3) {
                vis = Mask3VisibleLen(localQ, qAct, kAct, cmpRatio, residual);
            }
            if (vis >= kBase + cols) {
                continue;
            }

            AscendC::CreateVecIndex(ubTmp_, static_cast<DtypeScore>(0), calcLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Adds(ubTmp_, ubTmp_, static_cast<DtypeScore>(kBase), calcLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::CompareScalar(
                maskUb, ubTmp_, static_cast<DtypeScore>(vis), AscendC::CMPMODE::GE, calcLen);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Duplicate(ubTmpU32, kFloatNegInfBits, static_cast<int32_t>(calcLen));
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::Select(
                ubOut_[ri * colPerAiv], maskUb, ubTmp_, ubOut_[ri * colPerAiv],
                AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, calcLen);
        }
        AscendC::PipeBarrier<PIPE_V>();
    }

private:
    AscendC::LocalTensor<DtypeScore> const& ubOut_;
    AscendC::LocalTensor<DtypeScore> const& ubMaskScratch_;
    AscendC::LocalTensor<DtypeScore> const& ubTmp_;
};

} // namespace SparseIndexer::Common::Block

#endif // FLASHINFER_SPARSE_INDEXER_COMMON_BLOCK_MASK_POLICY_HPP
