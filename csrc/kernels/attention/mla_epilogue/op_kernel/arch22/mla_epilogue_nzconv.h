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
 * \file mla_epilogue_nzconv.h
 * \brief 设备内权重 ND→NZ 预转换（AIV 侧，按 S 门控）
 *
 * 动机：c NZ 已验证“布局换搬运效率”。B（w1+w2）仍是外部 ND，all-NZ 上限为
 *       S=96 −27.16us / S=1024 −128.32us / S=4096 −466.96us，其中大 S 的 83% 来自 w1。
 *       本文件在 kernel 内把 w1/w2 一次性转成 Matmul 期望的 NZ 布局（不改进程接口、
 *       不做跨调用缓存），转换成本按 S 门控，只在大 S 生效。
 *
 * 关键观察：**两种 NZ 布局都不需要真正的转置**
 *   w1 [G,K,N] ND → NZ [n/16][k][n%16]：源中连续 16 个 n（32 B）正好是目标中连续的一段，
 *     每个 (g,n16) 组的目标是 K*16 个元素的连续区间；搬运 = 按 32 B 块跨行的 strided 读 + 连续写。
 *   w2 [N,K] ND → NZ [k/16][n][k%16]：同理，源中连续 16 个 k（32 B）就是目标连续的一段，
 *     每个 k16 组的目标是 N*16 个元素的连续区间。
 *   因此全程只用 DataCopy 的 strided 形式（blockCount 行 × 32 B，行间隔固定），无向量转置。
 *
 * 同步：MTE2（GM→UB）/ MTE3（UB→GM）双缓冲，事件在本文件内自洽；不含 SyncAll。
 */

#ifndef __MLA_EPILOGUE_NZCONV_H__
#define __MLA_EPILOGUE_NZCONV_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mla_epilogue_tiling_data.h"

namespace NsMlaEpilogue {
using namespace AscendC;

class NzConvContext {
public:
    static constexpr uint32_t NZ_CHUNK_ROWS = 512U;      // 每个 chunk 搬运的行数
    static constexpr uint32_t NZ_CHUNK_ELEMS = 512U * 16U; // 8192 元素 = 16 KiB
    static constexpr uint32_t NZ_STAGE_BYTES = 2U * NZ_CHUNK_ELEMS * 2U; // 32 KiB 双缓冲

    __aicore__ inline void Init(const MlaEpilogueTilingData& td, TPipe* pipe, GM_ADDR userWs, GM_ADDR w1, GM_ADDR w2)
    {
        w1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w1));
        w2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(w2));
        w1NzGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWs + td.w1NzOffset));
        w2NzGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWs + td.w2NzOffset));
        pipe->InitBuffer(stageBuf_, NZ_STAGE_BYTES);
        evtMte2ToMte3_[0] = static_cast<int32_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>());
        evtMte2ToMte3_[1] = static_cast<int32_t>(pipe->AllocEventID<HardEvent::MTE2_MTE3>());
        evtMte3ToMte2_[0] = static_cast<int32_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
        evtMte3ToMte2_[1] = static_cast<int32_t>(pipe->AllocEventID<HardEvent::MTE3_MTE2>());
        pending_[0] = false;
        pending_[1] = false;
    }

    /** @brief w1：每核处理 total/cores 个 chunk，全部 8 个 batch 与 64 个 n16 组 */
    __aicore__ inline void ConvW1(const MlaEpilogueTilingData& td)
    {
        constexpr uint32_t CHUNKS_PER_GROUP = MLA_K1 / NZ_CHUNK_ROWS;   // 8
        constexpr uint32_t N16 = MLA_N1 / 16U;                          // 64
        constexpr uint32_t GROUPS = MLA_G * N16;                        // 512
        constexpr uint32_t TOTAL = GROUPS * CHUNKS_PER_GROUP;           // 4096
        constexpr uint16_t SRC_GAP = static_cast<uint16_t>((MLA_N1 * 2U - 32U) / 32U); // 63

        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t cores = td.aivUsedCores;
        uint32_t slot = 0U;
        for (uint32_t i = coreIdx; i < TOTAL; i += cores) {
            const uint32_t grp = i / CHUNKS_PER_GROUP;
            const uint32_t ck = i - grp * CHUNKS_PER_GROUP;
            const uint32_t g = grp / N16;
            const uint32_t n16 = grp - g * N16;
            const uint32_t k0 = ck * NZ_CHUNK_ROWS;
            const uint64_t srcOff = static_cast<uint64_t>(g) * MLA_K1 * MLA_N1 +
                                    static_cast<uint64_t>(k0) * MLA_N1 + static_cast<uint64_t>(n16) * 16U;
            const uint64_t dstOff = static_cast<uint64_t>(g) * MLA_K1 * MLA_N1 +
                                    (static_cast<uint64_t>(n16) * MLA_K1 + k0) * 16U;
            StageCopy(slot, w1Gm_, w1NzGm_, srcOff, dstOff, SRC_GAP);
            slot ^= 1U;
        }
        Drain();
    }

    /** @brief w1: AIV pair (2b,2b+1) converts exactly the N slice consumed by AIC b. */
    __aicore__ inline void ConvW1Private(const MlaEpilogueTilingData& td, uint32_t aicIdx, uint32_t sub)
    {
        constexpr uint32_t CHUNKS_PER_GROUP = MLA_K1 / NZ_CHUNK_ROWS;   // 8
        constexpr uint32_t N16 = MLA_N1 / 16U;                          // 64
        constexpr uint16_t SRC_GAP = static_cast<uint16_t>((MLA_N1 * 2U - 32U) / 32U); // 63
        const uint32_t nSplit = td.gemm1CoresPerBatch;
        if ((nSplit == 0U) || (aicIdx >= td.gemm1UsedCores)) {
            return;
        }
        const uint32_t g = aicIdx / nSplit;
        const uint32_t nIdx = aicIdx - g * nSplit;
        const bool isLast = ((nIdx + 1U) >= nSplit);
        const uint32_t nUnits = isLast ? td.gemm1NUnitsLast : td.gemm1NUnitsBase;
        const uint32_t n16Begin = isLast ? ((nSplit - 1U) * td.gemm1NUnitsBase) : (nIdx * td.gemm1NUnitsBase);
        const uint32_t n16End = n16Begin + nUnits;
        if ((g >= MLA_G) || (n16End > N16)) {
            return;
        }
        uint32_t slot = 0U;
        for (uint32_t n16 = n16Begin; n16 < n16End; ++n16) {
            for (uint32_t ck = sub; ck < CHUNKS_PER_GROUP; ck += 2U) {
                const uint32_t k0 = ck * NZ_CHUNK_ROWS;
                const uint64_t srcOff = static_cast<uint64_t>(g) * MLA_K1 * MLA_N1 +
                                        static_cast<uint64_t>(k0) * MLA_N1 + static_cast<uint64_t>(n16) * 16U;
                const uint64_t dstOff = static_cast<uint64_t>(g) * MLA_K1 * MLA_N1 +
                                        (static_cast<uint64_t>(n16) * MLA_K1 + k0) * 16U;
                StageCopy(slot, w1Gm_, w1NzGm_, srcOff, dstOff, SRC_GAP);
                slot ^= 1U;
            }
        }
        Drain();
    }

    /** @brief w2: AIV pair (2b,2b+1) converts exactly the N slice consumed by AIC b (original G2 partition). */
    __aicore__ inline void ConvW2Private(const MlaEpilogueTilingData& td, uint32_t aicIdx, uint32_t sub)
    {
        constexpr uint32_t K16 = MLA_K2 / 16U;                          // 512
        constexpr uint16_t SRC_GAP = static_cast<uint16_t>((MLA_K2 * 2U - 32U) / 32U); // 511
        if (aicIdx >= td.gemm2UsedCores) {
            return;
        }
        const uint32_t base2 = td.gemm2NUnitsBase;
        const uint32_t rem2 = td.gemm2NUnitsRem;
        const uint32_t hasRem = (aicIdx < rem2) ? 1U : 0U;
        const uint32_t nUnits = base2 + hasRem;
        const uint32_t n0 = MLA_ALIGN16 * (aicIdx * base2 + ((aicIdx < rem2) ? aicIdx : rem2));
        const uint32_t curN = MLA_ALIGN16 * nUnits;
        uint32_t slot = 0U;
        for (uint32_t k16 = sub; k16 < K16; k16 += 2U) {
            const uint64_t srcOff = static_cast<uint64_t>(n0) * MLA_K2 + static_cast<uint64_t>(k16) * 16U;
            const uint64_t dstOff = static_cast<uint64_t>(k16) * MLA_N2 * 16U + static_cast<uint64_t>(n0) * 16U;
            StageCopyRows(slot, w2Gm_, w2NzGm_, srcOff, dstOff, curN, SRC_GAP);
            slot ^= 1U;
        }
        Drain();
    }

    /** @brief w2：每核处理 total/cores 个 chunk，全部 512 个 k16 组 */
    __aicore__ inline void ConvW2(const MlaEpilogueTilingData& td)
    {
        constexpr uint32_t CHUNKS_PER_GROUP = MLA_N2 / NZ_CHUNK_ROWS;   // 10
        constexpr uint32_t K16 = MLA_K2 / 16U;                          // 512
        constexpr uint32_t TOTAL = K16 * CHUNKS_PER_GROUP;              // 5120
        constexpr uint16_t SRC_GAP = static_cast<uint16_t>((MLA_K2 * 2U - 32U) / 32U); // 511

        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t cores = td.aivUsedCores;
        uint32_t slot = 0U;
        for (uint32_t i = coreIdx; i < TOTAL; i += cores) {
            const uint32_t k16 = i / CHUNKS_PER_GROUP;
            const uint32_t ck = i - k16 * CHUNKS_PER_GROUP;
            const uint32_t n0 = ck * NZ_CHUNK_ROWS;
            const uint64_t srcOff = static_cast<uint64_t>(n0) * MLA_K2 + static_cast<uint64_t>(k16) * 16U;
            const uint64_t dstOff = static_cast<uint64_t>(k16) * MLA_N2 * 16U + static_cast<uint64_t>(n0) * 16U;
            StageCopy(slot, w2Gm_, w2NzGm_, srcOff, dstOff, SRC_GAP);
            slot ^= 1U;
        }
        Drain();
    }

private:
    __aicore__ inline void StageCopyRows(uint32_t slot, const GlobalTensor<bfloat16_t>& srcGm,
                                         const GlobalTensor<bfloat16_t>& dstGm, uint64_t srcOff, uint64_t dstOff,
                                         uint32_t rows, uint16_t srcGap)
    {
        if (pending_[slot]) {
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(evtMte3ToMte2_[slot]));
            pending_[slot] = false;
        }
        LocalTensor<bfloat16_t> stage = stageBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> buf = stage[slot * NZ_CHUNK_ELEMS];
        DataCopy(buf, srcGm[srcOff], DataCopyParams(static_cast<uint16_t>(rows), 1U, srcGap, 0U));
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(evtMte2ToMte3_[slot]));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(evtMte2ToMte3_[slot]));
        DataCopy(dstGm[dstOff], buf, rows * 16U);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(evtMte3ToMte2_[slot]));
        pending_[slot] = true;
    }

    __aicore__ inline void StageCopy(uint32_t slot, const GlobalTensor<bfloat16_t>& srcGm,
                                     const GlobalTensor<bfloat16_t>& dstGm, uint64_t srcOff, uint64_t dstOff,
                                     uint16_t srcGap)
    {
        if (pending_[slot]) {
            WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(evtMte3ToMte2_[slot]));
            pending_[slot] = false;
        }
        LocalTensor<bfloat16_t> stage = stageBuf_.Get<bfloat16_t>();
        LocalTensor<bfloat16_t> buf = stage[slot * NZ_CHUNK_ELEMS];
        // 512 行 × 32 B，行间源间隔 srcGap 个 32 B 块，目标连续
        DataCopy(buf, srcGm[srcOff], DataCopyParams(static_cast<uint16_t>(NZ_CHUNK_ROWS), 1U, srcGap, 0U));
        SetFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(evtMte2ToMte3_[slot]));
        WaitFlag<HardEvent::MTE2_MTE3>(static_cast<event_t>(evtMte2ToMte3_[slot]));
        DataCopy(dstGm[dstOff], buf, NZ_CHUNK_ELEMS);
        SetFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(evtMte3ToMte2_[slot]));
        pending_[slot] = true;
    }

    __aicore__ inline void Drain()
    {
        for (uint32_t s = 0U; s < 2U; ++s) {
            if (pending_[s]) {
                WaitFlag<HardEvent::MTE3_MTE2>(static_cast<event_t>(evtMte3ToMte2_[s]));
                pending_[s] = false;
            }
        }
        PipeBarrier<PIPE_ALL>();
    }

    TBuf<TPosition::VECCALC> stageBuf_;
    int32_t evtMte2ToMte3_[2] = {0, 0};
    int32_t evtMte3ToMte2_[2] = {0, 0};
    bool pending_[2] = {false, false};
    GlobalTensor<bfloat16_t> w1Gm_;    // [8, K1, N1] ND
    GlobalTensor<bfloat16_t> w2Gm_;    // [N2, K2] ND
    GlobalTensor<bfloat16_t> w1NzGm_;  // [8][N1/16][K1][16]
    GlobalTensor<bfloat16_t> w2NzGm_;  // [K2/16][N2][16]
};

} // namespace NsMlaEpilogue

#endif // __MLA_EPILOGUE_NZCONV_H__
