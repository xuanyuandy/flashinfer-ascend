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
 * \file mla_epilogue_rope.h
 * \brief mla_epilogue AIV 侧 RoPE 计算上下文（DESIGN §3.8 / §3.9 / §3.10）
 *
 * 语义（spec formula）：
 *   r[s, g, h, d] = x[s, g, h, 448 + d] * cos[s, d] + x[s, g, h, 448 + (d^1)] * (-1)^d * sin[s, d]
 *   其中 x 行内布局为 [8 g][8 h][512 d]，RoPE 只作用于每 head 的 [448, 512) 共 64 维。
 *
 * UB 布局（与 Host 的 ROPE_ROW_UB / ROPE_FIXED_UB 口径一一对应，见 DESIGN §3.4 第 3 步）：
 *   xQue_   : 2 × tileS × 4096 bf16 = tileS × 16384 B   （每行 4096 = 8 head × 512）
 *   ropeF32 : 8 × 64 f32 = 2048 B
 *   swapF32 : 8 × 64 f32 = 2048 B
 *   cos/sin : 2 × 64 f32 = 512 B each
 *   sinSigned / sign / offset : 3 × 64 = 768 B
 *   固定部分合计 5888 B，与 Host ROPE_FIXED_UB 完全一致。
 *
 * 约束（DC-2 / DC-3）：本文件不含任何 SyncAll；屏障只存在于 Kernel 入口公共控制流。
 */

#ifndef __MLA_EPILOGUE_ROPE_H__
#define __MLA_EPILOGUE_ROPE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mla_epilogue_tiling_data.h"

namespace NsMlaEpilogue {
using namespace AscendC;

constexpr uint32_t MLA_X_ROW = 32768;      // x 每行 bf16 元素数：64 head × 512
constexpr uint32_t MLA_X_GROUP = 4096;     // 每个 batch g 的 bf16 元素数：8 head × 512
constexpr uint32_t MLA_ROPE_OFF = 448;     // 每 head 内 RoPE 切片起始列
constexpr uint32_t MLA_ROPE_D = 64;        // RoPE 维（rotate_every_two 的配对长度）
constexpr uint32_t MLA_ROPE_HEADS = 8;     // 每 batch g 的 head 数
constexpr uint32_t MLA_ROPE_F32_NUM = MLA_ROPE_HEADS * MLA_ROPE_D; // 512

/**
 * @brief AIV 侧 RoPE 上下文：tile 输入缓冲 + 固定 f32 工作区
 */
class RopeContext {
public:
    __aicore__ inline void Init(const MlaEpilogueTilingData& td, TPipe* pipe, GM_ADDR userWs, GM_ADDR x, GM_ADDR cos,
                                GM_ADDR sin)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(x));
        cosGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cos));
        sinGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sin));
        rGm_.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(userWs + td.rOffset));

        // 一整行覆盖全部 group：32768 bf16 = 64 KiB，双缓冲峰值与原 tileS=8 一致。
        pipe->InitBuffer(xQue_, 2, MLA_X_ROW * static_cast<uint32_t>(sizeof(bfloat16_t)));
        pipe->InitBuffer(cosQue_, 2, MLA_ROPE_D * static_cast<uint32_t>(sizeof(float)));
        pipe->InitBuffer(sinQue_, 2, MLA_ROPE_D * static_cast<uint32_t>(sizeof(float)));
        pipe->InitBuffer(ropeBuf_, MLA_ROPE_F32_NUM * static_cast<uint32_t>(sizeof(float)));
        pipe->InitBuffer(swapBuf_, MLA_ROPE_F32_NUM * static_cast<uint32_t>(sizeof(float)));
        pipe->InitBuffer(sinSignedBuf_, MLA_ROPE_D * static_cast<uint32_t>(sizeof(float)));
        pipe->InitBuffer(signBuf_, MLA_ROPE_D * static_cast<uint32_t>(sizeof(float)));
        pipe->InitBuffer(offBuf_, MLA_ROPE_D * static_cast<uint32_t>(sizeof(uint32_t)));

        // rotate_every_two：pair 内第偶数元素取负号、奇数元素保持正号；offset = (d^1) 的字节偏移
        LocalTensor<float> sign = signBuf_.Get<float>();
        LocalTensor<uint32_t> off = offBuf_.Get<uint32_t>();
        for (uint32_t d = 0U; d < MLA_ROPE_D; ++d) {
            sign.SetValue(d, ((d & 1U) == 0U) ? -1.0f : 1.0f);
            // Gather 的 srcOffset 单位为字节（CANN 9.1.0 Gather Level 2 约束）
            off.SetValue(d, (d ^ 1U) * static_cast<uint32_t>(sizeof(float)));
        }
        PipeBarrier<PIPE_ALL>(); // 标量写 UB 完成后才能被 V 读取
    }

    /**
     * @brief 处理第 t 个流水段（行区间 [t*pipeTileRows, ...)）
     *        任务空间：pipeTileRows 个行任务；每行一次处理全部 8 个 group，
     *        cos/sin 每行只加载一次，按核号连续分配；
     *        末段额外清零 r 的填充行 [totalS, paddedM)（必须在屏障前完成）。
     *        纯计算体，不含 SyncAll。
     */
    __aicore__ inline void Tile(uint32_t t, const MlaEpilogueTilingData& td)
    {
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        if (coreIdx >= td.aivUsedCores) {
            return; // 未承担任务的核：只跳过计算体，仍参与入口屏障
        }
        const uint32_t m0 = t * td.pipeTileRows;
        const uint32_t taskBegin = coreIdx * td.ropeTasksPerCore;
        const uint32_t taskEndBase = taskBegin + td.ropeTasksPerCore;
        const uint32_t taskEnd = (taskEndBase < td.ropeTaskNum) ? taskEndBase : td.ropeTaskNum;

        for (uint32_t taskIdx = taskBegin; taskIdx < taskEnd; ++taskIdx) {
            const uint32_t row = m0 + taskIdx;
            if (row < td.totalS) {
                ProcessRow(row, td);
            }
        }

        if (((t + 1U) == td.pipeTileNum) && (td.padRows != 0U)) {
            ClearPadRows(td);
        }
    }

private:
    /**
     * @brief 单个行任务：整行加载一次，cos/sin 复用到全部 8 个 group，再分别写回 r[g]。
     */
    __aicore__ inline void ProcessRow(uint32_t row, const MlaEpilogueTilingData& td)
    {
        LocalTensor<float> rope = ropeBuf_.Get<float>();
        LocalTensor<float> swp = swapBuf_.Get<float>();
        LocalTensor<float> sinSigned = sinSignedBuf_.Get<float>();
        LocalTensor<float> sign = signBuf_.Get<float>();
        LocalTensor<uint32_t> off = offBuf_.Get<uint32_t>();

        LocalTensor<bfloat16_t> xTile = xQue_.AllocTensor<bfloat16_t>();
        DataCopy(xTile, xGm_[static_cast<uint64_t>(row) * MLA_X_ROW], MLA_X_ROW);
        xQue_.EnQue(xTile); // MTE2 → V
        xTile = xQue_.DeQue<bfloat16_t>();

        LocalTensor<float> cosV = cosQue_.AllocTensor<float>();
        LocalTensor<float> sinV = sinQue_.AllocTensor<float>();
        const uint64_t csOff = static_cast<uint64_t>(row) * MLA_ROPE_D;
        DataCopy(cosV, cosGm_[csOff], MLA_ROPE_D);
        DataCopy(sinV, sinGm_[csOff], MLA_ROPE_D);
        cosQue_.EnQue(cosV);
        sinQue_.EnQue(sinV);
        cosV = cosQue_.DeQue<float>();
        sinV = sinQue_.DeQue<float>();

        // sinSigned[d] = sin[d] * (-1)^d；同一行内被全部 group 复用。
        Mul(sinSigned, sinV, sign, MLA_ROPE_D, 1, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
        for (uint32_t g = 0U; g < MLA_G; ++g) {
            const uint64_t groupOff = static_cast<uint64_t>(g) * MLA_X_GROUP;
            // bf16 → f32（每 head 64 维，共 8 段，段间源跨距 512 元素）
            for (uint32_t h = 0U; h < MLA_ROPE_HEADS; ++h) {
                Cast(rope[h * MLA_ROPE_D], xTile[groupOff + h * 512U + MLA_ROPE_OFF],
                     RoundMode::CAST_NONE, MLA_ROPE_D);
            }
            // swap[h][d] = rope[h][d^1]（srcOffset 单位为字节）
            for (uint32_t h = 0U; h < MLA_ROPE_HEADS; ++h) {
                Gather(swp[h * MLA_ROPE_D], rope[h * MLA_ROPE_D], off, 0U, MLA_ROPE_D);
            }
            // out = rope * cos + swap * sinSigned（cos/sin 在 8 个 head 间广播：src1RepStride = 0）
            Mul(rope, rope, cosV, MLA_ROPE_D, MLA_ROPE_HEADS, BinaryRepeatParams(1, 1, 1, 8, 8, 0));
            Mul(swp, swp, sinSigned, MLA_ROPE_D, MLA_ROPE_HEADS, BinaryRepeatParams(1, 1, 1, 8, 8, 0));
            Add(rope, rope, swp, MLA_ROPE_D, MLA_ROPE_HEADS, BinaryRepeatParams(1, 1, 1, 8, 8, 8));
            // f32 → bf16（CAST_RINT：与 GEMM 输出的 BF16 舍入口径一致）
            for (uint32_t h = 0U; h < MLA_ROPE_HEADS; ++h) {
                Cast(xTile[groupOff + h * 512U + MLA_ROPE_OFF], rope[h * MLA_ROPE_D],
                     RoundMode::CAST_RINT, MLA_ROPE_D);
            }
        }

        cosQue_.FreeTensor(cosV);
        sinQue_.FreeTensor(sinV);
        PipeBarrier<PIPE_ALL>(); // V 完成后再发起 MTE3
        for (uint32_t g = 0U; g < MLA_G; ++g) {
            DataCopy(rGm_[static_cast<uint64_t>(g) * td.paddedM * MLA_X_GROUP +
                          static_cast<uint64_t>(row) * MLA_X_GROUP],
                     xTile[static_cast<uint64_t>(g) * MLA_X_GROUP], MLA_X_GROUP);
        }
        PipeBarrier<PIPE_ALL>(); // MTE3 完成后再归还缓冲
        xQue_.FreeTensor(xTile);
    }

    /**
     * @brief 清零 r 的填充行：r[g][totalS + p] = 0（g ∈ [0,8), p ∈ [0,padRows)）
     */
    __aicore__ inline void ClearPadRows(const MlaEpilogueTilingData& td)
    {
        const uint32_t coreIdx = static_cast<uint32_t>(GetBlockIdx());
        const uint32_t padBlocks = MLA_G * td.padRows;
        LocalTensor<bfloat16_t> zeroTile = xQue_.AllocTensor<bfloat16_t>();
        Duplicate(zeroTile, static_cast<bfloat16_t>(0), MLA_X_GROUP);
        PipeBarrier<PIPE_ALL>(); // V 完成后才能发起 MTE3
        for (uint32_t p = coreIdx; p < padBlocks; p += td.aivUsedCores) {
            const uint32_t g = p / td.padRows;
            const uint32_t row = td.totalS + (p - g * td.padRows);
            DataCopy(rGm_[static_cast<uint64_t>(g) * td.paddedM * MLA_X_GROUP +
                          static_cast<uint64_t>(row) * MLA_X_GROUP],
                     zeroTile, MLA_X_GROUP);
        }
        PipeBarrier<PIPE_ALL>(); // MTE3 完成后再归还缓冲
        xQue_.FreeTensor(zeroTile);
    }

    TQue<TPosition::VECIN, 2> xQue_;   // tileS 行 × 4096 bf16，双缓冲
    TQue<TPosition::VECIN, 2> cosQue_; // 每行 64 f32，双缓冲
    TQue<TPosition::VECIN, 2> sinQue_; // 每行 64 f32，双缓冲
    TBuf<TPosition::VECCALC> ropeBuf_;
    TBuf<TPosition::VECCALC> swapBuf_;
    TBuf<TPosition::VECCALC> sinSignedBuf_;
    TBuf<TPosition::VECCALC> signBuf_;
    TBuf<TPosition::VECCALC> offBuf_;

    GlobalTensor<bfloat16_t> xGm_;  // [S, 1, 64, 512]
    GlobalTensor<float> cosGm_;     // [S, 1, 1, 64]
    GlobalTensor<float> sinGm_;     // [S, 1, 1, 64]
    GlobalTensor<bfloat16_t> rGm_;  // [8, paddedM, 4096]
};

} // namespace NsMlaEpilogue

#endif // __MLA_EPILOGUE_ROPE_H__
