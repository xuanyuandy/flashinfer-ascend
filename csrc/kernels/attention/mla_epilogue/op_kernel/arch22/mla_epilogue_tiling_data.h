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
 * \file mla_epilogue_tiling_data.h
 * \brief mla_epilogue 自定义 TilingData 结构体（Host/Kernel 共享，DESIGN §3.3）
 *
 * 注意：本结构体由 Host tiling 写入、Kernel 通过 GET_TILING_DATA_WITH_STRUCT 读取，
 * 两侧必须使用同一份定义，禁止宏封装（不使用 BEGIN_TILING_DATA_DEF）。
 */

#ifndef __MLA_EPILOGUE_TILING_DATA_H__
#define __MLA_EPILOGUE_TILING_DATA_H__

#include "kernel_tiling/kernel_tiling.h"

// 固定维度（spec shape_constraints，不进入 TilingData）
constexpr uint32_t MLA_G = 8;         // BMM batch 数
constexpr uint32_t MLA_K1 = 4096;     // 第一段 GEMM 的 K
constexpr uint32_t MLA_N1 = 1024;     // 第一段 GEMM 的 N
constexpr uint32_t MLA_K2 = 8192;     // 第二段 GEMM 的 K
constexpr uint32_t MLA_N2 = 5120;     // 第二段 GEMM 的 N
constexpr uint32_t MLA_ALIGN16 = 16;  // M/N 分块对齐

#pragma pack(push, 8)
struct MlaEpilogueTilingData {
    uint32_t totalS;                  // 真实 S（1..131072）
    uint32_t paddedM;                 // alignUp(totalS, 16)，r/c 的 M 方向分配行数
    uint32_t padRows;                 // paddedM - totalS，需清零的填充行数
    uint32_t tileS;                   // RoPE 每 tile 行数 T_s（由 UB 动态推导，≤ 8）
    uint32_t pipeTileNum;             // M-tile 流水段数 T（Host 推导）
    uint32_t pipeTileRows;            // 每个流水段的最大 M 行数（16 对齐；末段可更小）
    uint32_t sTileNum;                // ceil(pipeTileRows / tileS)，单流水段内的最大 s-tile 数
    uint32_t ropeTaskNum;             // MLA_G * sTileNum，单流水段内的最大任务数
    uint32_t ropeTasksPerCore;        // ceil(ropeTaskNum / aivUsedCores)
    uint32_t aivUsedCores;            // AIV 数量（Host GetCoreNumAiv()）
    uint32_t aicUsedCores;            // AIC 实际参与核数（Host GetCoreNumAic()）

    uint32_t gemm1CoresPerBatch;      // nSplit = max(1, aicUsedCores / MLA_G)
    uint32_t gemm1NUnitsBase;         // 前 nSplit-1 个核各自的 16 元素单元数（128 B 对齐）
    uint32_t gemm1NUnitsLast;         // 最后一个核的 16 元素单元数（吃掉 n1Units 的余数）
    uint32_t gemm1SingleNMax;         // 16 * max(base, last)，Host tiling 用 N
    uint32_t gemm1SingleM;            // = pipeTileRows（第一段按流水段 M 计算）
    uint32_t gemm1UsedCores;          // = MLA_G * nSplit

    uint32_t gemm2NUnitsBase;         // (MLA_N2/16) / aicUsedCores
    uint32_t gemm2NUnitsRem;          // (MLA_N2/16) % aicUsedCores
    uint32_t gemm2SingleNMax;         // 16 * (base + (rem ? 1 : 0))
    uint32_t gemm2SingleM;            // = totalS（第二段按真实 M 计算）
    uint32_t gemm2UsedCores;          // = aicUsedCores

    uint32_t sysWorkspaceSize;        // Host PlatformAscendC::GetLibApiWorkSpaceSize() 返回的系统 workspace 字节数
    uint32_t reserved;                // 8 字节对齐占位
    uint64_t rOffset;                 // r 相对用户 workspace 起点的字节偏移（=0）
    uint64_t cOffset;                 // c 相对用户 workspace 起点的字节偏移
    uint64_t userWorkspaceSize;       // r 区 + c 区（含对齐余量）
};
#pragma pack(pop)

#endif // __MLA_EPILOGUE_TILING_DATA_H__
