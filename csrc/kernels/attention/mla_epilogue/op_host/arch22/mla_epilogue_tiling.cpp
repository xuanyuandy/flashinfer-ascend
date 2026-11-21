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
 * \file mla_epilogue_tiling.cpp
 * \brief mla_epilogue Host 侧 Tiling 实现（arch22 / DAV_2201，DESIGN §3.4）
 *
 */

#include <string>
#include <vector>

#include "register/op_def_registry.h"
#include "op_common/log/log.h"
#include "op_common/op_host/util/math_util.h"
#include "op_common/op_host/util/platform_util.h"
#include "tiling/tiling_api.h"
#include "base/err_msg.h"
#include "../../op_kernel/arch22/mla_epilogue_tiling_data.h"
#include "../../op_kernel/arch22/mla_epilogue_tiling_key.h"

namespace optiling {

constexpr size_t WORKSPACE_NUM = 1U;

// RoPE UB 口径（DESIGN §3.4 第 3 步 / §3.9 / §3.10）
constexpr uint64_t UB_RESERVED_SIZE = 8192ULL;      // 框架预留
constexpr uint64_t ROPE_ROW_UB = 16384ULL;          // 每行双缓冲 xTile：2 × 4096 × 2 B
constexpr uint64_t ROPE_FIXED_UB = 5888ULL;         // ropeF32 + swapF32 + cos/sin 双缓冲 + 3 个辅助向量
constexpr uint32_t TILE_S_CAP = 8U;                 // 设计余量封顶
constexpr uint32_t WS_ALIGN = 512U;

// shape 固定维度（spec shape_constraints）
constexpr int64_t DIM_X_S_HEAD = 64;
constexpr int64_t DIM_X_D = 512;
constexpr int64_t DIM_ROPE = 64;
constexpr int64_t MLA_MAX_S = 131072;

static inline uint32_t CeilDivU32(uint32_t a, uint32_t b)
{
    return (a + b - 1U) / b;
}

static inline uint32_t AlignUpU32(uint32_t a, uint32_t align)
{
    return (a + align - 1U) / align * align;
}

static inline uint64_t AlignUpU64(uint64_t a, uint64_t align)
{
    return (a + align - 1U) / align * align;
}

// =============================================================================
// 输入校验（DESIGN §3.4 第 1 步；对应 spec dtype_policy / shape_constraints / boundary_conditions）
// =============================================================================

// 目标版本 op_common/log/log.h 的 Ops::Base::ToString 实现位于 libops_base.so，
// 而本算子 tiling 库的链接链（libnnopbase / libregister / libtiling_api）未引入该库，
// 直接引用会留下加载期无法解析的强符号，故在本文件内实现等价的字符串拼接。
static std::string DataTypeToString(ge::DataType dataType)
{
    switch (dataType) {
        case ge::DT_FLOAT:
            return "FLOAT";
        case ge::DT_FLOAT16:
            return "FLOAT16";
        case ge::DT_BF16:
            return "BF16";
        case ge::DT_INT8:
            return "INT8";
        case ge::DT_INT16:
            return "INT16";
        case ge::DT_INT32:
            return "INT32";
        case ge::DT_INT64:
            return "INT64";
        case ge::DT_UINT8:
            return "UINT8";
        case ge::DT_DOUBLE:
            return "DOUBLE";
        case ge::DT_BOOL:
            return "BOOL";
        default:
            return "DT_" + std::to_string(static_cast<int32_t>(dataType));
    }
}

static std::string ShapeToString(const gert::Shape& shape)
{
    std::string result = "[";
    for (size_t i = 0U; i < shape.GetDimNum(); ++i) {
        if (i != 0U) {
            result += ",";
        }
        result += std::to_string(shape.GetDim(i));
    }
    result += "]";
    return result;
}

static std::string DimToString(size_t dimNum)
{
    return std::to_string(dimNum) + "D";
}

// 合法输入下的 Tiling / 平台 / 库调用失败：由根因层首报一次 EZ1008（Arglist: opName, reason），
// 调用点负责构造包含原始返回码与关键入参的可操作诊断。
static ge::graphStatus ReportTilingFailure(gert::TilingContext* context, const std::string& reason)
{
    OP_LOGE_WITHOUT_REPORT(context, "%s", reason.c_str());
    const std::string opInfo = Ops::Base::GetOpInfo(context);
    const std::vector<const char*> msgKey = {"opName", "reason"};
    const std::vector<const char*> msgValue = {opInfo.c_str(), reason.c_str()};
    (void)error_message::ReportPredefinedErrMsg("EZ1008", msgKey, msgValue);
    return ge::GRAPH_FAILED;
}

static ge::graphStatus CheckInputDtypes(gert::TilingContext* context)
{
    const gert::CompileTimeTensorDesc* xDesc = context->GetInputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xDesc);
    const gert::CompileTimeTensorDesc* cosDesc = context->GetInputDesc(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, cosDesc);
    const gert::CompileTimeTensorDesc* sinDesc = context->GetInputDesc(2);
    OP_CHECK_NULL_WITH_CONTEXT(context, sinDesc);
    const gert::CompileTimeTensorDesc* w1Desc = context->GetInputDesc(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, w1Desc);
    const gert::CompileTimeTensorDesc* w2Desc = context->GetInputDesc(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, w2Desc);
    const gert::CompileTimeTensorDesc* yDesc = context->GetOutputDesc(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, yDesc);

    const std::string opInfo = Ops::Base::GetOpInfo(context);
    const ge::DataType xDt = xDesc->GetDataType();
    if (xDt != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(opInfo, "x", DataTypeToString(xDt), "BF16");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType cosDt = cosDesc->GetDataType();
    if (cosDt != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(opInfo, "cos", DataTypeToString(cosDt), "FLOAT");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType sinDt = sinDesc->GetDataType();
    if (sinDt != ge::DT_FLOAT) {
        OP_LOGE_FOR_INVALID_DTYPE(opInfo, "sin", DataTypeToString(sinDt), "FLOAT");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType w1Dt = w1Desc->GetDataType();
    if (w1Dt != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(opInfo, "w1", DataTypeToString(w1Dt), "BF16");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType w2Dt = w2Desc->GetDataType();
    if (w2Dt != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(opInfo, "w2", DataTypeToString(w2Dt), "BF16");
        return ge::GRAPH_FAILED;
    }
    const ge::DataType yDt = yDesc->GetDataType();
    if (yDt != ge::DT_BF16) {
        OP_LOGE_FOR_INVALID_DTYPE(opInfo, "y", DataTypeToString(yDt), "BF16");
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus CheckInputShapes(gert::TilingContext* context, uint32_t& totalS)
{
    const std::string opInfo = Ops::Base::GetOpInfo(context);

    // x: [S, 1, 64, 512]
    const gert::StorageShape* xStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, xStorageShape);
    const gert::Shape& xShape = xStorageShape->GetStorageShape();
    if (xShape.GetDimNum() != 4U) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opInfo, "x", DimToString(xShape.GetDimNum()), "4D");
        return ge::GRAPH_FAILED;
    }
    if (xShape.GetDim(1) != 1 || xShape.GetDim(2) != DIM_X_S_HEAD || xShape.GetDim(3) != DIM_X_D) {
        OP_LOGE_FOR_INVALID_SHAPE(opInfo, "x", ShapeToString(xShape), "[S,1,64,512]");
        return ge::GRAPH_FAILED;
    }
    const int64_t s = xShape.GetDim(0);
    if (s < 1 || s > MLA_MAX_S) {
        OP_LOGE_FOR_INVALID_VALUE_WITH_REASON(opInfo, "x", std::to_string(s),
                                             "S must be in [1,131072]");
        return ge::GRAPH_FAILED;
    }

    // cos / sin: [S, 1, 1, 64]
    const char* const cosSinNames[] = {"cos", "sin"};
    for (size_t i = 1U; i <= 2U; ++i) {
        const gert::StorageShape* csStorageShape = context->GetInputShape(i);
        OP_CHECK_NULL_WITH_CONTEXT(context, csStorageShape);
        const gert::Shape& csShape = csStorageShape->GetStorageShape();
        if (csShape.GetDimNum() != 4U) {
            OP_LOGE_FOR_INVALID_SHAPEDIM(opInfo, cosSinNames[i - 1U], DimToString(csShape.GetDimNum()), "4D");
            return ge::GRAPH_FAILED;
        }
        if (csShape.GetDim(1) != 1 || csShape.GetDim(2) != 1 || csShape.GetDim(3) != DIM_ROPE) {
            OP_LOGE_FOR_INVALID_SHAPE(opInfo, cosSinNames[i - 1U], ShapeToString(csShape), "[S,1,1,64]");
            return ge::GRAPH_FAILED;
        }
        if (csShape.GetDim(0) != s) {
            const std::string paramNames = "x," + std::string(cosSinNames[i - 1U]);
            const std::string incorrectShapes = ShapeToString(xShape) + "," + ShapeToString(csShape);
            const std::string reason =
                "the S dimension of " + std::string(cosSinNames[i - 1U]) + " must equal the S dimension of x";
            OP_LOGE_FOR_INVALID_SHAPES_WITH_REASON(opInfo, paramNames, incorrectShapes, reason);
            return ge::GRAPH_FAILED;
        }
    }

    // w1: [8, 4096, 1024]
    const gert::StorageShape* w1StorageShape = context->GetInputShape(3);
    OP_CHECK_NULL_WITH_CONTEXT(context, w1StorageShape);
    const gert::Shape& w1Shape = w1StorageShape->GetStorageShape();
    if (w1Shape.GetDimNum() != 3U) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opInfo, "w1", DimToString(w1Shape.GetDimNum()), "3D");
        return ge::GRAPH_FAILED;
    }
    if (w1Shape.GetDim(0) != static_cast<int64_t>(MLA_G) ||
        w1Shape.GetDim(1) != static_cast<int64_t>(MLA_K1) ||
        w1Shape.GetDim(2) != static_cast<int64_t>(MLA_N1)) {
        OP_LOGE_FOR_INVALID_SHAPE(opInfo, "w1", ShapeToString(w1Shape), "[8,4096,1024]");
        return ge::GRAPH_FAILED;
    }

    // w2: [5120, 8192]
    const gert::StorageShape* w2StorageShape = context->GetInputShape(4);
    OP_CHECK_NULL_WITH_CONTEXT(context, w2StorageShape);
    const gert::Shape& w2Shape = w2StorageShape->GetStorageShape();
    if (w2Shape.GetDimNum() != 2U) {
        OP_LOGE_FOR_INVALID_SHAPEDIM(opInfo, "w2", DimToString(w2Shape.GetDimNum()), "2D");
        return ge::GRAPH_FAILED;
    }
    if (w2Shape.GetDim(0) != static_cast<int64_t>(MLA_N2) ||
        w2Shape.GetDim(1) != static_cast<int64_t>(MLA_K2)) {
        OP_LOGE_FOR_INVALID_SHAPE(opInfo, "w2", ShapeToString(w2Shape), "[5120,8192]");
        return ge::GRAPH_FAILED;
    }

    totalS = static_cast<uint32_t>(s);
    return ge::GRAPH_SUCCESS;
}


// =============================================================================
// Tiling 主体
// =============================================================================

static ge::graphStatus TilingMlaEpilogue(gert::TilingContext* context)
{
    // ---- 1. 平台信息（运行时获取，DC-6）----
    fe::PlatFormInfos* platformInfoPtr = context->GetPlatformInfo();
    OP_CHECK_NULL_WITH_CONTEXT(context, platformInfoPtr);
    platform_ascendc::PlatformAscendC platform(platformInfoPtr);
    const uint32_t aicNum = platform.GetCoreNumAic();
    const uint32_t aivNum = platform.GetCoreNumAiv();
    uint64_t ubSize = 0ULL;
    platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    if (aicNum == 0U) {
        return ReportTilingFailure(context, "GetCoreNumAic() returned 0");
    }
    if (aivNum == 0U) {
        return ReportTilingFailure(context, "GetCoreNumAiv() returned 0");
    }
    if (ubSize == 0U) {
        return ReportTilingFailure(context, "GetCoreMemSize(UB) returned 0");
    }

    // ---- 2. 输入校验 ----
    // 参数错误已在 CheckInputDtypes / CheckInputShapes 内部完成正式首报，此处仅记录上下文并透传。
    ge::graphStatus checkStatus = CheckInputDtypes(context);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "CheckInputDtypes failed, rc=%u", static_cast<unsigned int>(checkStatus));
        return checkStatus;
    }
    uint32_t totalS = 0U;
    checkStatus = CheckInputShapes(context, totalS);
    if (checkStatus != ge::GRAPH_SUCCESS) {
        OP_LOGE_WITHOUT_REPORT(context, "CheckInputShapes failed, rc=%u", static_cast<unsigned int>(checkStatus));
        return checkStatus;
    }

    // ---- 3. TilingData ----
    MlaEpilogueTilingData* tiling = context->GetTilingData<MlaEpilogueTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    const errno_t memsetRc = memset_s(tiling, sizeof(MlaEpilogueTilingData), 0, sizeof(MlaEpilogueTilingData));
    if (memsetRc != EOK) {
        return ReportTilingFailure(context, "memset_s(MlaEpilogueTilingData) failed, errno=" + std::to_string(memsetRc));
    }

    // ---- 4. M 对齐与流水段划分 ----
    const uint32_t paddedM = AlignUpU32(totalS, MLA_ALIGN16);
    const uint32_t padRows = paddedM - totalS;

    // RoPE 每 tile 行数：先算容量上限（§3.4 第 3 步），容量公式与 §3.9/§3.10 统一
    const uint64_t usableUB = (ubSize > UB_RESERVED_SIZE) ? (ubSize - UB_RESERVED_SIZE) : 0ULL;
    uint32_t tileSMax = 1U;
    if (usableUB > ROPE_FIXED_UB) {
        const uint64_t cap = (usableUB - ROPE_FIXED_UB) / ROPE_ROW_UB;
        tileSMax = static_cast<uint32_t>(cap < 1ULL ? 1ULL : (cap > TILE_S_CAP ? TILE_S_CAP : cap));
    }

    // M-tile 流水段划分（性能优化：单段）：
    //   原设计 ptn=2 让 RoPE(t+1) 与 GEMM1(t) 重叠，但每个流水段都要完整遍历一遍 w1：
    //   实测（S=96）GEMM1 阶段 AIC 的 L2 读请求量为 180.0 MB（msopprof Memory.read_main_memory，
    //   128 B 请求粒度计数），而 w1 唯一数据量仅 67.1 MB（=65,536 KB）——第二段对 w1 的请求
    //   再次穿到 L2/HBM，违反 REQUIREMENTS §4.8.1「W1/W2 各只读一次」。
    //   改为单段（ptn=1）后每核只读一遍自己那份 w1（全核合计 = 65,536 KB）。代价是 RoPE 回到
    //   串行：全部 RoPE 在 B0 之前的首段完成，不再与 GEMM1 重叠；S=96 实测 RoPE 串行暴露仅
    //   2.44 us（阶段隔离：lead-RoPE+GEMM1 114.50 vs GEMM1 112.06），远小于 w1 单趟化省下的
    //   58.8 us。整核（S=96）296.7 → 209.2 us（msopprof op-level 21 会话中位数）。
    const uint32_t pipeTileNum = 1U;
    const uint32_t pipeTileRows = paddedM;

    // 核利用率收缩：单段任务空间需覆盖全部 AIV
    const uint32_t sTilesNeeded = CeilDivU32(aivNum, MLA_G);
    uint32_t tileS = pipeTileRows / sTilesNeeded;
    if (tileS < 1U) {
        tileS = 1U;
    }
    if (tileS > tileSMax) {
        tileS = tileSMax;
    }
    const uint32_t sTileNum = CeilDivU32(pipeTileRows, tileS);
    // Row-major RoPE：每个任务处理一整行的全部 8 个 group，cos/sin 每行只加载一次。
    const uint32_t ropeTaskNum = pipeTileRows;
    const uint32_t ropeTasksPerCore = CeilDivU32(ropeTaskNum, aivNum);
    const uint32_t aivUsedCores = CeilDivU32(ropeTaskNum, ropeTasksPerCore);

    // 一行覆盖全部 group（32768 bf16 = 64 KiB），双缓冲与原 tileS=8 的峰值相同。
    const uint64_t ropeUB = static_cast<uint64_t>(TILE_S_CAP) * ROPE_ROW_UB + ROPE_FIXED_UB;
    if (ropeUB > usableUB) {
        return ReportTilingFailure(context, "RoPE UB peak " + std::to_string(ropeUB) + " > usableUB " +
                                                 std::to_string(usableUB) +
                                                 " (tileS=" + std::to_string(tileS) + ")");
    }

    // ---- 5. GEMM1 分核（只切 N，权重只读一次）----
    uint32_t nSplit = aicNum / MLA_G;
    if (nSplit < 1U) {
        nSplit = 1U; // 目标平台不成立（DC-7）；Kernel 用 aicUsedCores/nSplit 作为 g 步长兜底
    }
    const uint32_t n1Units = MLA_N1 / MLA_ALIGN16; // 64
    // 128 B 对齐分核（性能优化）：MTE2 的 GM→L1 请求粒度为 128 B。若各核 N 块起点只落在
    // 64 B 边界（均匀切分为 352/336/336 元素 → 704/672/672 B），除首核外每个 K 行读都会跨
    // sector，实测非首核 MTE2 时间高出约 1.5×（121 us vs 82 us）。故把每核单元数收成
    // GEMM1_UNIT_ALIGN 个 16 元素单元（64 元素 = 128 B）的整数倍，余数全部给最后一个核：
    //   nSplit=3, n1Units=64 → base=20（320 元素）, last=24（384 元素），n0 = 0/320/640 元素
    // 全程无 64 B 边界起点，singleNMax = 384（320/320/384）。
    const uint32_t gemm1UnitAlign = 4U; // 4 × 16 = 64 元素 = 128 B
    uint32_t base1 = n1Units;
    uint32_t last1 = n1Units;
    if (nSplit > 1U) {
        const uint32_t baseAligned = ((n1Units / nSplit) / gemm1UnitAlign) * gemm1UnitAlign;
        // 兜底：平均单元数不足一个对齐单元时退回均匀切分（正确性不受影响，仅损失对齐收益）
        base1 = (baseAligned > 0U) ? baseAligned : (n1Units / nSplit);
        last1 = n1Units - base1 * (nSplit - 1U);
    }
    if ((base1 == 0U) || (base1 * (nSplit - 1U) + last1 != n1Units)) {
        return ReportTilingFailure(context, "GEMM1 N split invalid (nSplit=" + std::to_string(nSplit) +
                                                 " base=" + std::to_string(base1) +
                                                 " last=" + std::to_string(last1) +
                                                 " units=" + std::to_string(n1Units) + ")");
    }
    const uint32_t gemm1SingleNMax = MLA_ALIGN16 * ((base1 > last1) ? base1 : last1);
    const uint32_t gemm1UsedCores = MLA_G * nSplit;

    // ---- 6. GEMM2 分核（只切 N，全部 aicNum 核参与）----
    const uint32_t n2Units = MLA_N2 / MLA_ALIGN16; // 320
    const uint32_t gemm2Cores = aicNum;
    const uint32_t base2 = n2Units / gemm2Cores;
    const uint32_t rem2 = n2Units % gemm2Cores;
    const uint32_t gemm2SingleNMax = MLA_ALIGN16 * (base2 + ((rem2 != 0U) ? 1U : 0U));
    const uint32_t gemm2UsedCores = gemm2Cores;
    if (base2 == 0U) {
        return ReportTilingFailure(context, "GEMM2 cores " + std::to_string(gemm2Cores) +
                                                 " > N units " + std::to_string(n2Units));
    }

    // ---- 9. workspace（用户区 r/c + 系统区）----
    const uint64_t rBytes =
        static_cast<uint64_t>(MLA_G) * static_cast<uint64_t>(paddedM) * static_cast<uint64_t>(MLA_K1) * 2ULL;
    const uint64_t cBytes =
        static_cast<uint64_t>(paddedM) * static_cast<uint64_t>(MLA_G) * static_cast<uint64_t>(MLA_N1) * 2ULL;
    // CATLASS 只使用 GM 上的 r/c 中间结果；w1/w2 直接以 ND 读取，不再生成 NZ 副本。
    const uint64_t cOffset = AlignUpU64(rBytes, WS_ALIGN);
    const uint64_t userWorkspaceSize = cOffset + AlignUpU64(cBytes, WS_ALIGN);
    const uint32_t sysWorkspaceSize = platform.GetLibApiWorkSpaceSize();
    if (sysWorkspaceSize == UINT32_MAX) {
        return ReportTilingFailure(context, "GetLibApiWorkSpaceSize() returned UINT32_MAX");
    }

    size_t* currentWorkspace = context->GetWorkspaceSizes(WORKSPACE_NUM);
    OP_CHECK_NULL_WITH_CONTEXT(context, currentWorkspace);
    currentWorkspace[0] = static_cast<size_t>(userWorkspaceSize + static_cast<uint64_t>(sysWorkspaceSize));

    // ---- 10. TilingData 字段落盘 ----
    tiling->totalS = totalS;
    tiling->paddedM = paddedM;
    tiling->padRows = padRows;
    tiling->tileS = tileS;
    tiling->pipeTileNum = pipeTileNum;
    tiling->pipeTileRows = pipeTileRows;
    tiling->sTileNum = sTileNum;
    tiling->ropeTaskNum = ropeTaskNum;
    tiling->ropeTasksPerCore = ropeTasksPerCore;
    tiling->aivUsedCores = aivUsedCores;
    tiling->aicUsedCores = aicNum;

    tiling->gemm1CoresPerBatch = nSplit;
    tiling->gemm1NUnitsBase = base1;
    tiling->gemm1NUnitsLast = last1;
    tiling->gemm1SingleNMax = gemm1SingleNMax;
    tiling->gemm1SingleM = pipeTileRows;
    tiling->gemm1UsedCores = gemm1UsedCores;

    tiling->gemm2NUnitsBase = base2;
    tiling->gemm2NUnitsRem = rem2;
    tiling->gemm2SingleNMax = gemm2SingleNMax;
    tiling->gemm2SingleM = totalS;
    tiling->gemm2UsedCores = gemm2UsedCores;

    tiling->sysWorkspaceSize = sysWorkspaceSize;
    tiling->rOffset = 0ULL;
    tiling->cOffset = cOffset;
    tiling->userWorkspaceSize = userWorkspaceSize;

    // ---- 11. 取证日志 ----
    OP_LOGI(context,
            "tiling: S=%u paddedM=%u padRows=%u aicNum=%u aivNum=%u ubSize=%lu usableUB=%lu "
            "sysWs=%u userWs=%lu totalWs=%lu",
            totalS, paddedM, padRows, aicNum, aivNum, static_cast<unsigned long>(ubSize),
            static_cast<unsigned long>(usableUB), sysWorkspaceSize, static_cast<unsigned long>(userWorkspaceSize),
            static_cast<unsigned long>(userWorkspaceSize + static_cast<uint64_t>(sysWorkspaceSize)));
    OP_LOGI(context,
            "RoPE: tileS=%u pipeTileNum=%u pipeTileRows=%u sTileNum=%u ropeTaskNum=%u "
            "ropeTasksPerCore=%u aivUsedCores=%u ropeUB=%lu (fixedUB=%lu rowUB=%lu)",
            tileS, pipeTileNum, pipeTileRows, sTileNum, ropeTaskNum, ropeTasksPerCore, aivUsedCores,
            static_cast<unsigned long>(ropeUB), static_cast<unsigned long>(ROPE_FIXED_UB),
            static_cast<unsigned long>(ROPE_ROW_UB));
    OP_LOGI(context,
            "GEMM1: nSplit=%u base=%u last=%u singleNMax=%u singleM=%u usedCores=%u",
            nSplit, base1, last1, gemm1SingleNMax, pipeTileRows, gemm1UsedCores);
    OP_LOGI(context,
            "GEMM2: base=%u rem=%u singleNMax=%u singleM=%u usedCores=%u",
            base2, rem2, gemm2SingleNMax, totalS, gemm2UsedCores);

    // ---- 12. TilingKey / batchmode / blockDim ----
    // 大 S 使用独立模板（更大的 M tile），小 S 保持原模板。
    const uint32_t schMode = (totalS > 128U) ? 1U : 0U;
    ASCENDC_TPL_SEL_PARAM(context, schMode);
    const ge::graphStatus scheduleModeRc = context->SetScheduleMode(1);
    if (scheduleModeRc != ge::GRAPH_SUCCESS) {
        return ReportTilingFailure(context, "SetScheduleMode(1) failed, rc=" +
                                                 std::to_string(static_cast<unsigned int>(scheduleModeRc)));
    }
    const ge::graphStatus blockDimRc = context->SetBlockDim(aicNum);
    if (blockDimRc != ge::GRAPH_SUCCESS) {
        return ReportTilingFailure(context, "SetBlockDim(" + std::to_string(aicNum) + ") failed, rc=" +
                                                 std::to_string(static_cast<unsigned int>(blockDimRc)));
    }
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus TilingParseForMlaEpilogue([[maybe_unused]] gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

struct MlaEpilogueCompileInfo {}; // 必须定义，入图场景依赖

IMPL_OP_OPTILING(FiMlaEpilogue)
    .Tiling(TilingMlaEpilogue)
    .TilingParse<MlaEpilogueCompileInfo>(TilingParseForMlaEpilogue);

} // namespace optiling
