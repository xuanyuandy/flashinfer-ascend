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
 * \file mla_prolog.cpp
 * \brief Device entry for the A+E3 DS4.1 MLA prolog kernel.
 *
 * The original CATLASS DeviceGemm adapter is a host launcher. This entry keeps
 * the same device math (mla_prolog_kernel.h is byte-identical to the
 * standalone Dsv41 namespace body, a frozen implementation detail) and
 * reconstructs the UnifiedMlaPrologKernel Params from the framework tiling
 * payload. Host tiling supplies the workspace offsets and hardware sync
 * address.
 */

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#include "mla_prolog_kernel.h"
#include "mla_prolog_tiling_data.h"
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

// CATLASS Arguments use generic GM_ADDR members. Reconstructing them inside a
// CATLASS_DEVICE helper is the same pattern as FAInferTla in flash_attention:
// the __global__ entry's global pointers are passed as GM_ADDR parameters.
template <class TilingData>
CATLASS_DEVICE void MlaPrologDevice(
    GM_ADDR x, GM_ADDR wQa, GM_ADDR gammaQa, GM_ADDR wQb, GM_ADDR wQbScale,
    GM_ADDR wKv, GM_ADDR gammaKv, GM_ADDR ropeCos, GM_ADDR ropeSin,
    GM_ADDR slotIndices, GM_ADDR kvCache, GM_ADDR qOutput, GM_ADDR qr,
    GM_ADDR kvOutput, GM_ADDR userWorkspace, const __gm__ TilingData *tilingData)
{
    const uint32_t t = tilingData->t;
    // Local Q head count for this TP shard (1..64); the host tiler validated it
    // against w_qb/w_qb_scale shapes. Q-B output channels N = headNum*512.
    const uint32_t headNum = tilingData->headNum;
    // false: the KV branch writes the fresh kv output and never touches
    // kv_cache or slot_indices; true keeps the fused in-place scatter.
    const bool updateKvCache = tilingData->updateKvCache != 0U;
    GM_ADDR qa = userWorkspace + tilingData->qaOffset;
    GM_ADDR kv = userWorkspace + tilingData->kvOffset;
    GM_ADDR kvNorm = userWorkspace + tilingData->kvNormOffset;
    GM_ADDR qInt8 = userWorkspace + tilingData->qInt8Offset;
    GM_ADDR qScale = userWorkspace + tilingData->qScaleOffset;
    GM_ADDR wScaleFp = userWorkspace + tilingData->wScaleFpOffset;
    GM_ADDR qmmWorkspace = userWorkspace + tilingData->qmmOffset;

    // Dsv41:: is the frozen standalone kernel namespace; the public repo name
    // is mla_prolog. The device math and struct layouts are unchanged.
    Dsv41::DualProjectionKernel::Arguments projectionArgs{
        GemmCoord(t, Dsv41::kQa, Dsv41::kHidden),
        GemmCoord(t, Dsv41::kKv, Dsv41::kHidden),
        x, wQa, wKv, qa, kv,
        tilingData->projectionSplits, tilingData->narrowM};
    // rmsNormEps travels in the host tiling payload: the CANN op-package
    // kernel ABI forwards only Tensor arguments plus workspace/tiling, so the
    // OpDef attribute is parsed by host tiling and read here.
    Dsv41::NormQuantKernel::Arguments normArgs{
        qa, kv, kvNorm, gammaQa, gammaKv, qInt8, qScale, wQbScale, wScaleFp,
        t, tilingData->projectionSplits, qr, tilingData->rmsNormEps, headNum};
    Dsv41::RopeScatterKernel::Arguments ropeArgs{
        qOutput, kvNorm, ropeCos, ropeSin, slotIndices, kvCache, t, headNum,
        updateKvCache, kvOutput};

    // QuantMatmul::Arguments stores generic uint8_t*; building its Params
    // directly keeps the __gm__ address space of the device entry and matches
    // the CATLASS Params layout without a host-only conversion.
    using Quant = Dsv41::QuantMatmul;
    const uint32_t m = t, n = headNum * Dsv41::kHeadDim, k = Dsv41::kQa;
    Quant::Params quantParams{};
    quantParams.problemShape = GemmCoord(m, n, k);
    quantParams.ptrA = reinterpret_cast<__gm__ Quant::ElementA *>(qInt8);
    quantParams.layoutA = Quant::LayoutA::template MakeLayout<Quant::ElementA>(m, k);
    quantParams.ptrB = reinterpret_cast<__gm__ Quant::ElementB *>(wQb);
    quantParams.layoutB = Quant::LayoutB::template MakeLayout<Quant::ElementB>(k, n);
    quantParams.ptrScale = reinterpret_cast<__gm__ Quant::ElementScale *>(wScaleFp);
    quantParams.layoutScale = Quant::LayoutScale{n};
    quantParams.ptrPerTokenScale =
        reinterpret_cast<__gm__ Quant::ElementPerTokenScale *>(qScale);
    quantParams.layoutPerTokenScale = Quant::LayoutPerTokenScale{m};
    quantParams.ptrD = reinterpret_cast<__gm__ Quant::ElementD *>(qOutput);
    quantParams.layoutD = Quant::LayoutD::template MakeLayout<Quant::ElementD>(m, n);
    quantParams.ptrWorkspace = qmmWorkspace;

    Dsv41::UnifiedMlaPrologKernel::Params params{
        Dsv41::DualProjectionKernel::ToUnderlyingArguments(projectionArgs, nullptr),
        normArgs, quantParams, ropeArgs};
    Dsv41::UnifiedMlaPrologKernel op;
    op(params);
}

extern "C" __global__ __aicore__ void fi_mla_prolog(
    GM_ADDR x, GM_ADDR wQa, GM_ADDR gammaQa, GM_ADDR wQb, GM_ADDR wQbScale,
    GM_ADDR wKv, GM_ADDR gammaKv, GM_ADDR ropeCos, GM_ADDR ropeSin,
    GM_ADDR slotIndices, GM_ADDR kvCache, GM_ADDR qOutput, GM_ADDR qr,
    GM_ADDR kvOutput, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MlaPrologTilingData);
    if (workspace == nullptr || tiling == nullptr) {
        return;
    }
    AscendC::SetSysWorkspace(workspace);
    auto *data = reinterpret_cast<__gm__ MlaPrologTilingData *>(tiling);
    if (data->version != 4 || data->t < 1 || data->t > Dsv41::kMaxT ||
        data->aicNum == 0 || data->hardwareSyncAddr == 0 ||
        data->headNum < Dsv41::kMinHeads || data->headNum > Dsv41::kMaxHeads ||
        data->updateKvCache > 1U ||
        !(data->rmsNormEps > 0.0F)) {
        return;
    }
    // The kv output is the only KV write target when the fused cache scatter
    // is disabled, so a missing pointer must not silently drop the row.
    if (data->updateKvCache == 0U && kvOutput == nullptr) {
        return;
    }
    AscendC::SetSyncBaseAddr(data->hardwareSyncAddr);
    GM_ADDR userWorkspace = AscendC::GetUserWorkspace(workspace);
    MlaPrologDevice<MlaPrologTilingData>(
        x, wQa, gammaQa, wQb, wQbScale, wKv, gammaKv, ropeCos, ropeSin,
        slotIndices, kvCache, qOutput, qr, kvOutput, userWorkspace, data);
}
