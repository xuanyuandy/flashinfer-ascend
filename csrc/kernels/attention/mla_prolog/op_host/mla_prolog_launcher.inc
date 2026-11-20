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
 * \file mla_prolog_launcher.inc
 * \brief Standalone host launcher kept for traceability only.
 *
 * The op-package path does not compile this file: host tiling computes the
 * workspace and the ACLNN adapter launches fi_mla_prolog. This copy is
 * the original host launcher from the standalone source (SHA256 443d75c7...).
 * The GetDsv41MlaPrologWorkspaceSize / Dsv41MlaProlog symbol names are frozen
 * standalone ABI implementation details, not the public repo operator name.
 * The required qr output and the rmsNormEps scalar mirror the op-package
 * contract; in the op-package path rmsNormEps reaches the kernel through the
 * host tiling payload because CANN kernel entries receive Tensor arguments
 * plus workspace/tiling only.
 */

#include <cmath>

uint64_t GetDsv41MlaPrologWorkspaceSize(const Dsv41MlaPrologArguments &args)
{
    if (args.t <= 0 || args.t > static_cast<int32_t>(Dsv41::kMaxT) ||
        args.headNum < static_cast<int32_t>(Dsv41::kMinHeads) ||
        args.headNum > static_cast<int32_t>(Dsv41::kMaxHeads)) {
        return 0;
    }
    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr || platform->GetCoreNumAic() == 0) {
        return 0;
    }
    return Dsv41::MakeWorkspaceLayout(
        static_cast<uint32_t>(args.t), platform->GetCoreNumAic(),
        static_cast<uint32_t>(args.headNum)).totalBytes;
}

aclError Dsv41MlaProlog(
    const Dsv41MlaPrologArguments &args,
    aclrtStream stream)
{
    if (args.x == nullptr || args.wQa == nullptr || args.gammaQa == nullptr ||
        args.wQb == nullptr || args.wQbScale == nullptr || args.wKv == nullptr ||
        args.gammaKv == nullptr || args.ropeCos == nullptr || args.ropeSin == nullptr ||
        (args.updateKvCache
             ? args.slotIndices == nullptr || args.kvCache == nullptr
             : args.kvOut == nullptr) ||
        args.qOutput == nullptr ||
        args.qr == nullptr || !std::isfinite(args.rmsNormEps) ||
        !(args.rmsNormEps > 0.0F) ||
        args.headNum < static_cast<int32_t>(Dsv41::kMinHeads) ||
        args.headNum > static_cast<int32_t>(Dsv41::kMaxHeads) ||
        args.workspace == nullptr || stream == nullptr || args.t <= 0 ||
        args.t > static_cast<int32_t>(Dsv41::kMaxT)) {
        return ACL_ERROR_INVALID_PARAM;
    }

    auto *platform = platform_ascendc::PlatformAscendCManager::GetInstance();
    if (platform == nullptr) {
        return ACL_ERROR_INTERNAL_ERROR;
    }
    const uint32_t aicCount = platform->GetCoreNumAic();
    const uint32_t aivCount = platform->GetCoreNumAiv();
    if (aicCount == 0 || aivCount == 0 || aivCount != 2U * aicCount) {
        return ACL_ERROR_INTERNAL_ERROR;
    }
    const uint32_t headNum = static_cast<uint32_t>(args.headNum);
    const auto workspace = Dsv41::MakeWorkspaceLayout(
        static_cast<uint32_t>(args.t), aicCount, headNum);
    if (args.workspaceSize < workspace.totalBytes) {
        return ACL_ERROR_INVALID_PARAM;
    }

    auto *workspaceBase = static_cast<uint8_t *>(args.workspace);
    auto *qa = workspaceBase + workspace.qaOffset;
    auto *kv = workspaceBase + workspace.kvOffset;
    auto *kvNorm = workspaceBase + workspace.kvNormOffset;
    auto *qInt8 = workspaceBase + workspace.qInt8Offset;
    auto *qScale = workspaceBase + workspace.qScaleOffset;
    auto *wScaleFp = workspaceBase + workspace.wScaleFpOffset;
    auto *qmmWorkspace = workspaceBase + workspace.qmmOffset;

    const GemmCoord qaShape(static_cast<uint32_t>(args.t), Dsv41::kQa, Dsv41::kHidden);
    const GemmCoord kvShape(static_cast<uint32_t>(args.t), Dsv41::kKv, Dsv41::kHidden);
    const uint32_t tTokens = static_cast<uint32_t>(args.t);
    uint32_t projectionSplits = 1U;
    uint32_t narrowM = 0U;
    if (tTokens <= Dsv41::kSplitProjectionMaxT) {
        projectionSplits = Dsv41::kProjectionSplitsSmall;
    } else if (tTokens <= Dsv41::kNarrowProjectionMaxT) {
        // T in (128, 192]: BM=64 rebalance of the split=1 tier (see
        // kNarrowProjectionMaxT above). ceil(T/64) = 3 M-tiles x 7 units
        // = 21 tiles <= 24 AIC with the busiest M extent halved to 64 rows.
        narrowM = 1U;
    } else if (tTokens <= Dsv41::kSplitProjectionMidMaxT) {
        projectionSplits = Dsv41::kProjectionSplitsMid;
    }
    Dsv41::DualProjectionKernel::Arguments projectionArgs{
        qaShape,
        kvShape,
        reinterpret_cast<uint8_t *>(const_cast<uint16_t *>(args.x)),
        reinterpret_cast<uint8_t *>(const_cast<uint16_t *>(args.wQa)),
        reinterpret_cast<uint8_t *>(const_cast<uint16_t *>(args.wKv)),
        qa,
        kv,
        projectionSplits,
        narrowM};
    Dsv41::NormQuantKernel::Arguments normArgs{
        qa,
        kv,
        kvNorm,
        reinterpret_cast<uint8_t *>(const_cast<uint16_t *>(args.gammaQa)),
        reinterpret_cast<uint8_t *>(const_cast<uint16_t *>(args.gammaKv)),
        qInt8,
        qScale,
        reinterpret_cast<uint8_t *>(const_cast<uint16_t *>(args.wQbScale)),
        wScaleFp,
        static_cast<uint32_t>(args.t),
        projectionSplits,
        reinterpret_cast<uint8_t *>(args.qr),
        args.rmsNormEps,
        headNum};
    const GemmCoord quantShape(
        static_cast<uint32_t>(args.t), headNum * Dsv41::kHeadDim, Dsv41::kQa);
    Dsv41::QuantMatmul::Arguments quantArgs{
        quantShape,
        aicCount,
        qInt8,
        reinterpret_cast<uint8_t *>(const_cast<int8_t *>(args.wQb)),
        wScaleFp,
        qScale,
        reinterpret_cast<uint8_t *>(args.qOutput)};
    Dsv41::RopeScatterKernel::Arguments ropeArgs{
        reinterpret_cast<uint8_t *>(args.qOutput),
        kvNorm,
        reinterpret_cast<uint8_t *>(const_cast<float *>(args.ropeCos)),
        reinterpret_cast<uint8_t *>(const_cast<float *>(args.ropeSin)),
        reinterpret_cast<uint8_t *>(const_cast<int32_t *>(args.slotIndices)),
        reinterpret_cast<uint8_t *>(args.kvCache),
        static_cast<uint32_t>(args.t),
        headNum,
        args.updateKvCache,
        reinterpret_cast<uint8_t *>(args.kvOut)};
    Dsv41::UnifiedMlaPrologKernel::Arguments unifiedArgs{
        projectionArgs,
        normArgs,
        quantArgs,
        ropeArgs};
    using UnifiedAdapter =
        Gemm::Device::DeviceGemm<Dsv41::UnifiedMlaPrologKernel>;
    if (UnifiedAdapter::CanImplement(unifiedArgs) != Status::kSuccess ||
        UnifiedAdapter::GetWorkspaceSize(unifiedArgs) > workspace.qmmBytes) {
        return ACL_ERROR_INVALID_PARAM;
    }

    uint64_t hardwareSyncAddr = 0;
    if (aclrtGetHardwareSyncAddr(reinterpret_cast<void **>(&hardwareSyncAddr)) != ACL_SUCCESS) {
        return ACL_ERROR_INTERNAL_ERROR;
    }
    UnifiedAdapter unified;
    unified.Initialize(unifiedArgs, qmmWorkspace, stream);
    unified(stream, aicCount, hardwareSyncAddr);
    return ACL_SUCCESS;
}
