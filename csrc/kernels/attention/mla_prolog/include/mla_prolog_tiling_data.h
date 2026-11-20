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
 * \file mla_prolog_tiling_data.h
 * \brief POD tiling payload shared by host tiling and the device entry.
 *
 * The offsets mirror the standalone MakeWorkspaceLayout. Workspace passed to
 * the entry is the framework workspace (CANN system prefix + user part); the
 * device entry calls GetUserWorkspace before applying these offsets.
 */

#ifndef MLA_PROLOG_TILING_DATA_H
#define MLA_PROLOG_TILING_DATA_H

#include <cstdint>

struct MlaPrologTilingData {
    // Layout version 4 adds the update_kv_cache attribute. The device entry
    // rejects any other version so a stale host/device pair cannot
    // misinterpret the payload.
    uint32_t version;
    uint32_t t;
    uint32_t aicNum;
    uint32_t projectionSplits;
    uint32_t narrowM;
    // Local Q head count in [1,64] (TP shard size); Q-B output channels are
    // headNum * 512 and q_output is [t, headNum, 512].
    uint32_t headNum;
    // 1 (default): scatter the computed KV row into kv_cache (fused behavior).
    // 0: write it to the fresh [t,512] kv output and never read slot_indices
    // or read/write kv_cache. A tiling scalar, not a device-side OpDef
    // attribute, because the kernel ABI forwards only tensors + tiling.
    uint32_t updateKvCache;
    // FP32 RMSNorm epsilon parsed from the rms_norm_eps OpDef attribute;
    // applied to both the Q and KV branches. Always finite and > 0.
    float rmsNormEps;
    uint32_t qaOffset;
    uint32_t qaBytes;
    uint32_t kvOffset;
    uint32_t kvBytes;
    uint32_t kvNormOffset;
    uint32_t kvNormBytes;
    uint32_t qInt8Offset;
    uint32_t qInt8Bytes;
    uint32_t qScaleOffset;
    uint32_t qScaleBytes;
    uint32_t wScaleFpOffset;
    uint32_t wScaleFpBytes;
    uint32_t qmmOffset;
    uint32_t qmmBytes;
    uint32_t totalBytes;
    uint64_t hardwareSyncAddr;
};

#endif // MLA_PROLOG_TILING_DATA_H
