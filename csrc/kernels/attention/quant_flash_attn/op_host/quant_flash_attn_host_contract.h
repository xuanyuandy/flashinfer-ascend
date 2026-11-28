/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FI_QFA_HOST_CONTRACT_H_
#define FI_QFA_HOST_CONTRACT_H_

#include <cstdint>

namespace quant_flash_attn_host {
enum Input : uint32_t {
    Q, 
    K, 
    V, 
    Q_DESCALE, 
    K_DESCALE, 
    V_DESCALE, 
    BLOCK_TABLE, 
    P_SCALE, 
    CU_SEQLENS_Q,
    CU_SEQLENS_KV, 
    SEQUSED_Q, 
    SEQUSED_KV, 
    SINKS, 
    ATTN_MASK, 
    METADATA,
};

enum Attr : uint32_t {
    QUANT_MODE, 
    SOFTMAX_SCALE, 
    MASK_MODE, 
    WIN_LEFT, 
    WIN_RIGHT, 
    MAX_SEQLEN_Q,
    MAX_SEQLEN_KV, 
    LAYOUT_Q, 
    LAYOUT_Q_DESCALE, 
    LAYOUT_KV, 
    LAYOUT_OUT,
    RETURN_SOFTMAX_LSE, 
    QUANT_SCHEMA,
};

constexpr int64_t kC4A4 = 5;
constexpr int64_t kNoMask = 0;
} // namespace quant_flash_attn_host

#endif
