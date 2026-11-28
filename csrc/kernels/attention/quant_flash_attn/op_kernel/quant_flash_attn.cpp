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
 * \file quant_flash_attn.cpp
 * \brief
 */

#if (__CCE_AICORE__ != 310)
#error "C4A4 QuantFlashAttn requires arch35 (Ascend950)"
#else

#include "kernel_operator.h"
#include "kernel_operator_list_tensor_intf.h"
#include "../../common/op_kernel/util.h"
#include "../../common/op_kernel/vector_common.h"
#include "../../common/op_kernel/arch35/flash_attention_score_common_regbase_arch35.h"
#include "arch35/quant_flash_attn_kernel_mxfp4.h"
#include "arch35/quant_flash_attn_template_tiling_key.h"

using namespace AscendC;
using namespace optiling;

// quant flash attn fullquant mxfp4
template <uint8_t inOutLayoutType, uint16_t config, uint8_t quantMode, bool hasAttenMask, uint8_t KvLayoutType,
          bool isFd, bool enableQf322F32Pre = false,
          BaseApi::QuantSchema QUANT_SCHEMA = BaseApi::QuantSchema::CX_DELTA_QUANT>
__aicore__ inline void quant_flash_attn_mxfp4(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dequantScaleQuery,
    __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* blockTable,
    __gm__ uint8_t* pScale, __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* sequsedQ,
    __gm__ uint8_t* sequsedKv, __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata,
    __gm__ uint8_t* attnOut,
    __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, const QuantFlashAttnTilingData& tilingData)
{
    using INPUT_T = fp4x2_e2m1_t;
    using OUT_T = bfloat16_t;

    fa_base_matmul::ResetIdCounter();

    constexpr LayOutTypeEnum inputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][0]);
    constexpr LayOutTypeEnum outputLayoutType = static_cast<LayOutTypeEnum>(InOutLayoutTypeValue[inOutLayoutType][1]);

    constexpr S1TemplateType s1TemplateType = static_cast<S1TemplateType>(ConfigValue[config].s1);
    constexpr S2TemplateType s2TemplateType = static_cast<S2TemplateType>(ConfigValue[config].s2);
    constexpr DTemplateType dTemplateType = static_cast<DTemplateType>(
        (static_cast<std::underlying_type_t<inferDTemplateType>>(ConfigValue[config].d) + 63) >> 6 << 6);
    constexpr DTemplateType dVTemplateType = static_cast<DTemplateType>(
        (static_cast<std::underlying_type_t<inferDTemplateType>>(ConfigValue[config].dv) + 63) >> 6 << 6);
    constexpr bool isDAligned = ConfigValue[config].d != inferDTemplateType::NotAligned &&
                                static_cast<int32_t>(ConfigValue[config].d) % 64 == 0;

    constexpr bool useDn = (quantMode == QFA_MXFP4_FP32_PREFILL);

    using CubeBlock =
        BaseApi::QuantFlashAttnBlockCubeMxfp4<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                              dTemplateType, dVTemplateType, KvLayoutType, useDn, isDAligned,
                                              enableQf322F32Pre>;
    using VecFaBlock =
        BaseApi::QuantFlashAttnBlockVecMxfp4<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType, s1TemplateType,
                                             s2TemplateType, dTemplateType, dVTemplateType, hasAttenMask, KvLayoutType,
                                             isFd, useDn, isDAligned, QUANT_SCHEMA, enableQf322F32Pre>;
    using VecFdBlock =
        BaseApi::QuantFlashAttnBlockVecFlashDecode<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                   s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                   hasAttenMask, KvLayoutType, useDn>;

    using CubeBlockDummy =
        BaseApi::QuantFlashAttnBlockCubeMxfp4Dummy<INPUT_T, float, inputLayoutType, s1TemplateType, s2TemplateType,
                                                   dTemplateType, dVTemplateType, KvLayoutType, useDn,
                                                   enableQf322F32Pre>;
    using VecFaBlockDummy =
        BaseApi::QuantFlashAttnBlockVecMxfp4Dummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                  s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                  hasAttenMask, KvLayoutType, isFd, useDn, isDAligned, QUANT_SCHEMA, enableQf322F32Pre>;
    using VecFdBlockDummy =
        BaseApi::QuantFlashAttnBlockVecFlashDecodeDummy<INPUT_T, float, OUT_T, inputLayoutType, outputLayoutType,
                                                        s1TemplateType, s2TemplateType, dTemplateType, dVTemplateType,
                                                        hasAttenMask, KvLayoutType, useDn>;

#ifdef __DAV_C310_CUBE__
    using Kernel = BaseApi::QuantFlashAttnKernelMxfp4<CubeBlock, VecFaBlockDummy, VecFdBlockDummy>;
#else
    using Kernel = BaseApi::QuantFlashAttnKernelMxfp4<CubeBlockDummy, VecFaBlock, VecFdBlock>;
#endif

    TPipe tPipe;
    Kernel op;
    op.Init(query, key, value, sinks, attnMask, cuSeqLensQ, cuSeqLensKv, blockTable, dequantScaleQuery, dequantScaleKey,
            dequantScaleValue, pScale, softmaxLse, attnOut, workspace, metadata, sequsedQ, sequsedKv, tilingData, &tPipe);
    op.Process();
}

template <uint8_t inOutLayoutType, uint16_t config, uint8_t quantMode, bool hasAttenMask, uint8_t KvLayoutType,
          bool isFd>
__global__ __aicore__ void fi_quant_flash_attn(
    __gm__ uint8_t* query, __gm__ uint8_t* key, __gm__ uint8_t* value, __gm__ uint8_t* dequantScaleQuery,
    __gm__ uint8_t* dequantScaleKey, __gm__ uint8_t* dequantScaleValue, __gm__ uint8_t* blockTable,
    __gm__ uint8_t* pScale, __gm__ uint8_t* cuSeqLensQ, __gm__ uint8_t* cuSeqLensKv, __gm__ uint8_t* sequsedQ,
    __gm__ uint8_t* sequsedKv, __gm__ uint8_t* sinks, __gm__ uint8_t* attnMask, __gm__ uint8_t* metadata,
    __gm__ uint8_t* attnOut, __gm__ uint8_t* softmaxLse, __gm__ uint8_t* workspace, __gm__ uint8_t* tiling)
{
    REGISTER_TILING_DEFAULT(QuantFlashAttnTilingData);
    // SK/静态图兼容：tiling 在外层入口统一经宏取一次（动=GM零拷贝/静=栈拷贝），向下传参
    GET_TILING_DATA(tilingData, tiling);
    __gm__ uint8_t* user = GetUserWorkspace(workspace);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    if constexpr (quantMode == QFA_MXFP4_FP32_PREFILL ||
                  quantMode == QFA_MXFP4_FP32_DECODE) {
        // Uniform dispatch before constructing the pipeline; hot loops remain compile-time specialized.
        const auto quantSchema =
            static_cast<BaseApi::QuantSchema>(tilingData.baseTiling.quantFlashAttnBaseParams.quantSchema);
        if (quantSchema == BaseApi::QuantSchema::OCP_QUANT) {
            quant_flash_attn_mxfp4<inOutLayoutType, config, quantMode, hasAttenMask, KvLayoutType, isFd,
                                  false, BaseApi::QuantSchema::OCP_QUANT>(
                query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable, pScale, cuSeqLensQ,
                cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata, attnOut, softmaxLse, user, tilingData);
        } else if (quantSchema == BaseApi::QuantSchema::CX_QUANT) {
            quant_flash_attn_mxfp4<inOutLayoutType, config, quantMode, hasAttenMask, KvLayoutType, isFd,
                                  false, BaseApi::QuantSchema::CX_QUANT>(
                query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable, pScale, cuSeqLensQ,
                cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata, attnOut, softmaxLse, user, tilingData);
        } else {
            quant_flash_attn_mxfp4<inOutLayoutType, config, quantMode, hasAttenMask, KvLayoutType, isFd>(
                query, key, value, dequantScaleQuery, dequantScaleKey, dequantScaleValue, blockTable, pScale, cuSeqLensQ,
                cuSeqLensKv, sequsedQ, sequsedKv, sinks, attnMask, metadata, attnOut, softmaxLse, user, tilingData);
        }
    }
}

#endif
