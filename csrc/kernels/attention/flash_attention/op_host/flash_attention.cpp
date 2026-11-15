// CANN host adapter for catlass example 49, revision
// 6c0a3242b2effd861ac91aab1999f2226b643614. Dense, batch=1 envelope only.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#if CATLASS_ARCH == 2201
#include "acl/acl_rt.h"
#include "arch22/fai_tiling.h"
#else
#include "arch35/fai_tiling.h"
#endif

#define FI_STRING_INNER(x) #x
#define FI_STRING(x) FI_STRING_INNER(x)
#define FI_REGISTER(x) OP_ADD(x)

namespace {
bool Sequence(int64_t dim) { return dim >= 128 && dim <= 512 && dim % 128 == 0; }
bool DenseShape(const gert::StorageShape *shape) {
    if (!shape || shape->GetStorageShape().GetDimNum() != 3 ||
        shape->GetOriginShape().GetDimNum() != 3) return false;
    for (size_t i = 0; i < 3; ++i) {
        if (shape->GetStorageShape().GetDim(i) != shape->GetOriginShape().GetDim(i)) return false;
    }
    return true;
}
}

// The source example itself rejects isVariedLen > 0. In this batch=1 dense
// specialization the real sequence lengths are exactly the storage shapes.
// No runtime length tensor, data-dependent tiling, or v3 contract is claimed.
ge::graphStatus FiFlashAttentionTiling(gert::TilingContext *context) {
    const auto *q = context->GetInputShape(0);
    const auto *k = context->GetInputShape(1);
    const auto *v = context->GetInputShape(2);
    const auto *out = context->GetOutputShape(0);
    if (!DenseShape(q) || !DenseShape(k) || !DenseShape(v) || !DenseShape(out))
        return ge::GRAPH_FAILED;
    const auto &qs = q->GetStorageShape();
    const auto &ks = k->GetStorageShape();
    const int64_t s = qs.GetDim(0), t = ks.GetDim(0), h = qs.GetDim(1);
    if (!Sequence(s) || !Sequence(t) || h < 1 || h > 8 || qs.GetDim(2) != 128 ||
        ks.GetDim(1) != h || ks.GetDim(2) != 128) return ge::GRAPH_FAILED;
    for (size_t i = 0; i < 3; ++i) {
        if (v->GetStorageShape().GetDim(i) != ks.GetDim(i) ||
            out->GetStorageShape().GetDim(i) != qs.GetDim(i)) return ge::GRAPH_FAILED;
    }
    auto *data = context->GetTilingData<FATilingData>();
    if (!data) return ge::GRAPH_FAILED;
    *data = {};
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    const uint32_t rowTiles = static_cast<uint32_t>(h * s / 128);
    const uint32_t blockDim = std::min(platform.GetCoreNumAic(), rowTiles);
    if (blockDim == 0) return ge::GRAPH_FAILED;
    FAInferTiling::FAInfo info;
#if CATLASS_ARCH == 2201
    info.batch = 1;
    info.numHeads = h;
    info.kvHeads = h;
    info.embeddingSize = 128;
    info.blockSize = 128;
    info.numBlocks = t / 128;
    info.maskType = FAInferTiling::MaskType::NO_MASK;
    int64_t qLength = s, kvLength = t;
    info.qSeqlenList = &qLength;
    info.kvSeqlenList = &kvLength;
    if (FAInferTiling::GetFATilingParam(info, blockDim, *data) != 0)
        return ge::GRAPH_FAILED;
    data->qSeqlen = s;
    void *syncAddress = nullptr;
    if (aclrtGetHardwareSyncAddr(&syncAddress) != ACL_SUCCESS || syncAddress == nullptr)
        return ge::GRAPH_FAILED;
    data->hardwareSyncAddr = reinterpret_cast<uint64_t>(syncAddress);
#else
    info.batchSize = 1;
    info.numOfHeads = h;
    info.numOfKVHeads = h;
    info.seqSize = s;
    info.seqInnerSize = t;
    info.headSize = 128;
    info.scaleValue = static_cast<float>(1.0 / std::sqrt(128.0));
    info.blockSize = 128;
    info.maxBlockNumPerBatch = static_cast<uint32_t>(t / 128);
    // actualSeqLengths/actualSeqLengthsKV remain null: the upstream tiler
    // derives full lengths from seqSize/seqInnerSize, without reading tensors.
    if (blockDim == 0 || blockDim >= MAX_CORE_NUM ||
        FAInferTiling::GetFATilingParam(info, blockDim, *data) != 0)
        return ge::GRAPH_FAILED;
#endif
    context->SetTilingKey(0);
#if CATLASS_ARCH == 2201
    context->SetBlockDim(blockDim);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + data->workSpaceSize;
#else
    context->SetBlockDim(static_cast<uint32_t>(data->multiCoreParamsRegbase.coreNum));
    // Dynamic, nonzero CANN system workspace only. All algorithm scratch is
    // local L1/UB in this source kernel; there is no extra user GM allocation.
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
#endif
    return ge::GRAPH_SUCCESS;
}

namespace {
ge::graphStatus Infer(gert::InferShapeContext *context) {
    const auto *q = context->GetInputShape(0);
    const auto *k = context->GetInputShape(1);
    const auto *v = context->GetInputShape(2);
    auto *out = context->GetOutputShape(0);
    if (!q || !k || !v || !out || q->GetDimNum() != 3 || k->GetDimNum() != 3 ||
        v->GetDimNum() != 3) return ge::GRAPH_FAILED;
    out->SetDimNum(3);
    for (size_t i = 0; i < 3; ++i) out->SetDim(i, q->GetDim(i));
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferType(gert::InferDataTypeContext *context) {
    const auto dtype = context->GetInputDataType(0);
    if ((dtype != ge::DT_FLOAT16 && dtype != ge::DT_BF16) ||
        context->GetInputDataType(1) != dtype || context->GetInputDataType(2) != dtype)
        return ge::GRAPH_FAILED;
    context->SetOutputDataType(0, dtype);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class FI_PACKAGE_OP_TYPE : public OpDef {
public:
    explicit FI_PACKAGE_OP_TYPE(const char *name) : OpDef(name) {
        Input("q").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        Input("k").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        Input("v").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        Output("out").ParamType(REQUIRED).DataType({ge::DT_FLOAT16, ge::DT_BF16})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND});
        SetInferShape(Infer).SetInferDataType(InferType);
        AICore().SetTiling(FiFlashAttentionTiling).AddConfig(FI_STRING(FI_COMPUTE_UNIT));
    }
};
FI_REGISTER(FI_PACKAGE_OP_TYPE);
}
