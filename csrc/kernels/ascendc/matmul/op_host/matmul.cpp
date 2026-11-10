// Shape-driven tiling for the common BF16 ND Matmul contract.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "flashinfer_ascend/matmul_tiling.h"

#define FI_STRING_INNER(x) #x
#define FI_STRING(x) FI_STRING_INNER(x)
#define FI_REGISTER(x) OP_ADD(x)

namespace {
bool Dimension(int64_t dim) { return dim >= 16 && dim <= 256 && dim % 16 == 0; }
bool DenseMatrix(const gert::StorageShape *shape) {
    if (!shape || shape->GetStorageShape().GetDimNum() != 2 ||
        shape->GetOriginShape().GetDimNum() != 2) return false;
    for (size_t i = 0; i < 2; ++i) {
        if (shape->GetStorageShape().GetDim(i) != shape->GetOriginShape().GetDim(i)) return false;
    }
    return true;
}
}

ge::graphStatus FiMatmulTiling(gert::TilingContext *context) {
    auto *a = context->GetInputShape(0);
    auto *b = context->GetInputShape(1);
    auto *out = context->GetOutputShape(0);
    if (!DenseMatrix(a) || !DenseMatrix(b) || !DenseMatrix(out)) return ge::GRAPH_FAILED;
    const auto &as = a->GetStorageShape();
    const auto &bs = b->GetStorageShape();
    int64_t m = as.GetDim(0), k = as.GetDim(1), n = bs.GetDim(1);
    if (!Dimension(m) || !Dimension(n) || !Dimension(k) || bs.GetDim(0) != k ||
        out->GetStorageShape().GetDim(0) != m || out->GetStorageShape().GetDim(1) != n)
        return ge::GRAPH_FAILED;
    auto *data = context->GetTilingData<FiMatmulTilingData>();
    if (!data) return ge::GRAPH_FAILED;
    *data = {2, static_cast<uint32_t>(m), static_cast<uint32_t>(n), static_cast<uint32_t>(k)};
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    uint32_t cores = platform.GetCoreNumAic();
    uint32_t tiles = static_cast<uint32_t>((m / 16) * (n / 16));
    if (!cores) return ge::GRAPH_FAILED;
    context->SetBlockDim(tiles < cores ? tiles : cores);
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}

namespace {
ge::graphStatus Infer(gert::InferShapeContext *context) {
    const auto *a = context->GetInputShape(0);
    const auto *b = context->GetInputShape(1);
    auto *out = context->GetOutputShape(0);
    if (!a || !b || !out || a->GetDimNum() != 2 || b->GetDimNum() != 2 ||
        a->GetDim(1) != b->GetDim(0)) return ge::GRAPH_FAILED;
    out->SetDimNum(2);
    out->SetDim(0, a->GetDim(0));
    out->SetDim(1, b->GetDim(1));
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, ge::DT_BF16);
    return ge::GRAPH_SUCCESS;
}
}

namespace ops {
class FI_PACKAGE_OP_TYPE : public OpDef {
public:
    explicit FI_PACKAGE_OP_TYPE(const char *name) : OpDef(name) {
        Input("a").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        Input("b").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        Output("out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        SetInferShape(Infer).SetInferDataType(InferType);
        AICore().SetTiling(FiMatmulTiling).AddConfig(FI_STRING(FI_COMPUTE_UNIT));
    }
};
FI_REGISTER(FI_PACKAGE_OP_TYPE);
}
