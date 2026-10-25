// Native BF16 GEMM + vector epilogue. No graph capture promise is made.
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "flashinfer_ascend/matmul_bias_silu_tiling.h"

#define FI_STRING_INNER(x) #x
#define FI_STRING(x) FI_STRING_INNER(x)
#define FI_REGISTER(x) OP_ADD(x)

namespace {
bool Dimension(int64_t dim) { return dim >= 16 && dim <= 256 && dim % 16 == 0; }
bool DenseShape(const gert::StorageShape *shape, size_t rank) {
    if (!shape || shape->GetStorageShape().GetDimNum() != rank ||
        shape->GetOriginShape().GetDimNum() != rank) return false;
    for (size_t i = 0; i < rank; ++i) {
        if (shape->GetStorageShape().GetDim(i) != shape->GetOriginShape().GetDim(i)) return false;
    }
    return true;
}
}

// CANN invokes this host entry during GetWorkspaceSize and copies the POD tiling
// payload to a device buffer before launching the mixed AIC/AIV kernel.
ge::graphStatus FiMatmulBiasSiluTiling(gert::TilingContext *context) {
    auto *a = context->GetInputShape(0);
    auto *b = context->GetInputShape(1);
    auto *bias = context->GetInputShape(2);
    auto *out = context->GetOutputShape(0);
    if (!DenseShape(a, 2) || !DenseShape(b, 2) || !DenseShape(bias, 1) || !DenseShape(out, 2))
        return ge::GRAPH_FAILED;
    const auto &as = a->GetStorageShape();
    const auto &bs = b->GetStorageShape();
    const int64_t m = as.GetDim(0), k = as.GetDim(1), n = bs.GetDim(1);
    if (!Dimension(m) || !Dimension(n) || !Dimension(k) || bs.GetDim(0) != k ||
        bias->GetStorageShape().GetDim(0) != n || out->GetStorageShape().GetDim(0) != m ||
        out->GetStorageShape().GetDim(1) != n) return ge::GRAPH_FAILED;
    auto *data = context->GetTilingData<FiMatmulBiasSiluTilingData>();
    if (!data) return ge::GRAPH_FAILED;
    *data = {2, static_cast<uint32_t>(m), static_cast<uint32_t>(n),
             static_cast<uint32_t>(k), 16, 16, 16, static_cast<uint32_t>(4 * m * n)};
    context->SetTilingKey(0);
    context->SetBlockDim(1);
    // GetUserWorkspace skips this CANN-owned prefix (16 MiB on Ascend950).
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() + data->workspace_bytes;
    return ge::GRAPH_SUCCESS;
}

namespace {
ge::graphStatus Infer(gert::InferShapeContext *context) {
    const auto *a = context->GetInputShape(0);
    const auto *b = context->GetInputShape(1);
    const auto *bias = context->GetInputShape(2);
    auto *out = context->GetOutputShape(0);
    if (!a || !b || !bias || !out || a->GetDimNum() != 2 || b->GetDimNum() != 2 ||
        bias->GetDimNum() != 1) return ge::GRAPH_FAILED;
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
        Input("bias").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        Output("out").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        SetInferShape(Infer).SetInferDataType(InferType);
        AICore().SetTiling(FiMatmulBiasSiluTiling).AddConfig(FI_STRING(FI_COMPUTE_UNIT));
    }
};
FI_REGISTER(FI_PACKAGE_OP_TYPE);
}
