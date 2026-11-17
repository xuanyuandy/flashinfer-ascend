// Fixed FP32 16x16x16 probe registration. This is deliberately not a general
// matmul operator: reject other shapes before launching its fixed device code.
#include "register/op_def_registry.h"
#include "flashinfer_ascend/probe_tiling.h"
#define FI_STRING_INNER(x) #x
#define FI_STRING(x) FI_STRING_INNER(x)
#define FI_REGISTER(x) OP_ADD(x)

namespace {
bool IsProbeShape(const gert::StorageShape *shape) {
    if (shape == nullptr) return false;
    const auto &storage = shape->GetStorageShape();
    const auto &origin = shape->GetOriginShape();
    return storage.GetDimNum() == 2 && storage.GetDim(0) == 16 && storage.GetDim(1) == 16
        && origin.GetDimNum() == 2 && origin.GetDim(0) == 16 && origin.GetDim(1) == 16;
}
ge::graphStatus Tile(gert::TilingContext *context) {
    if (!IsProbeShape(context->GetInputShape(0)) || !IsProbeShape(context->GetInputShape(1))
        || !IsProbeShape(context->GetOutputShape(0))) return ge::GRAPH_FAILED;
    auto *data = context->GetTilingData<FiMatmulProbeTilingData>();
    if (data == nullptr) return ge::GRAPH_FAILED;
    data->size = 256;
    context->SetTilingKey(0);
    context->SetBlockDim(FI_BLOCK_DIM);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus Infer(gert::InferShapeContext *context) {
    *context->GetOutputShape(0) = *context->GetInputShape(0);
    return ge::GRAPH_SUCCESS;
}
ge::graphStatus InferType(gert::InferDataTypeContext *context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    return ge::GRAPH_SUCCESS;
}
}
namespace ops {
class FI_PACKAGE_OP_TYPE : public OpDef {
public:
    explicit FI_PACKAGE_OP_TYPE(const char *name) : OpDef(name) {
        Input("a").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        Input("b").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        Output("c").ParamType(REQUIRED).DataType({ge::DT_FLOAT}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        SetInferShape(Infer).SetInferDataType(InferType);
        AICore().SetTiling(Tile).AddConfig(FI_STRING(FI_COMPUTE_UNIT));
    }
};
FI_REGISTER(FI_PACKAGE_OP_TYPE);
}
