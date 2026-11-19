// Local diagnostics for the imported host tiler; no runtime dependency.
#pragma once
#include <cstdio>
#define ASCENDC_EXTERN_C
#define OP_LOGD(node, ...) ((void)0)
#define OP_LOGE(node, ...) std::fprintf(stderr, __VA_ARGS__)
#define OPS_REPORT_VECTOR_INNER_ERR(node, ...) std::fprintf(stderr, __VA_ARGS__)
#define OP_CHECK_NULL_WITH_CONTEXT(context, pointer) \
    do { if (!(pointer)) { std::fprintf(stderr, "null tiling input: %s\n", #pointer); return ge::GRAPH_FAILED; } } while (0)
#define OP_CHECK_IF(condition, diagnostic, action) \
    do { if (condition) { diagnostic; action; } } while (0)
#define FI_REGISTER_TILING(op, data) REGISTER_TILING_DATA_CLASS(op, data)
#define FI_IMPL_TILING(op) IMPL_OP_OPTILING(op)
#define FI_IMPL_SHAPE(op) IMPL_OP_INFERSHAPE(op)
