// Local diagnostics for the imported host tiler; no ops-transformer runtime dependency.
#pragma once
#include <cstdio>
#define OP_LOGD(node, ...) ((void)0)
#define OP_LOGI(node, ...) ((void)0)
#define OP_LOGW(node, ...) std::fprintf(stderr, __VA_ARGS__)
#define OP_LOGE(node, ...) std::fprintf(stderr, __VA_ARGS__)
#define OPS_REPORT_VECTOR_INNER_ERR(node, ...) std::fprintf(stderr, __VA_ARGS__)
#define OP_CHECK_NULL_WITH_CONTEXT(context, pointer) \
    do { if (!(pointer)) { std::fprintf(stderr, "null tiling input: %s\n", #pointer); return ge::GRAPH_FAILED; } } while (0)
#define OP_CHECK_IF(condition, diagnostic, action) \
    do { if (condition) { diagnostic; action; } } while (0)
