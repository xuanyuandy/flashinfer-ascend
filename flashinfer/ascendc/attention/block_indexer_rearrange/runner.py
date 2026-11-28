"""Admission and the typed ACLNN runner for the block-index expansion."""

from flashinfer.jit.tensor_adapter import package_tensor_runner


def admission(recipe, signature):
    from flashinfer.ascendc._runtime import native_admission

    reasons = list(native_admission(recipe, signature))
    if reasons:
        return tuple(reasons)
    specializations = dict(signature.specializations)
    try:
        block_size = int(specializations["attribute.block_size"])
        block_count = int(signature.shapes[0][2])
        bq = int(signature.shapes[1][0])
        bk = int(signature.shapes[2][0])
        kw = int(signature.shapes[3][2])
    except (KeyError, ValueError, TypeError):
        return ("block_indexer_rearrange requires its complete tensor signature and integer attributes",)
    reasons = []
    if kw != block_count * block_size:
        reasons.append("topk_output last dim must equal blockCount * block_size")
    if bq != bk:
        reasons.append("acc_seq_qlen and acc_seq_kvlen must share the same [B+1] prefix")
    return tuple(reasons)


def make_runner(module, spec, recipe):
    return package_tensor_runner(module, recipe, spec=spec)
