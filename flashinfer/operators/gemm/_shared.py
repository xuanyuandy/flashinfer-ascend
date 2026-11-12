"""Metadata predicates shared by GEMM implementations; limits are caller-owned."""


def metadata_admission(signature, *, multiple=1, contiguous=False):
    for shape in signature.shapes:
        if any(int(value) % multiple for value in shape):
            return ('implementation requires dimensions multiple_of=' + str(multiple),)
    if contiguous and '0' in dict(signature.specializations).get('tensor_contiguity', ''):
        return ('implementation requires contiguous Tensors',)
    return ()
