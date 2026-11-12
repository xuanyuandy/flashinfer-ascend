"""triton backend: collect installed implementations without loading device code."""

def triton_descriptors():
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend="triton")
