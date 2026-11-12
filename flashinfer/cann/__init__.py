"""cann backend: collect installed implementations without loading device code."""

def cann_descriptors():
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend="cann")
