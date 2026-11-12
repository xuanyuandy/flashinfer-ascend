"""Compatibility entry for the opt-in Triton attention implementation."""
from .single_prefill_catlass.runner import TritonFlashAttentionRunner, TRITON_FLASH_ATTENTION_KERNEL_ID


def attention_descriptors():
    from flashinfer.operators.implementation import implementation_descriptors
    return implementation_descriptors(backend="triton", scope="experimental", origin="flashinfer.triton.attention")


def flash_attention_registry(base=None):
    """Opt-in registry for FA experiments; never extend the production singleton."""
    from flashinfer.runtime import Backend, KernelRegistry, get_global_registry

    base = get_global_registry() if base is None else base
    registry = KernelRegistry()
    for descriptor in base.descriptors() + attention_descriptors():
        if descriptor.backend == Backend.CUSTOM:
            registry.register_custom(descriptor)
        elif descriptor.backend == Backend.EXTERNAL:
            registry.register_external(descriptor)
        else:
            registry.register(descriptor)
    registry.freeze()
    return registry
