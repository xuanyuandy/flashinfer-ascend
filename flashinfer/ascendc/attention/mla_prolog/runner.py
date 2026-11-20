"""Runner skeleton for the DS4.1 MLA prolog native contract.

The tensor order is the NativeBinding order: eleven inputs (``kv_cache`` is an
input that the kernel updates in place) followed by the two outputs
``q_output`` and ``qr``. The shared package adapter owns device checks,
layout/contiguity checks, output overlap and the ACLNN call. This runner
preserves the cache aliasing semantics and forwards the ``rms_norm_eps``
ACLNN attribute.
"""

from dataclasses import dataclass

from flashinfer.jit.tensor_adapter import package_tensor_runner


@dataclass(frozen=True)
class MlaPrologRunner:
    native: object

    def __call__(self, x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
                 rope_cos, rope_sin, slot_indices, kv_cache, q_output, qr,
                 *, rms_norm_eps=1.0e-6):
        # kv_cache is deliberately passed as an input: the kernel scatters the
        # computed rows into the caller's storage. q_output and qr are fresh
        # outputs; their order matches NativeBinding and the ACLNN signature.
        return self.native(
            x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
            rope_cos, rope_sin, slot_indices, kv_cache, q_output, qr,
            rms_norm_eps=rms_norm_eps,
        )


def make_runner(module, spec, recipe):
    return MlaPrologRunner(package_tensor_runner(module, recipe, spec=spec))
