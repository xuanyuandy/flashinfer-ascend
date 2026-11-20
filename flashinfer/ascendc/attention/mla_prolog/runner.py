"""Runner skeleton for the DS4.1 MLA prolog native contract.

The tensor order is the NativeBinding order: eleven inputs (``kv_cache`` is an
input that the kernel updates in place when ``update_kv_cache`` is true)
followed by the three outputs ``q_output``, ``qr`` and ``kv``. The shared
package adapter owns device checks, layout/contiguity checks, output overlap
and the ACLNN call. This runner preserves the cache aliasing semantics and
forwards the ``rms_norm_eps``, ``q_head_num`` and ``update_kv_cache`` ACLNN
attributes. ``kv`` is always passed because the ACLNN ABI has a static output
list; the kernel writes it only when ``update_kv_cache`` is false.
"""

from dataclasses import dataclass

from flashinfer.jit.tensor_adapter import package_tensor_runner


@dataclass(frozen=True)
class MlaPrologRunner:
    native: object

    def __call__(self, x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
                 rope_cos, rope_sin, slot_indices, kv_cache, q_output, qr, kv,
                 *, rms_norm_eps=1.0e-6, q_head_num=64,
                 update_kv_cache=True):
        # kv_cache is deliberately passed as an input: in fused mode the
        # kernel scatters the computed rows into the caller's storage. q_output,
        # qr and kv are fresh outputs; their order matches NativeBinding and the
        # ACLNN signature. The kernel writes kv only when update_kv_cache is
        # false and never touches kv_cache in that mode.
        return self.native(
            x, w_qa, gamma_qa, w_qb, w_qb_scale, w_kv, gamma_kv,
            rope_cos, rope_sin, slot_indices, kv_cache, q_output, qr, kv,
            rms_norm_eps=rms_norm_eps, q_head_num=q_head_num,
            update_kv_cache=update_kv_cache,
        )


def make_runner(module, spec, recipe):
    return MlaPrologRunner(package_tensor_runner(module, recipe, spec=spec))
