"""JIT cache key 稳定性（§14.1：相同输入必同 key，任一组成变化必不同 key）。"""

from flashinfer.jit import build_cache_key, toolchain_fingerprint
from flashinfer.runtime.hardware import HardwareSignature
from flashinfer.runtime.signature import InputSignature

_HW = HardwareSignature(soc_model="Ascend910B", aicore_count=20)
_SIG = InputSignature(
    op="attention.decode",
    dtypes=("float16",),
    shapes=((1, 8, 128),),
    specializations=(("causal", "True"),),
)


def test_same_inputs_same_key():
    assert build_cache_key(_SIG, _HW, toolchain="t1") == build_cache_key(
        _SIG, _HW, toolchain="t1"
    )


def test_input_signature_change_changes_key():
    base = build_cache_key(_SIG, _HW, toolchain="t1")
    changed_dtype = InputSignature(
        op=_SIG.op, dtypes=("bfloat16",), shapes=_SIG.shapes,
        specializations=_SIG.specializations,
    )
    changed_shape = InputSignature(
        op=_SIG.op, dtypes=_SIG.dtypes, shapes=((2, 8, 128),),
        specializations=_SIG.specializations,
    )
    changed_spec = InputSignature(
        op=_SIG.op, dtypes=_SIG.dtypes, shapes=_SIG.shapes,
        specializations=(("causal", "False"),),
    )
    for sig in (changed_dtype, changed_shape, changed_spec):
        assert build_cache_key(sig, _HW, toolchain="t1") != base


def test_hardware_change_changes_key():
    base = build_cache_key(_SIG, _HW, toolchain="t1")
    other_soc = HardwareSignature(soc_model="Ascend910C", aicore_count=20)
    other_aicore = HardwareSignature(soc_model="Ascend910B", aicore_count=24)
    assert build_cache_key(_SIG, other_soc, toolchain="t1") != base
    assert build_cache_key(_SIG, other_aicore, toolchain="t1") != base


def test_toolchain_change_changes_key():
    base = build_cache_key(_SIG, _HW, toolchain="t1")
    assert build_cache_key(_SIG, _HW, toolchain="t2") != base


def test_default_toolchain_fingerprint_is_stable_on_host():
    # host 无工具链时 fingerprint 各项为 "none"，但两次构建必须一致
    assert toolchain_fingerprint() == toolchain_fingerprint()
    assert build_cache_key(_SIG, _HW) == build_cache_key(_SIG, _HW)
