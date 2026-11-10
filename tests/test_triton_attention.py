"""Isolated FA declarations and evidence validity; synthetic timings are host UT only."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
from pathlib import Path
import subprocess
import sys

import pytest

from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID
from flashinfer.runtime import (
    Backend, GraphCapability, InputSignature, KernelRegistry,
    SchemaError, get_global_registry, kernel_registry_scope,
)
from flashinfer.triton.attention import (
    TRITON_FLASH_ATTENTION_KERNEL_ID as TRITON_ID,
    TritonFlashAttentionRunner, attention_descriptors, flash_attention_registry,
)
from flashinfer.tuning import query_tuning_record, save_tuning_record
from flashinfer.tuning.benchmark import _bucket
from flashinfer.tuning.store import _matches
from tests.test_tuning import HW, SIG, fixture_record, local_db


def test_opt_in_registry_keeps_exact_default_inventory_and_existing_bindings():
    default = get_global_registry()
    original = tuple(d.kernel_id for d in default.descriptors())
    registry = flash_attention_registry(default)
    assert len(original) == 84
    assert tuple(d.kernel_id for d in registry.descriptors()) == tuple(sorted(original + (TRITON_ID,)))
    assert registry.frozen
    assert registry.get(FLASH_ATTENTION_KERNEL_ID) is default.get(FLASH_ATTENTION_KERNEL_ID)
    assert get_global_registry() is default
    assert tuple(d.kernel_id for d in default.descriptors()) == original


def test_scope_nested_exception_and_thread_isolation():
    default = get_global_registry()
    first, second = flash_attention_registry(default), KernelRegistry()
    second.freeze()
    with kernel_registry_scope(first):
        assert get_global_registry() is first
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(get_global_registry).result() is default
        with pytest.raises(RuntimeError, match="deliberate"):
            with kernel_registry_scope(second):
                assert get_global_registry() is second
                raise RuntimeError("deliberate")
        assert get_global_registry() is first
    assert get_global_registry() is default


def test_scope_rejects_mutable_registry():
    with pytest.raises(ValueError, match="frozen"):
        with kernel_registry_scope(KernelRegistry()):
            pass


def test_opt_in_import_and_registration_do_not_load_optional_providers():
    code = '''
import importlib.abc, sys
class Guard(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'triton', 'acl'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Guard())
from flashinfer.triton.attention import flash_attention_registry
from flashinfer.runtime import get_global_registry, kernel_registry_scope
default = get_global_registry()
with kernel_registry_scope(flash_attention_registry(default)):
    assert len(get_global_registry().descriptors()) == 85
assert len(default.descriptors()) == 84
'''
    result = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_declaration_pins_source_and_preserves_public_contract():
    descriptor = attention_descriptors()[0]
    existing = get_global_registry().get(FLASH_ATTENTION_KERNEL_ID)
    assert descriptor.op == existing.op
    assert descriptor.capability == existing.capability
    assert descriptor.graph_capability == GraphCapability.NOT_CAPTURABLE
    assert descriptor.backend == Backend.TRITON
    path = Path(__file__).resolve().parents[1] / "flashinfer" / descriptor.source.locator
    assert hashlib.sha256(path.read_bytes()).hexdigest() == descriptor.source.digest
    first, second = (descriptor.runner.spec(dtype) for dtype in ("float16", "bfloat16"))
    assert first.fingerprint != second.fingerprint
    assert first.source == second.source == descriptor.source
    assert 'binary={"mode":"mix"}' in first.compile_options
    with pytest.raises(SchemaError, match="float16 or bfloat16"):
        descriptor.runner.spec("float32")


def test_shape_bucket_is_exact_and_dtype_sensitive():
    descriptor = attention_descriptors()[0]
    signature = InputSignature(descriptor.op, dtypes=("float16",) * 4,
                               shapes=((128, 1, 128),) * 4)
    bucket = _bucket(signature, [descriptor], exact=True)
    assert not bucket["shape_ranges"]
    assert _matches(bucket, signature)
    assert not _matches(bucket, replace(signature, shapes=((256, 1, 128),) * 4))
    assert not _matches(bucket, replace(signature, dtypes=("bfloat16",) * 4))


def test_launch_options_participate_in_execution_identity(monkeypatch):
    import flashinfer.triton.attention as attention
    descriptor = attention_descriptors()[0]
    signature = InputSignature(descriptor.op, dtypes=("float16",) * 4)
    first = TritonFlashAttentionRunner(descriptor).tuning_identity(signature, HW)
    monkeypatch.setattr(attention, "_CONSTANTS", dict(D=128, BLOCK_M=16, BLOCK_N=64))
    assert TritonFlashAttentionRunner(descriptor).tuning_identity(signature, HW) != first


def test_execution_identity_change_rejects_old_measurement(local_db, monkeypatch):
    # Existing fixture supplies explicitly synthetic clocks, never device evidence.
    registry, record = fixture_record(monkeypatch)

    class Runner:
        def __init__(self, identity):
            self.identity = identity

        def tuning_identity(self, signature, hardware):
            return self.identity

    original = registry.descriptors()
    changed_id = sorted(record["candidates"])[0]
    record["candidates"][changed_id]["execution_fingerprint"] = "a" * 64
    current = KernelRegistry([replace(d, runner=Runner("a" * 64)) if d.kernel_id == changed_id else d
                              for d in original])
    save_tuning_record(record)
    assert query_tuning_record(SIG, HW, registry=current) == record["result"]["winner"]
    current.get(changed_id).runner.identity = "b" * 64
    assert query_tuning_record(SIG, HW, registry=current) is None


@pytest.mark.parametrize("symbols", [{"entry_mix_aic"}, {"entry_mix_aiv"}, {"entry"}, {"other_mix_aic", "other_mix_aiv"}])
def test_mixed_binary_requires_both_exact_entries(monkeypatch, symbols):
    import flashinfer.jit._triton_abi as abi
    monkeypatch.setattr(abi, "elf_symbols", lambda *args, **kwargs: symbols)
    with pytest.raises(SchemaError, match="missing defined entry"):
        abi.verify_triton_symbols(b"unused", "entry", "mix")


def test_single_binary_guard_does_not_accept_mixed_pair(monkeypatch):
    import flashinfer.jit._triton_abi as abi
    monkeypatch.setattr(abi, "elf_symbols", lambda *args, **kwargs: {"entry_mix_aic", "entry_mix_aiv"})
    with pytest.raises(SchemaError, match="missing defined entry"):
        abi.verify_triton_symbols(b"unused", "entry")
    assert abi.verify_triton_symbols(b"unused", "entry", "mix")


@pytest.mark.parametrize("declaration", [{"mode": "unknown"}, {"mode": "mix", "extra": 1}, {}])
def test_binary_contract_rejects_unknown_or_incomplete_modes(declaration):
    from flashinfer.jit._triton_abi import triton_binary_mode
    with pytest.raises(ValueError, match="exactly mode"):
        triton_binary_mode({"binary": declaration})
