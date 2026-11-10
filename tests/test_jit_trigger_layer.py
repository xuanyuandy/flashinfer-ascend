"""T18A: declarations and observations, without a compiler or loader."""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from flashinfer.jit import (
    AscendCBackend, JitArtifactState, JitBackend, JitBuildStore, JitCacheIndex,
    JitCacheRecord, JitModuleLoader, JitSpec, JitSpecRegistry, TritonBackend,
    declare_kernel, describe_artifact,
)
from flashinfer.runtime import (
    ArtifactFormat, ArtifactKind, ArtifactRef, Backend, CompileMode,
    KernelDescriptor, KernelLaunchABI, KernelRegistry, SchemaError, SourceRef,
    get_global_registry,
)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def descriptor(language="triton", **changes):
    values = dict(
        kernel_id=language + "/test/Ascend910B", op="test.kernel",
        backend=Backend.TRITON if language == "triton" else Backend.ASCENDC,
        compile_mode=CompileMode.JIT, target_soc="Ascend910B",
        source=SourceRef(language, "kernels/test.py" if language == "triton"
                         else "csrc/test.cpp", digest("source")),
    )
    values.update(changes)
    return KernelDescriptor(**values)


def declare(d=None, backend=None, specs=None, **changes):
    d = descriptor() if d is None else d
    args = dict(
        backend=TritonBackend() if backend is None else backend,
        output_path="jit/test/kernel.bin", entry_point="test_entry",
        environment_fingerprint=digest("explicit environment"),
        specialization_fingerprint=digest("explicit specialization"),
        compile_options=("-O3",), registry=KernelRegistry((d,)),
        spec_registry=JitSpecRegistry() if specs is None else specs,
    )
    args.update(changes)
    return declare_kernel(d.kernel_id, **args)


def compiled_artifact(spec=None, **changes):
    values = dict(
        kind=ArtifactKind.FILE, format=ArtifactFormat.SHARED_LIBRARY,
        locator=spec.output_path if spec is not None else "jit/test/kernel.bin",
        digest=digest("fixture payload"), target_soc="Ascend910B",
        build_id="metadata-fixture-not-a-real-build", size_bytes=15,
        entry_point="test_entry",
    )
    values.update(changes)
    return ArtifactRef(**values)


def cache_record(spec, **changes):
    values = dict(
        spec_name=spec.name, spec_fingerprint=spec.fingerprint,
        environment_fingerprint=spec.environment_fingerprint,
        artifact=compiled_artifact(spec), producer_id="metadata-fixture",
        build_metadata_fingerprint=digest("metadata"),
    )
    values.update(changes)
    return JitCacheRecord(**values)


@pytest.mark.parametrize("language,backend", [
    ("triton", TritonBackend()), ("ascendc", AscendCBackend()),
])
def test_backends_share_declaration_entry_and_preserve_registry(language, backend):
    d = descriptor(language)
    registry = KernelRegistry((d,))
    registry.freeze()
    specs = JitSpecRegistry()
    before = (registry.revision, d.to_dict(), d.fingerprint)
    spec = declare(d, backend, specs, registry=registry)
    assert isinstance(backend, JitBackend)
    assert spec.is_compilation_unit
    assert spec.source is d.source
    assert spec.kernel_id == d.kernel_id
    assert spec.target_soc == d.target_soc
    assert spec.output_path == "jit/test/kernel.bin"
    assert spec.entry_points == ("test_entry",)
    assert spec.compile_options == ("-O3",)
    assert spec.source_artifacts == ()
    assert spec.source_materialized
    assert specs.get_spec_status(spec.name).source_materialized
    assert specs.specs_for_kernel(d.kernel_id) == (spec,)
    assert declare(d, backend, specs, registry=registry) is spec
    assert (registry.revision, d.to_dict(), d.fingerprint) == before
    assert registry.get(d.kernel_id) is d
    restored = JitSpec.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert restored == spec
    assert restored.fingerprint == spec.fingerprint


@pytest.mark.parametrize("changes", [
    dict(compile_options=("-O0",)),
    dict(compile_options=("-O3", "-O0")),
    dict(environment_fingerprint=digest("different toolchain")),
    dict(specialization_fingerprint=digest("different shapes")),
    dict(output_path="other/output.bin"), dict(entry_point="other_entry"),
])
def test_unit_identity_covers_build_inputs(changes):
    first = declare()
    changed = declare(**changes)
    assert first.name != changed.name
    assert first.fingerprint != changed.fingerprint


def test_source_and_target_changes_produce_distinct_units():
    d = descriptor()
    first = declare(d)
    changed_source = replace(d, source=replace(d.source, digest=digest("edited source")))
    changed_target = replace(d, target_soc="Ascend950")
    assert declare(changed_source).fingerprint != first.fingerprint
    assert declare(changed_target).fingerprint != first.fingerprint
    assert declare(compile_options=("-O0", "-O3")).fingerprint != declare(
        compile_options=("-O3", "-O0"),
    ).fingerprint


@pytest.mark.parametrize("path", ["", ".", "./kernel.bin", "../kernel.bin",
                                     "/kernel.bin", "a//b", "a/../b", "a\\b",
                                     "kernel\0.bin", "kernel\n.bin", 7])
def test_unit_rejects_invalid_output_paths(path):
    with pytest.raises(SchemaError, match="output_path"):
        declare(output_path=path)


@pytest.mark.parametrize("options", ["-O3", None, (7,), ("",), ("a\0b",), ("a\nb",)])
def test_unit_rejects_invalid_compile_options(options):
    with pytest.raises(SchemaError, match="compile options"):
        declare(compile_options=options)


@pytest.mark.parametrize("entry", ["", "invalid-entry", "two symbols", "run\0"])
def test_unit_rejects_invalid_entry_symbols(entry):
    with pytest.raises(SchemaError, match="entry points"):
        declare(entry_point=entry)


@pytest.mark.parametrize("changes,message", [
    (dict(source=None), "requires kernel_id, source and output_path"),
    (dict(source="kernels/test.py"), "SourceRef"),
    (dict(kernel_id=None), "requires kernel_id, source and output_path"),
    (dict(output_path=None), "requires kernel_id, source and output_path"),
    (dict(entry_points=()), "exactly one"),
    (dict(entry_points=("a", "b")), "exactly one"),
    (dict(entry_points="a"), "list or tuple"),
    (dict(target_soc=None), "SOC identifier"),
    (dict(target_soc=950), "SOC identifier"),
    (dict(target_soc="Ascend 950"), "SOC identifier"),
])
def test_partial_or_invalid_units_cannot_enter_registry(changes, message):
    with pytest.raises(SchemaError, match=message):
        replace(declare(), **changes)


@pytest.mark.parametrize("changes", [
    dict(compile_options="-O3"), dict(entry_points="x"),
    dict(source={"language": "triton"}), dict(unexpected=True),
])
def test_serialized_units_cannot_bypass_declaration_validation(changes):
    value = dict(declare().to_dict(), **changes)
    with pytest.raises(SchemaError):
        JitSpec.from_dict(value)


def test_declaration_needs_source_and_target_and_checks_backend():
    with pytest.raises(SchemaError, match="no SourceRef"):
        declare(replace(descriptor(), source=None))
    with pytest.raises(SchemaError, match="explicit target_soc"):
        declare(replace(descriptor(), target_soc=None))
    assert declare(replace(descriptor(), target_soc=None), target_soc="Ascend950").target_soc == "Ascend950"
    with pytest.raises(SchemaError, match="source and target_soc"):
        declare(target_soc="Ascend950")
    with pytest.raises(SchemaError, match="source language"):
        declare(backend=AscendCBackend())


def test_ascendc_legacy_declaration_requires_a_resolved_recipe_to_compile(tmp_path):
    from flashinfer.jit import JitCompilationError

    backend = AscendCBackend()
    spec = declare(descriptor(backend.backend_id), backend)
    with pytest.raises(JitCompilationError, match="explicitly resolved BuildRecipe") as caught:
        backend.compile(spec, build_directory=tmp_path)
    assert caught.value.log_path == tmp_path / "build.log"
    assert caught.value.log_path.is_file()


def test_triton_rejects_missing_executable_options_with_a_readable_log(tmp_path):
    from flashinfer.jit import JitCompilationError

    backend = TritonBackend()
    with pytest.raises(JitCompilationError, match="signature/constexprs/options") as caught:
        backend.compile(declare(), build_directory=tmp_path)
    assert caught.value.log_path == tmp_path / "build.log"
    assert caught.value.log_path.is_file()
    assert "flashinfer.jit._triton_compile" in str(caught.value)


def test_new_backend_needs_only_the_two_hooks():
    class AlternateCompiler:
        backend_id = "alternative"

        def declare(self, d, **kwargs):
            # Reuse a source generator, while providing a different compiler.
            spec = TritonBackend().declare(d, **kwargs)
            return replace(spec, name="alternative.test", generator_id=self.backend_id)

        def compile(self, spec, *, build_directory):
            raise NotImplementedError("alternative compiler awaits 18B")

    backend = AlternateCompiler()
    assert isinstance(backend, JitBackend)
    spec = declare(backend=backend)
    assert spec.generator_id == "alternative"


@pytest.mark.parametrize("result", [None, "a spec"])
def test_unified_layer_rejects_silent_or_invalid_declaration_results(result):
    backend = Mock(spec=JitBackend)
    backend.backend_id = "triton"
    backend.declare.return_value = result
    specs = JitSpecRegistry()
    with pytest.raises(TypeError, match="must return JitSpec"):
        declare(backend=backend, specs=specs)
    assert specs.specs() == ()


def test_unified_layer_checks_hook_result_before_registration():
    backend = Mock(spec=JitBackend)
    backend.backend_id = "triton"
    backend.declare.return_value = replace(declare(), output_path="different.bin")
    specs = JitSpecRegistry()
    with pytest.raises(SchemaError, match="requested compilation unit"):
        declare(backend=backend, specs=specs)
    assert specs.specs() == ()
    with pytest.raises(TypeError, match="implement JitBackend"):
        declare(backend=object())


def test_source_only_and_unknown_kernel_queries_are_explicit():
    d = descriptor()
    registry = KernelRegistry((d,))
    specs = JitSpecRegistry()
    status, = describe_artifact(d.kernel_id, registry=registry, spec_registry=specs)
    assert status.state == JitArtifactState.SOURCE_ONLY
    assert status.source is d.source
    assert status.target_soc == d.target_soc
    assert status.is_compiled is False
    assert status.entry_point is None
    assert status.expected_artifact_path is None
    unknown, = describe_artifact("missing/kernel", registry=registry, spec_registry=specs)
    assert not unknown.known_kernel
    assert unknown.state == JitArtifactState.UNKNOWN_KERNEL
    assert unknown.is_compiled is None


def test_all_builtin_records_are_queryable_without_inventing_metadata():
    registry = get_global_registry()
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID, FLASH_ATTENTION_KERNEL_ID
    from flashinfer.runtime.registry import _builtin_descriptors

    descriptors = tuple(registry.get(d.kernel_id) for d in _builtin_descriptors())
    assert len(descriptors) == 81
    assert len(registry.descriptors()) == 84
    assert {d.kernel_id for d in registry.descriptors()} - {d.kernel_id for d in descriptors} == {
        MATMUL_BIAS_SILU_KERNEL_ID, FLASH_ATTENTION_KERNEL_ID,
        "ascendc/gemm/matmul/native/op_package/Ascend950",
    }
    before = [(d.to_dict(), d.fingerprint) for d in registry.descriptors()]
    revision = registry.revision
    for d in descriptors:
        status, = describe_artifact(d.kernel_id, registry=registry, spec_registry=JitSpecRegistry())
        assert status.known_kernel
        if d.kernel_id == "triton/gemm/matmul_v1/Ascend950":
            assert status.state == JitArtifactState.SOURCE_ONLY and status.is_compiled is False
            assert status.source is d.source is not None
            assert status.target_soc == d.target_soc == "Ascend950"
            assert status.artifact is None and status.entry_point is None and status.spec_name is None
            continue
        assert status.state == JitArtifactState.UNDECLARED
        assert status.is_compiled is None
        assert status.source is d.source is None
        assert status.artifact is d.artifact is None
        assert status.target_soc is d.target_soc is None
        assert status.entry_point is None
        assert status.spec_name is None
    production = registry.get(MATMUL_BIAS_SILU_KERNEL_ID)
    status, = describe_artifact(production.kernel_id, registry=registry, spec_registry=JitSpecRegistry())
    assert status.known_kernel and status.state == JitArtifactState.SOURCE_ONLY
    assert status.is_compiled is False and status.artifact is None
    assert status.source is production.source is not None
    assert status.target_soc == "Ascend950"
    assert status.entry_point == "aclnnFiMatmulBiasSilu"
    assert status.spec_name is None
    assert [(d.to_dict(), d.fingerprint) for d in registry.descriptors()] == before
    assert registry.revision == revision


def test_query_reports_every_specialization_without_selecting_or_building(monkeypatch):
    d = descriptor()
    specs = JitSpecRegistry()
    first = declare(d, specs=specs)
    second = declare(d, specs=specs, specialization_fingerprint=digest("second variant"),
                     output_path="jit/second/kernel.bin")
    registry = KernelRegistry((d,))
    # All potential execution/file boundaries fail if a query touches them.
    def forbidden(*args, **kwargs):
        raise AssertionError("status query must not build, load or read files")
    monkeypatch.setattr(TritonBackend, "compile", forbidden)
    monkeypatch.setattr(AscendCBackend, "compile", forbidden)
    monkeypatch.setattr(JitModuleLoader, "load", forbidden)
    monkeypatch.setattr(ArtifactRef, "verify_file", forbidden)
    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    monkeypatch.setattr("builtins.open", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    rows = describe_artifact(d.kernel_id, registry=registry, spec_registry=specs)
    assert tuple(row.spec_name for row in rows) == tuple(sorted((first.name, second.name)))
    for row in rows:
        spec = specs.get(row.spec_name)
        assert row.state == JitArtifactState.DECLARED
        assert row.is_compiled is False
        assert row.artifact is None
        assert row.artifact_path is None
        assert row.expected_artifact_path == spec.output_path
        assert row.entry_point == "test_entry"
        assert row.compile_options == spec.compile_options
        assert row.source == d.source
        assert row.target_soc == d.target_soc
        assert json.loads(json.dumps(row.to_dict()))["source"] == d.source.to_dict()
    # The artifact-recorded path has the same read-only boundary.
    rows = describe_artifact(d.kernel_id, registry=registry, spec_registry=specs,
                             cache=JitCacheIndex((cache_record(first),)))
    assert sum(row.is_compiled is True for row in rows) == 1


def test_query_accepts_only_cache_records_bound_to_exact_unit():
    d = descriptor()
    specs = JitSpecRegistry()
    spec = declare(d, specs=specs)
    record = cache_record(spec)
    status, = describe_artifact(d.kernel_id, registry=KernelRegistry((d,)),
                               spec_registry=specs, cache=JitCacheIndex((record,)))
    assert status.state == JitArtifactState.ARTIFACT_RECORDED
    assert status.is_compiled is True
    assert status.artifact == record.artifact
    assert status.artifact_path == spec.output_path
    assert "filesystem and symbols not checked" in status.reason
    stale = replace(record, spec_fingerprint=digest("stale spec"))
    with pytest.raises(SchemaError, match="does not bind"):
        describe_artifact(d.kernel_id, registry=KernelRegistry((d,)),
                          spec_registry=specs, cache=JitCacheIndex((stale,)))


@pytest.mark.parametrize("changes,message", [
    (dict(locator="wrong/output.bin"), "output path and entry point"),
    (dict(entry_point="wrong_entry"), "output path and entry point"),
    (dict(entry_point=None), "output path and entry point"),
    (dict(target_soc="Ascend950"), "does not bind"),
])
def test_cache_cannot_publish_an_unrelated_kernel_artifact(changes, message):
    spec = declare()
    record = cache_record(spec, artifact=compiled_artifact(spec, **changes))
    with pytest.raises(SchemaError, match=message):
        record.validate_spec(spec)


def test_descriptor_artifact_does_not_prove_a_specific_specialization_compiled():
    d = descriptor(artifact=compiled_artifact())
    specs = JitSpecRegistry()
    spec = declare(d, specs=specs)
    registry = KernelRegistry((d,))
    declared, recorded = describe_artifact(d.kernel_id, registry=registry, spec_registry=specs)
    assert declared.spec_name == spec.name
    assert declared.is_compiled is False
    assert recorded.spec_name is None
    assert recorded.artifact is d.artifact
    assert recorded.is_compiled is True
    # With an exact cache binding there is just one mapping.
    rows = describe_artifact(d.kernel_id, registry=registry, spec_registry=specs,
                             cache=JitCacheIndex((cache_record(spec, artifact=d.artifact),)))
    assert len(rows) == 1
    assert rows[0].spec_name == spec.name
    assert rows[0].artifact == d.artifact


def test_descriptor_artifacts_legacy_sources_and_builtin_providers():
    d = descriptor(artifact=compiled_artifact(entry_point=None), launch_abi=KernelLaunchABI(
        "test.v1", "test_entry", ("x", "stream"),
    ))
    status, = describe_artifact(d.kernel_id, registry=KernelRegistry((d,)))
    assert status.is_compiled is True
    assert status.entry_point == d.launch_abi.entry_point
    assert status.artifact_path == d.artifact.locator
    source_artifact = ArtifactRef(
        ArtifactKind.JIT_SOURCE, ArtifactFormat.ASCENDC_SOURCE, "csrc/test.cpp",
        digest("legacy source"), "Ascend910B", "legacy-recipe", 13,
        entry_point="test_entry",
    )
    d = descriptor("ascendc", source=None, artifact=source_artifact)
    status, = describe_artifact(d.kernel_id, registry=KernelRegistry((d,)))
    assert status.state == JitArtifactState.SOURCE_ONLY
    assert status.is_compiled is False
    assert status.artifact_path is None
    assert status.artifact is source_artifact
    builtin = ArtifactRef(ArtifactKind.BUILTIN, ArtifactFormat.ACLNN_BUILTIN,
                          "builtin:aclnnTest", digest("builtin"), "Ascend910B", "provider",
                          entry_point="aclnnTest")
    d = descriptor(backend=Backend.CANN, artifact=builtin, source=None)
    status, = describe_artifact(d.kernel_id, registry=KernelRegistry((d,)))
    assert status.state == JitArtifactState.BUILTIN
    assert status.is_compiled is None
    assert status.artifact_path is None


def test_query_rejects_stale_source_bindings_instead_of_reattributing_a_build():
    d = descriptor()
    specs = JitSpecRegistry()
    declare(d, specs=specs)
    edited = replace(d, source=replace(d.source, digest=digest("new source")))
    with pytest.raises(SchemaError, match="source and target_soc"):
        describe_artifact(edited.kernel_id, registry=KernelRegistry((edited,)), spec_registry=specs)


def test_old_attention_recipes_keep_identity_and_have_no_inferred_kernel_binding():
    from tests.test_checkpoint_030_attention_jit_framework import generic_spec

    spec = generic_spec()
    value = spec.to_dict()
    assert not {"kernel_id", "source", "output_path"} & value.keys()
    assert not spec.is_compilation_unit
    assert JitSpec.from_dict(value) == spec
    assert JitSpecRegistry((spec,)).specs_for_kernel(spec.name) == ()
    with pytest.raises(SchemaError, match="legacy JIT recipe"):
        spec.validate_descriptor(descriptor())


def test_disk_cache_protocol_does_not_install_a_filesystem_implementation(tmp_path):
    class UnimplementedStore(JitBuildStore):
        pass

    store = UnimplementedStore()
    spec = declare()
    assert isinstance(store, JitBuildStore)
    for operation in (
        lambda: store.lookup(spec), lambda: store.lock(spec), lambda: store.staging(spec),
        lambda: store.publish(spec, tmp_path, compiled_artifact(spec)),
    ):
        with pytest.raises(NotImplementedError, match="18B"):
            operation()
    assert list(tmp_path.iterdir()) == []


def test_import_selection_and_declarations_with_no_optional_dependencies_or_compiler():
    root = str(Path(__file__).resolve().parents[1])
    code = r'''
import importlib.abc
import json
import shutil
import subprocess
import sys

attempts = []
compiler_probes = []
class NoCompilerPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'triton', 'torch', 'torch_npu', 'cann'}:
            attempts.append(fullname)
            raise ModuleNotFoundError('optional dependency blocked: ' + fullname)
sys.meta_path.insert(0, NoCompilerPackages())
sys.path.insert(0, ROOT)
original_which = shutil.which
def forbidden_probe(*args, **kwargs):
    compiler_probes.append(str(args))
    raise AssertionError('compiler discovery/subprocess during import or selection')
shutil.which = forbidden_probe
subprocess.Popen = forbidden_probe
import flashinfer
from flashinfer.jit import (AscendCBackend, JitSpecRegistry, TritonBackend,
                           declare_kernel, describe_artifact)
from flashinfer.runtime import (Backend, HardwareSignature, InputSignature,
                               KernelDescriptor, KernelRegistry, SourceRef)
from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID, FLASH_ATTENTION_KERNEL_ID
from flashinfer.runtime.registry import _builtin_descriptors
registry = flashinfer.get_global_registry()
legacy = _builtin_descriptors()
assert len(legacy) == 81 and len(registry.descriptors()) == 84
assert {d.kernel_id for d in registry.descriptors()} - {d.kernel_id for d in legacy} == {
    MATMUL_BIAS_SILU_KERNEL_ID, FLASH_ATTENTION_KERNEL_ID,
    "ascendc/gemm/matmul/native/op_package/Ascend950",
}
selected = []
for d in legacy:
    cap = d.capability
    signature = InputSignature(d.op, dtypes=(sorted(cap.dtypes)[0] if cap.dtypes else 'float16',),
        layouts=(sorted(cap.layouts)[0] if cap.layouts else '',), shapes=((4, 128),),
        specializations=tuple((k, sorted(v)[0]) for k, v in cap.specializations if v))
    soc = sorted(cap.soc_models)[0] if cap.soc_models else 'Ascend910B'
    if d.op == 'gemm.matmul':
        signature = InputSignature(d.op, dtypes=('bfloat16',) * 3,
            shapes=((16, 16),) * 3, physical_layouts=('ND',) * 3,
            specializations=(('tensor_contract', 'supported'),))
    selected.append(flashinfer.dispatch(signature, HardwareSignature(soc, 32),
                                       allow_reference=True).kernel_id)
    expected_state = 'source_only' if d.kernel_id == 'triton/gemm/matmul_v1/Ascend950' else 'undeclared'
    assert describe_artifact(d.kernel_id)[0].state.value == expected_state
production = registry.get(MATMUL_BIAS_SILU_KERNEL_ID)
production_signature = InputSignature(
    production.op, dtypes=('bfloat16',) * 4,
    shapes=((16, 16), (16, 16), (16,), (16, 16)),
    physical_layouts=('ND',) * 4, specializations=(('tensor_contract', 'supported'),))
production_decision = flashinfer.dispatch(production_signature, HardwareSignature('Ascend950', 32))
assert production_decision.kernel_id == MATMUL_BIAS_SILU_KERNEL_ID
assert production.runner.provenance is None
assert describe_artifact(production.kernel_id)[0].state.value == 'source_only'
selected.append(production_decision.kernel_id)
for language, implementation in [('triton', TritonBackend()), ('ascendc', AscendCBackend())]:
    d = KernelDescriptor(language + '/test', 'test', Backend(language),
        source=SourceRef(language, 'kernels/test.src', 'a' * 64), target_soc='Ascend910B')
    specs = JitSpecRegistry()
    spec = declare_kernel(d.kernel_id, backend=implementation, registry=KernelRegistry((d,)),
        spec_registry=specs, output_path='jit/kernel.bin', entry_point='test_entry',
        environment_fingerprint='b' * 64, specialization_fingerprint='c' * 64)
    assert spec.source is d.source
assert attempts == [], attempts
assert compiler_probes == [], compiler_probes
shutil.which = original_which
assert all(shutil.which(name) is None for name in ('bisheng', 'ccec', 'ascendc'))
print(json.dumps({'records': len(selected), 'legacy_records': len(legacy),
                  'selected_production': production_decision.kernel_id,
                  'optional_import_attempts': attempts,
                  'compiler_probe_attempts': compiler_probes,
                  'compiler_paths': {n: shutil.which(n) for n in ('bisheng', 'ccec', 'ascendc')},
                  'selected_gemma': selected[[d.kernel_id for d in legacy].index(
                      'triton/norm/gemma_fused_add_rmsnorm_v1')]}))
'''.replace("ROOT", repr(root))
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("FLASHINFER_NPU_", "ASCEND_", "CANN_"))}
    env["PATH"] = ""
    completed = subprocess.run([sys.executable, "-S", "-c", code], check=True,
                               capture_output=True, text=True, env=env)
    print(completed.stdout.strip())
    assert json.loads(completed.stdout) == {
        "records": 82, "legacy_records": 81, "optional_import_attempts": [],
        "selected_production": "ascendc/t2/matmul_bias_silu/native/op_package/Ascend950",
        "compiler_probe_attempts": [],
        "compiler_paths": {"bisheng": None, "ccec": None, "ascendc": None},
        "selected_gemma": "triton/norm/gemma_fused_add_rmsnorm_v1",
    }


def test_new_interfaces_are_tracked_in_reproducible_parity_ledger():
    from flashinfer.parity import ParityEntry, load_packaged_manifest
    from tools.update_parity_inventory import annotate

    class NoUpstreamExtension:
        def resolve(self, symbol):
            return None

    entries = {entry.local: entry for entry in load_packaged_manifest("all").entries}
    for module, names in (
        ("backends", ("JitBackend", "TritonBackend", "AscendCBackend", "declare_kernel")),
        ("cache", ("JitBuildStore",)),
        ("status", ("JitArtifactState", "JitArtifactStatus", "describe_artifact")),
    ):
        for name in names:
            entry = entries["flashinfer.jit." + module + "." + name]
            assert entry.api_kind == "npu_extension"
            assert entry.implementation_status == ("functional" if name in {"TritonBackend", "AscendCBackend"} else "framework")
            assert ParityEntry.from_dict(annotate(entry.to_dict(), NoUpstreamExtension())) == entry
    for symbol in (
        "build.build_and_load", "cache.DiskJitBuildStore", "cache.jit_cache_index",
        "loading.DefaultJitModuleLoader", "errors.JitBuildError", "errors.JitCompilationError",
        "errors.JitLoadError", "errors.JitSymbolError",
    ):
        entry = entries["flashinfer.jit." + symbol]
        assert entry.api_kind == "npu_extension"
        assert ParityEntry.from_dict(annotate(entry.to_dict(), NoUpstreamExtension())) == entry
    for name in ("JitSpec", "JitSpecRegistry", "gen_jit_spec"):
        entry = entries["flashinfer.jit.core." + name]
        assert entry.semantic_status == "intentionally_different"
        assert entry.upstream_source["path"] == "flashinfer/jit/core.py"
        assert "18B" in entry.reason
