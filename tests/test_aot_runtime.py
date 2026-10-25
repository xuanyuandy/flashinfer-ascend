"""Generic lazy binding contracts complementary to the real NPU scenarios."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from types import SimpleNamespace

import pytest

from flashinfer.jit import build as build_module
from flashinfer.jit import discovery
from flashinfer.jit.cache import JitCacheRecord
from flashinfer.jit.core import JitSpec
from flashinfer.jit.env import JitCompilationPolicy
from flashinfer.jit.op_package import OP_PACKAGE_ADAPTER_HASH
from flashinfer.jit.runtime import bind_aot_kernel
from flashinfer.runtime import (
    ArtifactFormat, ArtifactKind, ArtifactRef, Backend, CompileMode,
    KernelDescriptor, KernelLaunchABI, KernelRegistry,
    SchemaError, SourceRef,
)


@pytest.fixture
def declaration():
    descriptor = KernelDescriptor(
        "ascendc/test/generic/Ascend950", "test.generic", Backend.ASCENDC,
        compile_mode=CompileMode.AOT, target_soc="Ascend950",
        source=SourceRef("ascendc", "operators/generic.cpp", "a" * 64),
        launch_abi=KernelLaunchABI("test.generic.v1", "aclnnGeneric", ("stream",)),
    )
    spec = JitSpec(
        name="test.generic.runtime", domain="testing", generator_id="ascendc",
        generator_version="2", target_soc=descriptor.target_soc,
        environment_fingerprint="b" * 64, specialization_fingerprint="c" * 64,
        input_fingerprints=(("kernel", descriptor.fingerprint),
                            ("output_adapter", OP_PACKAGE_ADAPTER_HASH)),
        entry_points=("aclnnGeneric",), kernel_id=descriptor.kernel_id,
        source=descriptor.source, output_path="op-package.json",
    )
    return descriptor, spec


def forbidden(*args, **kwargs):
    raise AssertionError("unexpected discovery, adapter, or compiler invocation")


def test_registration_preserves_descriptor_identity_without_io(declaration, monkeypatch):
    descriptor, spec = declaration
    with monkeypatch.context() as guarded:
        guarded.setattr(Path, "open", forbidden)
        guarded.setattr(Path, "stat", forbidden)
        guarded.setattr(discovery, "get_op_package_index", forbidden)
        bound = bind_aot_kernel(descriptor, spec=spec, runner_factory=forbidden,
                                jit_backend_factory=forbidden)
        registry = KernelRegistry((bound,))
        assert registry.get(descriptor.kernel_id) is bound
        assert bound.fingerprint == descriptor.fingerprint
        assert bound.to_dict() == descriptor.to_dict()
        assert bound.runner.provenance is None
        assert descriptor.runner is None


def test_binding_rejects_spec_for_different_abi(declaration):
    descriptor, spec = declaration
    different = replace(descriptor, launch_abi=replace(
        descriptor.launch_abi, abi_name="test.generic.v2"))
    with pytest.raises(SchemaError, match="descriptor fingerprint, including its ABI"):
        bind_aot_kernel(different, spec=spec, runner_factory=forbidden)


@pytest.mark.parametrize("policy", [JitCompilationPolicy.DISABLED, JitCompilationPolicy.CACHE_ONLY])
def test_missing_without_compile_permission_never_calls_backend(
    declaration, monkeypatch, tmp_path, policy,
):
    descriptor, spec = declaration
    rejection = discovery.AotIncompatibleCandidate(tmp_path / "old", ("target_soc mismatch",))

    def find(*args, **kwargs):
        raise discovery.AotMissingError(spec, spec.target_soc, (tmp_path,), (rejection,))

    monkeypatch.setattr(build_module, "build_and_load", forbidden)
    runner = bind_aot_kernel(
        descriptor, spec=spec, runner_factory=forbidden, jit_backend_factory=forbidden,
        index=SimpleNamespace(find=find), policy=policy,
    ).runner
    with pytest.raises(discovery.AotMissingError, match="no matching AOT op-package") as failed:
        runner(7)
    assert "compilation policy=" + policy.value in str(failed.value)
    assert "found 1 candidate(s), all incompatible" in str(failed.value)
    assert "target_soc mismatch" in str(failed.value)
    assert failed.value.rejections == (rejection,)
    assert isinstance(failed.value.__cause__, discovery.AotMissingError)
    assert runner.provenance is None


def test_missing_obeys_global_jit_disable(declaration, monkeypatch, tmp_path):
    descriptor, spec = declaration
    monkeypatch.setenv("FLASHINFER_NPU_JIT_DISABLE", "1")

    def find(*args, **kwargs):
        raise discovery.AotMissingError(spec, spec.target_soc, (tmp_path,))

    runner = bind_aot_kernel(
        descriptor, spec=spec, runner_factory=forbidden, jit_backend_factory=forbidden,
        index=SimpleNamespace(find=find),
    ).runner
    with pytest.raises(discovery.AotMissingError, match="compilation policy=disabled"):
        runner(7)


@pytest.mark.parametrize("phase", ["corrupt", "load", "launch"])
def test_corruption_load_and_cann_launch_errors_never_fall_back(
    declaration, monkeypatch, tmp_path, phase,
):
    descriptor, spec = declaration
    original = RuntimeError("specific CANN or digest failure")
    error_type = (discovery.AotCorruptError if phase == "corrupt"
                  else discovery.AotLoadError)

    def fail():
        if phase == "launch":
            raise original
        raise error_type(tmp_path, original) from original

    def find(*args, **kwargs):
        if phase == "corrupt":
            fail()
        return SimpleNamespace(directory=tmp_path, load=load)

    def load():
        if phase == "load":
            fail()
        return object()

    monkeypatch.setattr(build_module, "build_and_load", forbidden)
    runner = bind_aot_kernel(
        descriptor, spec=spec, runner_factory=lambda module, expected: fail,
        jit_backend_factory=forbidden, index=SimpleNamespace(find=find),
        policy=JitCompilationPolicy.ENABLED,
    ).runner
    with pytest.raises(RuntimeError if phase == "launch" else error_type) as failed:
        runner()
    if phase == "launch":
        assert failed.value is original
        assert runner.provenance.source == "aot"
    else:
        assert failed.value.__cause__ is original
        assert "specific CANN or digest failure" in str(failed.value)
        assert runner.provenance is None


@pytest.mark.parametrize("bad_result", [False, True])
def test_abi_adapter_failure_is_a_load_error_without_fallback(declaration, tmp_path, bad_result):
    descriptor, spec = declaration
    original = ValueError("ABI adapter refused module")

    def adapt(module, expected):
        if bad_result:
            return None
        raise original

    runner = bind_aot_kernel(
        descriptor, spec=spec, runner_factory=adapt, jit_backend_factory=forbidden,
        index=SimpleNamespace(find=lambda *args, **kwargs: SimpleNamespace(
            directory=tmp_path, load=object)), policy=JitCompilationPolicy.ENABLED,
    ).runner
    with pytest.raises(discovery.AotLoadError) as failed:
        runner()
    if bad_result:
        assert isinstance(failed.value.__cause__, TypeError)
    else:
        assert failed.value.__cause__ is original
    assert runner.provenance is None


def test_missing_raised_after_discovery_does_not_authorize_jit(declaration, tmp_path):
    descriptor, spec = declaration
    original = discovery.AotMissingError(spec, spec.target_soc, (tmp_path,))

    def load():
        raise original

    runner = bind_aot_kernel(
        descriptor, spec=spec, runner_factory=forbidden, jit_backend_factory=forbidden,
        index=SimpleNamespace(find=lambda *args, **kwargs: SimpleNamespace(
            directory=tmp_path, load=load)), policy=JitCompilationPolicy.ENABLED,
    ).runner
    with pytest.raises(discovery.AotMissingError) as failed:
        runner()
    assert failed.value is original
    assert runner.provenance is None


def test_only_discovery_missing_allows_enabled_jit(declaration, monkeypatch, tmp_path):
    descriptor, spec = declaration
    events = []
    backend, module = object(), object()
    artifact = ArtifactRef(
        ArtifactKind.FILE, ArtifactFormat.ASCENDC_OP_PACKAGE, spec.output_path,
        "d" * 64, spec.target_soc, spec.fingerprint, 1, entry_point=spec.entry_points[0],
    )
    record = JitCacheRecord(spec.name, spec.fingerprint, spec.environment_fingerprint,
                            artifact, "ascendc", "e" * 64)
    store = SimpleNamespace(lookup=forbidden,
                            directory=lambda actual: tmp_path / "cache-generation")

    def find(expected, **kwargs):
        assert expected is spec
        events.append("find")
        raise discovery.AotMissingError(spec, spec.target_soc, (tmp_path,))

    def backend_factory():
        events.append("backend")
        return backend

    def build(expected, **kwargs):
        assert expected is spec
        cache = kwargs.pop("cache")
        assert kwargs == dict(backend=backend, store=store, policy=JitCompilationPolicy.ENABLED)
        events.append("build")
        cache.publish(record)
        return module

    def adapt(loaded, expected):
        assert loaded is module and expected is spec
        events.append("adapt")
        return lambda value: value + 1

    monkeypatch.setattr(build_module, "build_and_load", build)
    runner = bind_aot_kernel(
        descriptor, spec=spec, runner_factory=adapt, jit_backend_factory=backend_factory,
        index=SimpleNamespace(find=find), store=store, policy=JitCompilationPolicy.ENABLED,
    ).runner
    assert runner(41) == 42
    assert runner(2) == 3
    assert events == ["find", "backend", "build", "adapt"]
    assert runner.provenance.source == "jit"
    assert runner.provenance.directory == tmp_path / "cache-generation"
    assert runner.provenance.spec_fingerprint == spec.fingerprint
    assert "no matching AOT op-package" in runner.provenance.fallback_reason


def test_concurrent_aot_calls_load_and_bind_once(declaration, tmp_path):
    descriptor, spec = declaration
    events = []
    module = object()
    start = Barrier(8)

    def load():
        events.append("load")
        return module

    def find(expected, **kwargs):
        assert expected is spec
        assert kwargs == dict(descriptor=descriptor, target_soc=spec.target_soc)
        events.append("find")
        return SimpleNamespace(directory=tmp_path, load=load)

    def adapt(loaded, expected):
        assert loaded is module and expected is spec
        events.append("adapt")
        return lambda value: value * 2

    runner = bind_aot_kernel(
        descriptor, spec=spec, runner_factory=adapt, jit_backend_factory=forbidden,
        index=SimpleNamespace(find=find), policy=JitCompilationPolicy.DISABLED,
    ).runner

    def execute(value):
        start.wait(timeout=10)
        return runner(value)

    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(execute, range(8))) == [value * 2 for value in range(8)]
    assert events == ["find", "load", "adapt"]
    assert runner.provenance.source == "aot"
    assert runner.provenance.directory == tmp_path
    assert runner.provenance.spec_fingerprint == spec.fingerprint
