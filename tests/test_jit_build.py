"""T18B: real host loading, process/thread races, corruption and error boundaries.

The host compiler below is an external test backend, not an Ascend C emulator.
No NPU or Triton is required. The separate A5 example tests the actual device ABI.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import time

import pytest

from flashinfer.jit import (
    DefaultJitModuleLoader, DiskJitBuildStore, JitBuildStore, JitCacheIndex,
    JitCompilationError, JitCompilationPolicy, JitLoadError, JitSpec, JitSpecRegistry,
    JitSymbolError, MissingJITCacheError, TritonBackend, build_and_load,
    declare_kernel, describe_artifact, verify_jit_cache_record_payload,
)
from flashinfer.runtime import Backend, KernelDescriptor, KernelRegistry, SourceRef


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


class HostCompiler:
    backend_id = "test_host"

    def __init__(self, library, counter):
        self.library = Path(library)
        self.counter = Path(counter)

    def declare(self, descriptor, **kwargs):
        entry = kwargs.pop("entry_point")
        return JitSpec(name="host." + sha(json.dumps(kwargs, sort_keys=True)),
            domain="test", generator_id=self.backend_id, generator_version="1",
            kernel_id=descriptor.kernel_id, source=descriptor.source,
            entry_points=(entry,), **kwargs)

    def compile(self, spec, *, build_directory):
        with self.counter.open("a") as counter:
            counter.write("compiled\n")
        time.sleep(0.1)  # Ensure competing requests overlap the build.
        output = build_directory / spec.output_path
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.library, output)
        (build_directory / "build.log").write_text("test host fixture copied after native compilation\n")


@pytest.fixture
def native_library(tmp_path):
    compiler = shutil.which("gcc")
    if compiler is None:
        pytest.skip("real host-library tests require gcc")
    source = tmp_path / "entry.c"
    source.write_text("int test_entry(void) { return 42; }\n")
    output = tmp_path / "entry.so"
    subprocess.run([compiler, "-shared", "-fPIC", str(source), "-o", str(output)], check=True)
    return output


def unit(tmp_path, library):
    backend = HostCompiler(library, tmp_path / "count")
    descriptor = KernelDescriptor("t18b-fixture/host", "test.host", Backend.EXTERNAL,
        source=SourceRef("c++", "entry.c", sha("int test_entry(void) { return 42; }\n")),
        target_soc="host_test")
    registry = KernelRegistry()
    registry.register_external(descriptor)
    specs = JitSpecRegistry()
    spec = declare_kernel(descriptor.kernel_id, backend=backend, registry=registry,
        spec_registry=specs, output_path="native/kernel.so", entry_point="test_entry",
        environment_fingerprint=sha("test-host-compiler"), specialization_fingerprint=sha("void"))
    store = DiskJitBuildStore(tmp_path / "cache")
    return backend, spec, store, registry, specs


def request_in_process(spec_dict, library, counter, root, ready, start, results):
    try:
        spec = JitSpec.from_dict(spec_dict)
        backend = HostCompiler(library, counter)
        store = DiskJitBuildStore(root)
        ready.put(True)
        if not start.wait(10):
            raise RuntimeError("concurrency test did not start")
        loaded = build_and_load(spec, backend=backend, store=store, cache=JitCacheIndex())
        results.put((loaded.opaque_module.test_entry(), loaded.receipt.artifact_fingerprint))
    except Exception as error:
        results.put(("error", str(error)))


def test_two_processes_compile_once_and_both_load_correct_bytes(tmp_path, native_library):
    backend, spec, store, _, _ = unit(tmp_path, native_library)
    assert isinstance(store, JitBuildStore)
    context = multiprocessing.get_context("spawn")
    ready, results, start = context.Queue(), context.Queue(), context.Event()
    workers = [context.Process(target=request_in_process, args=(
        spec.to_dict(), str(native_library), str(backend.counter), str(tmp_path / "cache"),
        ready, start, results)) for _ in range(2)]
    for worker in workers:
        worker.start()
    try:
        assert all(ready.get(timeout=15) for _ in workers)
        start.set()
        outcomes = [results.get(timeout=15) for _ in workers]
        assert [outcome[0] for outcome in outcomes] == [42, 42], outcomes
        assert outcomes[0][1] == outcomes[1][1]
        assert backend.counter.read_text().splitlines() == ["compiled"]
    finally:
        start.set()
        for worker in workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
                worker.join()
            assert worker.exitcode == 0


def test_two_threads_share_one_build_and_loaded_module(tmp_path, native_library):
    backend, spec, store, _, _ = unit(tmp_path, native_library)
    cache = JitCacheIndex()
    loader = DefaultJitModuleLoader(store)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(build_and_load, spec, backend=backend, store=store,
                               cache=cache, loader=loader) for _ in range(2)]
        first, second = [future.result(timeout=15) for future in futures]
    assert first is second
    assert first.opaque_module.test_entry() == 42
    assert backend.counter.read_text().splitlines() == ["compiled"]
    assert list((store.root / ".staging").iterdir()) == []


def test_status_records_real_path_and_symbol_without_query_io(tmp_path, native_library, monkeypatch):
    backend, spec, store, registry, specs = unit(tmp_path, native_library)
    cache = JitCacheIndex()
    before, = describe_artifact(spec.kernel_id, registry=registry, spec_registry=specs, cache=cache)
    assert not before.is_compiled and before.artifact_path is None
    build_and_load(spec, backend=backend, store=store, cache=cache)
    path = store.artifact_path(cache.lookup(spec))

    def forbidden(*args, **kwargs):
        raise AssertionError("query must not access the filesystem")

    monkeypatch.setattr(Path, "read_bytes", forbidden)
    monkeypatch.setattr(Path, "stat", forbidden)
    after, = describe_artifact(spec.kernel_id, registry=registry, spec_registry=specs, cache=cache)
    assert after.is_compiled is True
    assert after.artifact_path == str(path)
    assert after.entry_point == "test_entry"


@pytest.mark.parametrize("corrupt", ["artifact", "manifest", "auxiliary"])
def test_corruption_rebuilds_atomically_and_retains_old_generation(tmp_path, native_library, corrupt):
    backend, spec, store, _, _ = unit(tmp_path, native_library)
    build_and_load(spec, backend=backend, store=store, cache=JitCacheIndex())
    original = store.lookup(spec)
    old_directory = store.directory(original)
    if corrupt == "artifact":
        replacement = tmp_path / "corrupt"
        replacement.write_bytes(b"truncated")
        replacement.replace(old_directory / spec.output_path)
    elif corrupt == "manifest":
        (old_directory / "manifest.json").write_text("{")
    else:
        (old_directory / "injected.py").write_text("unrecorded file")
    assert store.lookup(spec) is None
    loaded = build_and_load(spec, backend=backend, store=store, cache=JitCacheIndex())
    assert loaded.opaque_module.test_entry() == 42
    assert backend.counter.read_text().splitlines() == ["compiled", "compiled"]
    assert old_directory.is_dir()
    assert store.directory(store.lookup(spec)) != old_directory


def test_no_reader_sees_a_partial_generation_and_lock_inode_survives_publish(tmp_path, native_library):
    backend, spec, store, _, _ = unit(tmp_path, native_library)
    from flashinfer.jit.build import _verify_output

    with store.lock(spec):
        inode = (store.root / ".locks" / (spec.fingerprint + ".lock")).stat().st_ino
        with store.staging(spec) as directory:
            (directory / "native").mkdir()
            (directory / spec.output_path).write_bytes(b"partial")
            assert DiskJitBuildStore(tmp_path / "cache").lookup(spec) is None
            backend.compile(spec, build_directory=directory)
            artifact = _verify_output(spec, directory)
            assert store.lookup(spec) is None
            record = store.publish(spec, directory, artifact)
        assert store.lookup(spec) == record
        assert (store.root / ".locks" / (spec.fingerprint + ".lock")).stat().st_ino == inode


@pytest.mark.parametrize("changes", [
    {"kernel_id": "external/other"}, {"compile_options": ("-O0",)},
    {"source": SourceRef("c++", "entry.c", sha("edited source"))},
    {"specialization_fingerprint": sha("other specialization")},
    {"environment_fingerprint": sha("other compiler")}, {"target_soc": "other_target"},
])
def test_full_spec_identity_drives_disk_cache(tmp_path, native_library, changes, monkeypatch):
    backend, spec, store, _, _ = unit(tmp_path, native_library)
    def forbidden(*args, **kwargs):
        raise AssertionError("incomplete legacy cache key must not be used")
    monkeypatch.setattr("flashinfer.jit.build_cache_key", forbidden)
    build_and_load(spec, backend=backend, store=store, cache=JitCacheIndex())
    changed = replace(spec, **changes)
    assert changed.fingerprint != spec.fingerprint
    assert store.lookup(changed) is None


@pytest.mark.parametrize("policy", [JitCompilationPolicy.CACHE_ONLY, JitCompilationPolicy.DISABLED])
def test_compilation_policy_blocks_miss_but_allows_verified_hit(tmp_path, native_library, policy):
    backend, spec, store, _, _ = unit(tmp_path, native_library)
    with pytest.raises(MissingJITCacheError, match=policy.value):
        build_and_load(spec, backend=backend, store=store, policy=policy, cache=JitCacheIndex())
    assert not backend.counter.exists()
    build_and_load(spec, backend=backend, store=store, cache=JitCacheIndex())
    loaded = build_and_load(spec, backend=backend, store=store, policy=policy, cache=JitCacheIndex())
    assert loaded.opaque_module.test_entry() == 42


def test_failed_build_is_not_published_and_retains_diagnostics(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("unused")
    descriptor = KernelDescriptor("triton/fail", "test", Backend.TRITON,
        source=SourceRef("triton", "kernel.py", sha("unused")), target_soc="Ascend950")
    backend = TritonBackend(source_root=tmp_path, python_executable="/missing/python")
    spec = declare_kernel(descriptor.kernel_id, backend=backend, registry=KernelRegistry((descriptor,)),
        spec_registry=JitSpecRegistry(), output_path="kernel.npubin", entry_point="entry",
        environment_fingerprint=sha("env"), specialization_fingerprint=sha("specialization"),
        compile_options=('signature={"x":"*fp32"}',))
    store = DiskJitBuildStore(tmp_path / "cache")
    with pytest.raises(JitCompilationError) as caught:
        build_and_load(spec, backend=backend, store=store, cache=JitCacheIndex())
    error = caught.value
    assert error.log_path.is_file()
    assert str(error.log_path).startswith(str(store.root / ".logs"))
    assert "command=/missing/python" in str(error) and spec.fingerprint in str(error)
    assert store.lookup(spec) is None
    assert list((store.root / ".staging").iterdir()) == []


def test_nonzero_compiler_return_code_and_stderr_survive_staging_cleanup(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("unused")
    executable = tmp_path / "failed-python"
    executable.write_text("#!/bin/sh\necho 'toolchain diagnostic' >&2\nexit 7\n")
    executable.chmod(0o755)
    descriptor = KernelDescriptor("triton/fail", "test", Backend.TRITON,
        source=SourceRef("triton", "kernel.py", sha("unused")), target_soc="Ascend950")
    backend = TritonBackend(source_root=tmp_path, python_executable=str(executable))
    spec = declare_kernel(descriptor.kernel_id, backend=backend, registry=KernelRegistry((descriptor,)),
        spec_registry=JitSpecRegistry(), output_path="kernel.npubin", entry_point="entry",
        environment_fingerprint=sha("env"), specialization_fingerprint=sha("specialization"),
        compile_options=('signature={"x":"*fp32"}',))
    store = DiskJitBuildStore(tmp_path / "cache")
    with pytest.raises(JitCompilationError, match="toolchain diagnostic") as caught:
        build_and_load(spec, backend=backend, store=store, cache=JitCacheIndex())
    assert caught.value.returncode == 7
    assert "toolchain diagnostic" in caught.value.log_path.read_text()
    assert store.lookup(spec) is None


def test_missing_symbol_is_distinct_from_load_failure(tmp_path, native_library):
    backend, spec, store, _, _ = unit(tmp_path, native_library)
    build_and_load(spec, backend=backend, store=store, cache=JitCacheIndex())
    record = store.lookup(spec)
    verification = verify_jit_cache_record_payload(record, store.artifact_path(record).read_bytes(), verifier_id="test")
    loader = DefaultJitModuleLoader(store)
    with pytest.raises(JitSymbolError, match="missing_entry") as missing:
        loader.load(record, verification, ("test_entry", "missing_entry"))
    assert missing.value.log_path.is_file()
    # A previously verified identity cannot authorize changed bytes at load time.
    path = store.artifact_path(record)
    replacement = tmp_path / "corrupt"
    replacement.write_bytes(b"not ELF")
    replacement.replace(path)
    with pytest.raises(JitLoadError, match="inventory mismatch") as failed:
        loader.load(record, verification, ("test_entry",))
    assert not isinstance(failed.value, JitSymbolError)
    assert failed.value.log_path.is_file()


def test_missing_compiled_symbol_never_publishes(tmp_path, native_library):
    backend, spec, store, _, _ = unit(tmp_path, native_library)
    spec = replace(spec, entry_points=("missing_entry",))
    with pytest.raises(JitSymbolError, match="missing_entry"):
        build_and_load(spec, backend=backend, store=store, cache=JitCacheIndex())
    assert store.lookup(spec) is None


def test_publication_failure_keeps_the_previous_complete_generation(tmp_path, native_library, monkeypatch):
    backend, spec, store, _, _ = unit(tmp_path, native_library)
    build_and_load(spec, backend=backend, store=store, cache=JitCacheIndex())
    original = store.lookup(spec)
    from flashinfer.jit.build import _verify_output

    def failed_replace(*args, **kwargs):
        raise OSError("simulated pointer publication failure")

    with store.lock(spec):
        with store.staging(spec) as directory:
            backend.compile(spec, build_directory=directory)
            artifact = _verify_output(spec, directory)
            monkeypatch.setattr("flashinfer.jit.cache.os.replace", failed_replace)
            with pytest.raises(OSError, match="publication failure"):
                store.publish(spec, directory, artifact)
    assert store.lookup(spec) == original
    assert not list((store.root / spec.fingerprint[:2]).glob(".publish-*"))


def test_foreign_directory_at_a_key_is_quarantined_before_rebuild(tmp_path, native_library):
    backend, spec, store, _, _ = unit(tmp_path, native_library)
    target = store.root / spec.fingerprint[:2] / spec.fingerprint
    target.mkdir(parents=True)
    (target / "manifest.json").write_text("invalid")
    assert store.lookup(spec) is None
    loaded = build_and_load(spec, backend=backend, store=store, cache=JitCacheIndex())
    assert target.is_symlink()
    assert loaded.opaque_module.test_entry() == 42
