"""Host-only failure and ownership checks for the private graph update runtime."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from flashinfer.npu import GraphCapability, GraphReplaySpec, ReplayParameterUpdates, graph_requirements
from flashinfer.runtime import Backend, KernelDescriptor, KernelRegistry, SchemaError
from flashinfer.runtime import _graph_params as runtime


def test_private_runtime_and_eager_hook_need_no_optional_imports_or_probes():
    code = '''
import importlib.abc
import shutil
import subprocess
import sys
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torch_npu', 'triton'}:
            raise AssertionError('optional import: ' + fullname)
def forbidden(*args, **kwargs):
    raise AssertionError('compiler probe')
sys.meta_path.insert(0, BlockOptional())
shutil.which = forbidden
subprocess.Popen = forbidden
import flashinfer
from flashinfer.runtime._graph_params import _ReplayParameterUpdates, _record_call
from flashinfer.npu import graph_requirements, ReplayParameterUpdates
updates = _ReplayParameterUpdates()
assert updates.tasks == ()
assert ReplayParameterUpdates().tasks == ()
capability, params = graph_requirements('cann/attention/single_decode_fia_v1')
assert capability.value == 'needs_param_update' and params == ('actual_seq_lengths_kv',)
assert _record_call(lambda **kw: kw, {'lengths': [17]}, updates=('lengths',)) == {'lengths': [17]}
print('PASS import flashinfer + NPU extension + graph query + eager hook: optional imports=0 compiler probes=0')
'''
    result = subprocess.run([sys.executable, "-S", "-c", code],
                            cwd=Path(__file__).resolve().parents[2],
                            text=True, capture_output=True, check=True)
    print(result.stdout.strip())


class Tensor:
    shape = (8, 64)
    dtype = "float16"
    device = "npu:0"

    def __init__(self):
        self.pointer = 128

    def data_ptr(self):
        return self.pointer

    def stride(self):
        return (64, 1)


@pytest.fixture
def device(monkeypatch):
    log = []

    class Stream:
        def __init__(self, name, device="npu:0"):
            self.name, self.device = name, device

        def wait_stream(self, other):
            log.append(("wait_stream", self, other))

    capture, replay, update = (Stream(name) for name in ("capture", "replay", "update"))

    class Event:
        def wait(self, stream):
            log.append(("wait", self, stream))

        def reset(self, stream):
            log.append(("reset", self, stream))

        def record(self, stream):
            log.append(("record", self, stream))

    api = SimpleNamespace(current=capture, capturing=True, ExternalEvent=Event)
    api.current_stream = lambda: api.current
    api.is_current_stream_capturing = lambda: api.capturing

    @contextmanager
    def stream_context(stream):
        before, api.current = api.current, stream
        try:
            yield
        finally:
            api.current = before

    api.stream = stream_context
    api.graph_task_group_begin = lambda s: log.append(("group_begin", s))

    def end(s):
        handle = object()
        log.append(("group_end", s, handle))
        return handle

    api.graph_task_group_end = end
    api.graph_task_update_begin = lambda s, h: log.append(("update_begin", s, h))
    api.graph_task_update_end = lambda s: log.append(("update_end", s))
    monkeypatch.setattr(runtime, "_npu", lambda: api)
    return SimpleNamespace(api=api, capture=capture, replay=replay, update=update, log=log, Stream=Stream)


def record(device, count=1, operation=None):
    updates = runtime._ReplayParameterUpdates()

    def launch(**args):
        device.log.append(("launch", device.api.current, args))
        return args["out"]

    operation = launch if operation is None else operation
    tensor = Tensor()
    arguments = dict(query=tensor, workspace=Tensor(), out=[Tensor()], lengths=[128])
    with updates.recording():
        for _ in range(count):
            runtime._record_call(operation, arguments, updates=("lengths",))
    device.api.capturing = False
    device.api.current = device.replay
    return updates, arguments


def test_multiple_tasks_keep_named_arguments_and_signal_after_updates(device):
    updates, source = record(device, count=2)
    source["lengths"][0] = 999  # The caller no longer owns the host argument list.
    assert updates.tasks[0].arguments["lengths"] == [128]
    assert updates.tasks[0].arguments["query"] is source["query"]
    assert updates.tasks[0].layout.names == tuple(source)
    device.log.clear()
    updates.update(({"lengths": [17]}, {"lengths": [63]}), stream=device.update)
    assert [item[0] for item in device.log] == [
        "wait_stream", "update_begin", "launch", "update_end", "record",
        "update_begin", "launch", "update_end", "record",
    ]
    assert device.log[0][1:] == (device.update, device.replay)
    launches = [item for item in device.log if item[0] == "launch"]
    assert [item[2]["lengths"] for item in launches] == [[17], [63]]
    assert all(item[1] is device.update for item in launches)
    assert all(item[2]["out"][0] is source["out"][0] for item in launches)
    assert device.api.current is device.replay
    assert updates.tasks[0].arguments["lengths"] == [128]


@pytest.mark.parametrize("bad", [(), ({"lengths": [3]},),
                                 ({"lengths": [3]}, {"other": [3]}),
                                 ({"lengths": [3]}, {"lengths": [3, 4]}),
                                 ({"lengths": [3]}, {"lengths": [True]}),
                                 ({"lengths": [3]}, {"lengths": Tensor()})])
def test_invalid_update_rejects_every_task_before_device_work(device, bad):
    updates, _ = record(device, count=2)
    device.log.clear()
    with pytest.raises(SchemaError):
        updates.update(bad, stream=device.update)
    assert device.log == []
    # A preflight failure leaves the graph usable.
    updates.update(({"lengths": [3]}, {"lengths": [4]}), stream=device.update)


@pytest.mark.parametrize("resource", ["query", "workspace", "out"])
def test_repointed_captured_storage_fails_before_update(device, resource):
    updates, source = record(device)
    tensor = source[resource][0] if resource == "out" else source[resource]
    tensor.pointer += 128
    device.log.clear()
    with pytest.raises(SchemaError, match="tensor address/shape/stride/dtype/device changed"):
        updates.update(({"lengths": [17]},), stream=device.update)
    assert device.log == []


def test_recording_requires_caller_capture_and_cannot_be_reused_or_nested(device):
    updates = runtime._ReplayParameterUpdates()
    with pytest.raises(SchemaError, match="completed recording"):
        updates.update((), stream=device.update)
    device.api.capturing = False
    with pytest.raises(SchemaError, match="caller-owned"):
        with updates.recording():
            pass
    device.api.capturing = True
    with updates.recording():
        with pytest.raises(SchemaError, match="nested or reused"):
            with runtime._ReplayParameterUpdates().recording():
                pass
        runtime._record_call(lambda **kw: None, {"lengths": [128]}, updates=("lengths",))
    with pytest.raises(SchemaError, match="nested or reused"):
        with updates.recording():
            pass


def test_failed_capture_restores_context_and_closes_task_group(device):
    updates = runtime._ReplayParameterUpdates()

    def broken(**kw):
        raise RuntimeError("operator failed")

    with pytest.raises(RuntimeError, match="operator failed"):
        with updates.recording():
            runtime._record_call(broken, {"lengths": [128]}, updates=("lengths",))
    assert device.log[-1][0] == "group_end"
    assert updates.tasks == () and not runtime._is_recording()
    with pytest.raises(SchemaError, match="completed recording"):
        updates.update((), stream=device.update)
    # Ordinary eager calls work without touching the device API.
    device.log.clear()
    assert runtime._record_call(lambda **kw: kw, {"lengths": [9]}, updates=("lengths",)) == {"lengths": [9]}
    assert device.log == []


def test_failed_update_closes_update_region_without_signaling_replay(device):
    def operation(**kw):
        if not device.api.capturing:
            raise RuntimeError("update failed")

    updates, _ = record(device, operation=operation)
    device.log.clear()
    with pytest.raises(RuntimeError, match="update failed"):
        updates.update(({"lengths": [17]},), stream=device.update)
    assert [item[0] for item in device.log] == ["wait_stream", "update_begin", "update_end"]
    assert device.api.current is device.replay
    with pytest.raises(SchemaError, match="completed recording"):
        updates.update(({"lengths": [17]},), stream=device.update)


def test_operator_exception_caught_by_forward_still_invalidates_recording(device):
    updates = runtime._ReplayParameterUpdates()

    def broken(**kw):
        raise RuntimeError("caught inside forward")

    with pytest.raises(SchemaError, match="recording failed"):
        with updates.recording():
            runtime._record_call(lambda **kw: None, {"lengths": [128]}, updates=("lengths",))
            with pytest.raises(RuntimeError, match="caught inside forward"):
                runtime._record_call(broken, {"lengths": [128]}, updates=("lengths",))
    assert updates.tasks == () and not runtime._is_recording()


def test_wrong_device_and_update_during_capture_reject_without_device_work(device):
    updates, _ = record(device)
    device.log.clear()
    with pytest.raises(SchemaError, match="capture device"):
        updates.update(({"lengths": [17]},), stream=device.Stream("other", "npu:1"))
    device.api.capturing = True
    with pytest.raises(SchemaError, match="outside capture"):
        updates.update(({"lengths": [17]},), stream=device.update)
    assert device.log == []


def test_empty_recording_and_capture_stream_switch_fail_closed(device):
    with pytest.raises(SchemaError, match="no updatable"):
        with runtime._ReplayParameterUpdates().recording():
            pass
    with pytest.raises(SchemaError, match="capture stream"):
        with runtime._ReplayParameterUpdates().recording():
            device.api.current = device.update
            runtime._record_call(lambda **kw: None, {"lengths": [128]}, updates=("lengths",))
    assert not runtime._is_recording()
    assert device.log == []


def graph_descriptor(**spec_changes):
    kernel_id = "cann/test/declared_graph"
    spec = GraphReplaySpec(
        kernel_id, ("tests/framework/test_replay_param_runtime.py::test_descriptor_drives_multiple_tasks_and_public_callback",),
        "host test double; not NPU evidence", required_arguments=("workspace", "out"),
        parameter_sources=(("lengths", "kv_seq_lengths"),),
        fixed_arguments=(("layout", "BSND"),),
        tensor_metadata=(("query", (8, 64), ("float16",)),),
        parameter_ranges=(("lengths", 1, 128),),
    )
    return KernelDescriptor(
        kernel_id, op="test.graph", backend=Backend.CANN,
        graph_capability=GraphCapability.NEEDS_PARAM_UPDATE, graph_param_updates=("lengths",),
        graph_replay=replace(spec, **spec_changes),
    )


def graph_arguments():
    return dict(query=Tensor(), workspace=Tensor(), out=[Tensor()], lengths=[128], layout="BSND")


def test_descriptor_drives_multiple_tasks_and_public_callback(device):
    descriptor = graph_descriptor()
    registry = KernelRegistry((descriptor,))
    assert graph_requirements(descriptor.kernel_id, registry=registry) == (
        GraphCapability.NEEDS_PARAM_UPDATE, ("lengths",),
    )
    launches = []
    updater = ReplayParameterUpdates()
    arguments = graph_arguments()
    with updater.recording() as recording:
        assert recording is updater
        for _ in range(2):
            runtime._record_kernel_call(descriptor, lambda **kw: launches.append(kw), arguments)
    assert [t.kernel_id for t in updater.tasks] == [descriptor.kernel_id] * 2
    assert all(t.descriptor is descriptor for t in updater.tasks)
    device.api.capturing, device.api.current = False, device.replay
    device.log.clear()
    updater.before_replay(({"kv_seq_lengths": [17]}, {"kv_seq_lengths": [63]}), stream=device.update)
    assert [kw["lengths"] for kw in launches] == [[128], [128], [17], [63]]
    assert [e[0] for e in device.log] == [
        "wait_stream", "update_begin", "update_end", "record", "update_begin", "update_end", "record",
    ]
    device.log.clear()
    with pytest.raises(SchemaError, match="verified graph range"):
        updater.before_replay(({"kv_seq_lengths": [17]}, {"kv_seq_lengths": [129]}), stream=device.update)
    assert device.log == []  # All tasks preflight before ANY update or event.


@pytest.mark.parametrize("change,match", [
    ({"workspace": None}, "requires argument workspace"),
    ({"layout": "BNSD"}, "requires layout='BSND'"),
    ({"query": None}, "outside the verified graph tensor metadata"),
    ({"lengths": [0]}, "verified graph range"),
])
def test_descriptor_scope_rejects_before_device_nodes(device, change, match):
    updater = ReplayParameterUpdates()
    with pytest.raises(SchemaError, match=match):
        with updater.recording():
            runtime._record_kernel_call(graph_descriptor(), lambda **kw: None,
                                        {**graph_arguments(), **change})
    assert updater.tasks == () and device.log == []


def test_declaration_evidence_is_required_and_cannot_be_copied_to_another_kernel(device):
    from tools.t11_replay_param_audit import audit_declarations

    safe = KernelDescriptor("cann/test/unverified", op="test.graph", graph_capability="capture_safe")
    with pytest.raises(SchemaError, match="no graph replay validation evidence"):
        graph_requirements(safe.kernel_id, registry=KernelRegistry((safe,)))
    with pytest.raises(SchemaError, match="no graph replay validation evidence"):
        audit_declarations((safe,))
    with pytest.raises(SchemaError, match="different kernel"):
        replace(graph_descriptor(), kernel_id="cann/test/another")
    with pytest.raises(SchemaError, match="missing graph validation test"):
        audit_declarations((graph_descriptor(validation_tests=("tests/missing.py::test_graph",)),))
    with pytest.raises(SchemaError, match="not_capturable"):
        with ReplayParameterUpdates().recording():
            runtime._record_kernel_call(replace(safe, graph_capability="not_capturable"),
                                        lambda **kw: None, {})
    assert device.log == []


def test_new_npu_namespace_and_t10_option_are_tracked_as_extensions():
    from flashinfer.parity import ParityEntry, validate_packaged_manifests
    from tools.update_parity_inventory import annotate

    class NoUpstreamExtensions:
        def resolve(self, symbol):
            return None

    manifests = validate_packaged_manifests()
    for manifest in manifests:
        extensions = [e for e in manifest.entries if e.local.startswith("flashinfer.npu.")]
        assert len(extensions) == 8
        for entry in extensions:
            assert entry.api_kind == "npu_extension"
            assert entry.semantic_status == "intentionally_different"
            assert entry.extra_fields["upstream_counterpart"] is None
            assert entry.extra_fields["drop_in"] is False
            assert ParityEntry.from_dict(annotate(entry.to_dict(), NoUpstreamExtensions())) == entry
        for entry in manifest.entries:
            if entry.local in {"flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper",
                              "flashinfer.prefill.BatchPrefillWithPagedKVCacheWrapper",
                              "flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper"}:
                option = entry.extra_fields["parameters"]["use_max_workspace"]
                assert option["api_kind"] == "npu_extension" and option["default"] is False
