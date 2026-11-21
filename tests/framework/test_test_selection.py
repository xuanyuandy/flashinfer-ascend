"""Hardware scoping must retain host tests and report unexecuted targets."""

from types import SimpleNamespace

import pytest

from tests.support.pytest_plugin import _soc_scope, pytest_collection_finish, pytest_collection_modifyitems


class Item:
    nodeid = "example"

    def __init__(self, *soc):
        self.marker = pytest.mark.requires_soc(*soc).mark if soc else None

    def iter_markers(self, name):
        return (self.marker,) if self.marker is not None else ()


def configuration(target):
    rejected = []
    return SimpleNamespace(getoption=lambda _: target,
                           hook=SimpleNamespace(pytest_deselected=lambda items: rejected.extend(items))), rejected


def test_all_preserves_the_complete_cross_chip_collection():
    config, rejected = configuration("all")
    items = [Item(), Item("Ascend910B"), Item("Ascend950")]
    original = items.copy()
    pytest_collection_modifyitems(config, items)
    assert items == original and rejected == []


@pytest.mark.parametrize("target", ["Ascend910B", "Ascend910C", "Ascend950"])
def test_explicit_target_retains_host_and_exact_matching_requirements(target):
    config, rejected = configuration(target)
    host, shared, a5 = Item(), Item("Ascend910B", "Ascend910C"), Item("Ascend950")
    items = [host, shared, a5]
    pytest_collection_modifyitems(config, items)
    assert items == ([host, a5] if target == "Ascend950" else [host, shared])
    assert rejected == ([shared] if target == "Ascend950" else [a5])
    assert config._flashinfer_soc_scope == (target, 1)


def test_auto_uses_the_visible_chip_without_changing_requirements(monkeypatch):
    import flashinfer.runtime

    monkeypatch.setattr(flashinfer.runtime, "current_hardware", lambda: SimpleNamespace(soc_model="Ascend910B"))
    config, rejected = configuration("auto")
    host, a5 = Item(), Item("Ascend950")
    items = [host, a5]
    pytest_collection_modifyitems(config, items)
    assert items == [host] and rejected == [a5]
    assert config._flashinfer_soc_scope == ("Ascend910B", 1)


def test_auto_missing_device_fails_instead_of_silently_deselecting_everything(monkeypatch):
    import flashinfer.runtime

    monkeypatch.setattr(flashinfer.runtime, "current_hardware", lambda: SimpleNamespace(soc_model="unknown"))
    config, _ = configuration("auto")
    with pytest.raises(pytest.UsageError, match="visible supported NPU"):
        pytest_collection_modifyitems(config, [Item("Ascend950")])


@pytest.mark.parametrize("models", [(), ("typo",), ("Ascend910B", "Ascend95")])
def test_invalid_requirements_fail_instead_of_hiding_tests(models):
    config, _ = configuration("all")
    item = Item()
    item.marker = pytest.mark.requires_soc(*models).mark
    with pytest.raises(pytest.UsageError, match="canonical SoC names"):
        pytest_collection_modifyitems(config, [item])


@pytest.mark.parametrize("alias,target", [
    ("A2", "Ascend910B"), ("Ascend910B4", "Ascend910B"),
    ("a3", "Ascend910C"), ("Ascend910_9362", "Ascend910C"),
    ("A5", "Ascend950"), ("Ascend950DT_9582", "Ascend950"),
])
def test_board_and_device_names_select_the_same_scope(alias, target):
    assert _soc_scope(alias) == target


def test_module_and_parameter_requirements_apply_together():
    config, rejected = configuration("Ascend910B")
    item = Item()
    item.iter_markers = lambda name: (pytest.mark.requires_soc("Ascend910B", "Ascend910C").mark,
                                      pytest.mark.requires_soc("Ascend910C").mark)
    items = [item]
    pytest_collection_modifyitems(config, items)
    assert items == [] and rejected == [item]


def test_cross_target_execution_rejected_even_for_unmarked_device_cases(monkeypatch):
    monkeypatch.setattr("flashinfer.runtime.current_hardware", lambda: SimpleNamespace(soc_model="Ascend950"))
    options = {"--soc": "Ascend910C", "collectonly": False}
    session = SimpleNamespace(config=SimpleNamespace(getoption=options.__getitem__), items=[Item()])
    with pytest.raises(pytest.UsageError, match="local device is Ascend950"):
        pytest_collection_finish(session)
    options["collectonly"] = True
    pytest_collection_finish(session)


@pytest.mark.parametrize("soc", ("Ascend910B", "Ascend910C", "Ascend950"))
def test_backend_test_parameters_match_reviewed_hardware_support(soc):
    from flashinfer.operators.implementation import implementations
    from flashinfer.operators.pytest_support import implementation_param
    from tests.production_inventory import REVIEWED_CAPABILITIES, REVIEWED_BACKENDS, REVIEWED_EXPERIMENTAL

    selected = set()
    for impl in implementations(scope=None):
        param = implementation_param(impl)
        if all(soc in mark.args for mark in param.marks if mark.name == "requires_soc"):
            selected.add((param.id, impl.backend.value))
    reviewed = {key: (REVIEWED_BACKENDS[key], cap) for key, cap in REVIEWED_CAPABILITIES.items()}
    reviewed.update(REVIEWED_EXPERIMENTAL)
    assert selected == {(key, backend) for key, (backend, cap) in reviewed.items()
                        if not cap["soc_models"] or soc in cap["soc_models"]}


def test_production_aot_cases_match_reviewed_ids_and_exact_socs():
    from flashinfer.operators.catalog import cases_module
    from flashinfer.operators.pytest_support import aot_params
    from flashinfer.operators.testing import case_parameters
    from tests.production_inventory import PRODUCTION

    params = aot_params()
    actual = []
    for param in params:
        impl, = param.values
        soc, = next(mark.args for mark in param.marks if mark.name == "requires_soc")
        actual.append((param.id, impl.op, soc))
        assert case_parameters(cases_module(impl.op), param.id)
    assert sorted(actual) == sorted((key, op, soc) for key, op, _, _, soc in PRODUCTION)


def test_two_local_implementations_are_collected_and_executed_separately(monkeypatch):
    import torch
    from flashinfer.operators import implementation, testing
    from flashinfer.operators.pytest_support import aot_params
    from flashinfer.runtime import CompileMode

    op = "test.double"
    def variant(key, soc):
        desc = SimpleNamespace(kernel_id=key, target_soc=soc, compile_mode=CompileMode.AOT)
        return SimpleNamespace(kernel_id=key, op=op, bind=lambda: desc)

    first, second, foreign = (variant("first", "Ascend910B"), variant("second", "Ascend910B"),
                              variant("foreign", "Ascend910C"))
    monkeypatch.setattr(implementation, "implementations", lambda **_: (first, second, foreign))
    monkeypatch.setattr("flashinfer.runtime.current_hardware", lambda: SimpleNamespace(soc_model="Ascend910B"))
    monkeypatch.setattr(testing, "definition", lambda _: SimpleNamespace(op=op, schema="(Tensor x) -> Tensor",
                                                                         output_names=("y",)))
    assert [param.id for param in aot_params(op)] == ["first", "second", "foreign"]
    assert testing.select_aot_implementation(op, "first") is first
    assert testing.select_aot_implementation(op, "second") is second
    for requested in (None, "foreign"):
        with pytest.raises(ValueError, match="select a local --kernel-id from first, second"):
            testing.select_aot_implementation(op, requested)

    configurations = {"first": ({"value": 1},), "second": ({"value": 2},)}
    def sample(torch, torch_npu, *, value):
        return (torch.tensor([value]),), {}, torch.tensor([value * 2])
    cases = SimpleNamespace(parameters_for=configurations.__getitem__, sample=sample, TOLERANCES=((0, 0),))
    monkeypatch.setattr(testing, "cases_module", lambda _: cases)
    calls = []
    def double(x, *, kernel_id, out=None):
        calls.append((kernel_id, x.item(), out is not None))
        return x * 2 if out is None else out.copy_(x * 2)
    monkeypatch.setattr(testing, "public_function", lambda _: double)
    for key in configurations:
        testing.numerical_cases(torch, None, op, kernel_id=key)
    assert calls == [("first", 1, False), ("first", 1, True), ("second", 2, False), ("second", 2, True)]
