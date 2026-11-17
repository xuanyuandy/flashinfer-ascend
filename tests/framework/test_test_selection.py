"""Hardware scoping must retain host tests and report unexecuted targets."""

from types import SimpleNamespace

import pytest

from tests.support.pytest_plugin import pytest_collection_modifyitems


class Item:
    nodeid = "example"

    def __init__(self, *soc):
        self.marker = pytest.mark.requires_soc(*soc).mark if soc else None

    def get_closest_marker(self, name):
        return self.marker


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
