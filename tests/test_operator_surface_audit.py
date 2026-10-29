"""Static inventory must distinguish API aliases, lifecycle methods and unknowns."""

from tools.audit_operator_surface import SourceTree, scan


def source(root, path, text):
    destination = root / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)


def fixture_tree(root):
    source(root, "flashinfer/_core.py", """
def direct(x=None): pass
class Batch:
    def plan(self): pass
    def run(self): pass
class Child(Batch): pass
class Segment:
    def run(self): pass
class Legacy:
    def begin_forward(self): pass
    def forward(self): pass
""")
    source(root, "flashinfer/api.py", "from ._core import direct, Batch, Child, Segment, Legacy\n")
    source(root, "flashinfer/__init__.py", "from .api import *\n")


def test_aliases_inheritance_and_methods_keep_one_declaration_identity(tmp_path):
    fixture_tree(tmp_path)
    source(tmp_path, "tests/test_calls.py", """
import flashinfer as fi
from flashinfer.api import direct as operation
def test_direct():
    operation()
    alias = fi.direct
    alias()
def test_batch():
    wrapper = fi.Child()
    wrapper.plan()
    wrapper.run()
    segment = fi.Segment()
    segment.run()
    legacy = fi.Legacy()
    legacy.begin_forward()
    legacy.forward()
""")
    result = scan(tmp_path)
    rows = {row["symbol"]: row for row in result["symbols"]}
    assert rows["flashinfer.direct"]["definition"]["canonical"] == "flashinfer._core.direct"
    assert rows["flashinfer.api.direct"]["definition"]["canonical"] == "flashinfer._core.direct"
    assert rows["flashinfer.Child"]["call_style"] == "wrapper-plan-run"
    assert rows["flashinfer.Segment"]["call_style"] == "wrapper-run"
    assert rows["flashinfer.Legacy"]["call_style"] == "wrapper-begin-forward"
    assert rows["flashinfer.Child.plan"]["definition"]["canonical"] == "flashinfer._core.Batch.plan"
    assert any(u["usage"] == "bound-call" for u in rows["flashinfer.Child.run"]["usages"])


def test_literal_reflection_self_members_and_parameter_shadowing(tmp_path):
    fixture_tree(tmp_path)
    source(tmp_path, "tests/test_calls.py", """
import flashinfer as fi
operation = fi.direct
def test_shadow(operation):
    operation()
class TestCase:
    def test_member(self):
        self.wrapper = fi.Batch()
        self.wrapper.plan()
        self.wrapper.run()
def test_reflection():
    for name in ["direct"]:
        getattr(fi, name)()
    fi.unknown()
""")
    rows = {row["symbol"]: row for row in scan(tmp_path)["symbols"]}
    calls = [u for u in rows["flashinfer.direct"]["usages"] if u["usage"] == "call"]
    assert [u["context"] for u in calls] == ["test_reflection"]
    assert any(u["usage"] == "bound-call" for u in rows["flashinfer.Batch.run"]["usages"])
    assert rows["flashinfer.unknown"]["definition"] is None


def test_finite_lazy_exports_resolve_without_executing_imports(tmp_path):
    fixture_tree(tmp_path)
    source(tmp_path, "flashinfer/lazy/__init__.py", """
_EXPORTS = {"direct": "._core"}
def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is not None:
        value = getattr(importlib.import_module(module_name, __name__), name)
        return value
""")
    source(tmp_path, "flashinfer/__init__.py", """
_NAMES = frozenset({"direct"})
def __getattr__(name):
    if name not in _NAMES:
        raise AttributeError(name)
    from . import lazy as implementation
    return getattr(implementation, name)
""")
    source(tmp_path, "flashinfer/lazy/_core.py", "def direct(): pass\n")
    tree = SourceTree(tmp_path)
    assert tree.resolve("flashinfer.direct")["canonical"] == "flashinfer.lazy._core.direct"


def test_reexported_function_can_shadow_the_submodule_name(tmp_path):
    fixture_tree(tmp_path)
    source(tmp_path, "flashinfer/functional.py", "def compute(): pass\n")
    source(tmp_path, "flashinfer/__init__.py", "from .functional import compute as functional\n")
    source(tmp_path, "tests/test_calls.py", "import flashinfer\ndef test_call(): flashinfer.functional()\n")
    rows = {row["symbol"]: row for row in scan(tmp_path)["symbols"]}
    assert rows["flashinfer.functional"]["definition"]["canonical"] == "flashinfer.functional.compute"
    assert rows["flashinfer.functional"]["call_style"] == "direct-function"


def test_parametrize_columns_do_not_invent_api_names(tmp_path):
    fixture_tree(tmp_path)
    source(tmp_path, "tests/test_calls.py", """
import flashinfer as fi
import pytest
@pytest.mark.parametrize("api_name,label", [("direct", "not_an_api")])
def test_named(api_name,label):
    getattr(fi,api_name)()
""")
    rows = {row["symbol"]: row for row in scan(tmp_path)["symbols"]}
    assert "flashinfer.not_an_api" not in rows
    assert any(u["usage"] == "call" for u in rows["flashinfer.direct"]["usages"])
