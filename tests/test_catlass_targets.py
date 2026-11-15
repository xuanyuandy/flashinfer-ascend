"""CATLASS target selection and release pin stay explicit and independent."""

from pathlib import Path

import pytest

from flashinfer.runtime import HardwareSignature, InputSignature, SchemaError, dispatch


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("soc", ("Ascend910B", "Ascend910C", "Ascend950"))
def test_same_public_prefill_signature_selects_the_matching_target(soc):
    request = InputSignature("ascendc.single_prefill_catlass",
        dtypes=("float16",) * 4,
        shapes=((128, 2, 128), (256, 2, 128), (256, 2, 128), (128, 2, 128)),
        physical_layouts=("ND",) * 4,
        specializations=(("tensor_contract", "supported"),))
    decision = dispatch(request, HardwareSignature(soc, 24), backend="ascendc")
    assert decision.kernel_id == "ascendc/t4/flash_attention/catlass/op_package/" + soc


def test_probe_target_mapping_updates_catlass_template_architecture():
    from flashinfer.aot.build import _for_target
    from flashinfer.jit import load_build_recipes

    recipe = next(r for r in load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json")
                  if r.implementation == "catlass" and r.target_soc == "Ascend950")
    mapped = _for_target(recipe, "Ascend910B")
    assert mapped.npu_arch == "dav-2201" and mapped.definitions == ("CATLASS_ARCH=2201",)
    assert recipe.npu_arch == "dav-3510" and recipe.definitions == ("CATLASS_ARCH=3510",)


def test_legacy_catlass_pin_cannot_be_used_by_public_v2_recipes():
    from dataclasses import replace
    from flashinfer.jit import load_build_recipes
    from flashinfer.jit.recipe import BuildDependency, _LegacyBuildDependency, _LEGACY_CATLASS_REVISION

    with pytest.raises(SchemaError, match="pinned CATLASS"):
        BuildDependency("catlass", _LEGACY_CATLASS_REVISION)
    old = _LegacyBuildDependency("catlass", _LEGACY_CATLASS_REVISION)
    recipe = next(r for r in load_build_recipes(ROOT / "csrc/recipes/matmul_probe.json")
                  if r.implementation == "catlass")
    with pytest.raises(SchemaError, match="current pinned"):
        replace(recipe, dependencies=(old,))
