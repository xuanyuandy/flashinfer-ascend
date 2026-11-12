"""Trusted installed Ascend C declarations; no source or compiler discovery."""

from functools import lru_cache
import json
from pathlib import Path
from flashinfer.runtime import SchemaError
from flashinfer._resources import data_root, source_root

_DATA = data_root() / "kernels"


@lru_cache(maxsize=1)
def declarations():
    """Installed static metadata is independent of optional frozen binaries."""
    from flashinfer._resources import recipe_root
    from flashinfer.jit.recipe import load_build_recipes
    from flashinfer.runtime.manifest import load_kernel_manifest

    descriptors = {
        d.kernel_id: d for d in load_kernel_manifest(_DATA / "registry.json")
    }
    recipes = {}
    for path in sorted(recipe_root().glob("*.json")):
        for recipe in load_build_recipes(path):
            if recipe.kernel_id not in descriptors:
                continue
            if recipe.kernel_id in recipes:
                raise SchemaError(
                    "duplicate installed native recipe: " + recipe.kernel_id
                )
            recipe.validate_descriptor(descriptors[recipe.kernel_id])
            recipes[recipe.kernel_id] = recipe
    if set(recipes) != set(descriptors):
        raise SchemaError("installed native declarations and recipes disagree")
    return {key: (descriptor, recipes[key]) for key, descriptor in descriptors.items()}


@lru_cache(maxsize=1)
def _contracts():
    from flashinfer.jit.core import JitSpec
    from flashinfer.jit.recipe import BuildRecipe
    from flashinfer.runtime.manifest import load_kernel_manifest

    if not (_DATA / "ascendc_specs.json").is_file():
        raise SchemaError(
            "frozen Ascend C metadata is missing; run python -m flashinfer.aot --freeze before execution or packaging"
        )
    descriptors = load_kernel_manifest(_DATA / "registry.json")
    payload = json.loads((_DATA / "ascendc_specs.json").read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "kernels"} or payload["schema_version"] != 1:
        raise SchemaError("unsupported frozen Ascend C declaration schema")
    frozen = {}
    for record in payload["kernels"]:
        if (
            set(record) != {"kernel_id", "recipe", "spec"}
            or record["kernel_id"] in frozen
        ):
            raise SchemaError("invalid or duplicate frozen Ascend C declaration")
        frozen[record["kernel_id"]] = record
    if set(frozen) != {descriptor.kernel_id for descriptor in descriptors}:
        raise SchemaError("registry and frozen Ascend C declarations disagree")
    contracts = {}
    for descriptor in descriptors:
        record = frozen[descriptor.kernel_id]
        recipe = BuildRecipe.from_dict(record["recipe"])
        spec = JitSpec.from_dict(record["spec"])
        recipe.validate_descriptor(descriptor)
        spec.validate_descriptor(descriptor)
        if dict(spec.input_fingerprints).get("build_recipe") != recipe.fingerprint:
            raise SchemaError("frozen Ascend C spec does not bind its recipe")
        contracts[descriptor.kernel_id] = (descriptor, recipe, spec)
    return contracts


def load_contract(kernel_id):
    """Read trusted installed metadata without touching sources or a compiler."""
    try:
        contract = _contracts()[kernel_id]
    except KeyError as error:
        raise SchemaError("no frozen Ascend C declaration for " + kernel_id) from error
    prepared = source_contract(kernel_id)
    if prepared is not None:
        if (
            prepared["descriptor"].fingerprint != contract[0].fingerprint
            or prepared["recipe"].fingerprint != contract[1].fingerprint
        ):
            raise SchemaError(
                "source declaration differs from frozen Ascend C metadata; regenerate and freeze: "
                + kernel_id
            )
    return contract


def source_contract(kernel_id):
    """Read author inputs without frozen metadata, SDK discovery or compilation."""
    root = source_root()
    if root is None or _DATA != data_root() / "kernels":
        return None
    from flashinfer.operators.implementation import implementations

    declaration = next(
        (d for d in implementations(backend="ascendc") if d.kernel_id == kernel_id),
        None,
    )
    return None if declaration is None else declaration.prepare_outputs(root)
