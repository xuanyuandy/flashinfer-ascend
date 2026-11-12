"""Trusted installed Ascend C declarations; no source or compiler discovery."""

from functools import lru_cache
import json
from pathlib import Path
from flashinfer.runtime import SchemaError

_DATA = Path(__file__).resolve().parents[1] / "data/kernels"

@lru_cache(maxsize=1)
def _contracts():
    from flashinfer.jit.core import JitSpec
    from flashinfer.jit.recipe import BuildRecipe
    from flashinfer.runtime.manifest import load_kernel_manifest

    descriptors = load_kernel_manifest(_DATA / "registry.json")
    payload = json.loads((_DATA / "ascendc_specs.json").read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "kernels"} or payload["schema_version"] != 1:
        raise SchemaError("unsupported frozen Ascend C declaration schema")
    frozen = {}
    for record in payload["kernels"]:
        if set(record) != {"kernel_id", "recipe", "spec"} or record["kernel_id"] in frozen:
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
        return _contracts()[kernel_id]
    except KeyError as error:
        raise SchemaError("no frozen Ascend C declaration for " + kernel_id) from error
