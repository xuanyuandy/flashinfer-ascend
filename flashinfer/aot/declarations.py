"""Freeze a production catalog from descriptors and recipe manifests.

Operator authors supply data, not a Python build/registration script. Existing
specialization labels and spec flags are explicit release inputs retained on
refreeze; complete recipe/source/environment fingerprints still bind the code.
"""

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile

from flashinfer.jit import (
    AscendCBackend,
    AscendCToolchain,
    BuildRecipe,
    JitSpec,
    JitSpecRegistry,
    declare_kernel,
    load_build_recipes,
    resolve_build_recipe,
)
from flashinfer.jit.dependencies import dependency_roots as resolve_dependency_roots
from flashinfer.runtime import Backend, CompileMode, KernelRegistry, SchemaError
from flashinfer.runtime.manifest import load_kernel_manifest


def _write_if_changed(path, payload):
    encoded = (json.dumps(payload, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() == encoded:
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, path.stat().st_mode & 0o777 if path.exists() else 0o644)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def freeze_production_kernels(
    *,
    source_root=None,
    kernel_ids=(),
    cann_root=None,
    dependency_roots=None,
    registry_path=None,
    frozen_path=None,
    targets=()
):
    """Validate the whole selection, then publish its frozen declarations.

    The central registry is the authoring input for capability/workspace/ABI.
    Recipes are joined by kernel_id, never by an operator-name switch. Source
    identities are refreshed from recipes; capability and ABI are not inferred
    or broadened. This operation declares units and never compiles them.
    """
    root = Path(source_root) if source_root else Path(__file__).resolve().parents[2]
    from flashinfer._resources import data_root, recipe_root

    if (root / "pyproject.toml").is_file():
        from flashinfer.operators.authoring import generate

        generate(source_root=root)
    recipe_dir = recipe_root(root)
    if not recipe_dir.is_dir():
        raise SchemaError("freezing requires generated build recipes")
    data = data_root(root) / "kernels"
    registry_path = Path(registry_path) if registry_path else data / "registry.json"
    frozen_path = Path(frozen_path) if frozen_path else data / "ascendc_specs.json"
    descriptors = {d.kernel_id: d for d in load_kernel_manifest(registry_path)}
    selected = tuple(kernel_ids) if kernel_ids else tuple(descriptors)
    if not selected or len(selected) != len(set(selected)):
        raise SchemaError(
            "freeze requires a nonempty selection of unique kernel_id values"
        )
    unknown = set(selected) - set(descriptors)
    if unknown:
        raise SchemaError("no production descriptor for " + ", ".join(sorted(unknown)))

    recipes = {}
    for path in sorted(recipe_dir.glob("*.json")):
        for recipe in load_build_recipes(path):
            if recipe.kernel_id in recipes:
                raise SchemaError("duplicate recipe kernel_id: " + recipe.kernel_id)
            recipes[recipe.kernel_id] = recipe
    for kernel_id in selected:
        descriptor = descriptors[kernel_id]
        if (
            descriptor.backend != Backend.ASCENDC
            or descriptor.compile_mode != CompileMode.AOT
        ):
            raise SchemaError(
                "production freeze requires an ascendc AOT descriptor: " + kernel_id
            )
        if kernel_id not in recipes:
            raise SchemaError("no recipe for production kernel_id: " + kernel_id)
        if targets and set(targets) != {recipes[kernel_id].target_soc}:
            raise SchemaError(
                "frozen production kernel %s requires target %s; requested %s"
                % (kernel_id, recipes[kernel_id].target_soc, ", ".join(targets))
            )

    registry_payload = json.loads(registry_path.read_text(encoding="utf-8"))
    frozen_payload = (
        json.loads(frozen_path.read_text(encoding="utf-8"))
        if frozen_path.exists()
        else dict(schema_version=1, kernels=[])
    )
    if (
        set(frozen_payload) != {"schema_version", "kernels"}
        or frozen_payload["schema_version"] != 1
        or not isinstance(frozen_payload["kernels"], list)
    ):
        raise SchemaError("unsupported frozen Ascend C declaration schema")
    records = {}
    previous_specs = {}
    for record in frozen_payload["kernels"]:
        if (
            set(record) != {"kernel_id", "recipe", "spec"}
            or record["kernel_id"] in records
        ):
            raise SchemaError("invalid or duplicate frozen Ascend C declaration")
        records[record["kernel_id"]] = record
        previous_specs[record["kernel_id"]] = JitSpec.from_dict(record["spec"])
    if set(records) - set(descriptors):
        raise SchemaError("frozen declarations contain unregistered kernel_id values")

    # Complete dependency and source preflight before writing either manifest.
    roots = {
        kernel_id: resolve_dependency_roots(
            recipes[kernel_id], root, overrides=dependency_roots
        )
        for kernel_id in selected
    }
    toolchain = AscendCToolchain.discover(cann_root)
    prepared = []
    for kernel_id in selected:
        recipe = recipes[kernel_id]
        descriptor = replace(descriptors[kernel_id], source=recipe.source)
        recipe.validate_descriptor(descriptor)
        resolved = resolve_build_recipe(
            recipe,
            source_root=root,
            toolchain_fingerprint=toolchain.fingerprint,
            dependency_roots=roots[kernel_id],
        )
        backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
        previous = previous_specs.get(kernel_id)
        spec = declare_kernel(
            kernel_id,
            backend=backend,
            registry=KernelRegistry((descriptor,)),
            spec_registry=JitSpecRegistry(),
            output_path="op-package.json",
            entry_point=recipe.host_entry,
            environment_fingerprint=toolchain.fingerprint,
            specialization_fingerprint=(
                previous.specialization_fingerprint
                if previous is not None
                else recipe.fingerprint
            ),
            compile_options=previous.compile_options if previous is not None else (),
        )
        prepared.append((descriptor, recipe, spec))
        descriptors[kernel_id] = descriptor
        records[kernel_id] = dict(
            kernel_id=kernel_id, recipe=recipe.to_dict(), spec=spec.to_dict()
        )

    # A partial selection may not leave an incomplete or stale runtime catalog.
    if set(records) != set(descriptors):
        raise SchemaError(
            "freeze all newly registered kernels before publishing the catalog"
        )
    for kernel_id, descriptor in descriptors.items():
        record = records[kernel_id]
        recipe, spec = BuildRecipe.from_dict(record["recipe"]), JitSpec.from_dict(
            record["spec"]
        )
        recipe.validate_descriptor(descriptor)
        spec.validate_descriptor(descriptor)
        if (
            dict(spec.input_fingerprints).get("kernel") != descriptor.fingerprint
            or dict(spec.input_fingerprints).get("build_recipe") != recipe.fingerprint
        ):
            raise SchemaError(
                "unselected declaration is stale; include kernel_id in freeze: "
                + kernel_id
            )

    # Preserve catalog order and all authored capability/workspace/ABI fields.
    for record in registry_payload["kernels"]:
        if record["kernel_id"] in selected:
            record["source"] = descriptors[record["kernel_id"]].source.to_dict()
    frozen_payload["kernels"] = list(records.values())
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    frozen_path.parent.mkdir(parents=True, exist_ok=True)
    _write_if_changed(registry_path, registry_payload)
    _write_if_changed(frozen_path, frozen_payload)
    from flashinfer.ascendc import _contracts

    _contracts.cache_clear()
    return tuple(prepared)


__all__ = ["freeze_production_kernels"]
