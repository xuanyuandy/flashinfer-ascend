"""Shared release-author operations for the production Ascend C tools."""

import argparse
import hashlib
import json
import os
from pathlib import Path


def production_descriptor(recipe, *, op, dtype, workspace_bytes):
    from flashinfer.runtime import (
        Backend, CapabilitySpec, CompileMode, GraphCapability, KernelDescriptor,
        KernelConstraints, KernelLaunchABI, ShapeRange, WorkspaceFormula,
    )

    bounds = {bound.name: bound for bound in recipe.shape_constraints}
    ranges = tuple(
        ShapeRange(symbol, index, axis, bounds[symbol].minimum, bounds[symbol].maximum)
        for index, parameter in enumerate(recipe.tensor_parameters)
        for axis, symbol in enumerate(parameter.shape)
    )
    return KernelDescriptor(
        recipe.kernel_id, op, Backend.ASCENDC,
        constraints=KernelConstraints(dtype_signatures=recipe.dtype_signatures),
        capability=CapabilitySpec(
            op=op, dtypes=(dtype,) if isinstance(dtype, str) else tuple(dtype),
            soc_models=("Ascend950",), physical_layouts=("ND",),
            shape_ranges=ranges, specializations=(("tensor_contract", ("supported",)),),
        ),
        compile_mode=CompileMode.AOT, source=recipe.source,
        target_soc=recipe.target_soc, tiling_schema_version=recipe.tiling_schema_version,
        workspace=WorkspaceFormula(constant_bytes=workspace_bytes),
        launch_abi=KernelLaunchABI(
            recipe.launch_abi, recipe.host_entry,
            ("workspace", "workspace_bytes", "executor", "stream"),
        ),
        graph_capability=GraphCapability.NOT_CAPTURABLE,
    )


def upsert(records, record, *, append=False):
    """Preserve each tool's historical replacement order."""
    if append:
        records[:] = [item for item in records if item["kernel_id"] != record["kernel_id"]]
    else:
        for index, previous in enumerate(records):
            if previous["kernel_id"] == record["kernel_id"]:
                records[index] = record
                return
    records.append(record)


def freeze_declaration(root, *, recipe_name, kernel_id, descriptor_factory,
                       specialization, dependency_roots=None, append=False,
                       generated_at=None, allow_new_frozen=False):
    """Resolve, declare and persist a single explicitly selected release entry."""
    from flashinfer.ascendc import _contracts
    from flashinfer.jit import (
        AscendCBackend, AscendCToolchain, JitSpecRegistry, declare_kernel,
        load_build_recipes, resolve_build_recipe,
    )
    from flashinfer.runtime import KernelRegistry

    root = Path(root)
    recipes = load_build_recipes(root / "csrc/recipes" / (recipe_name + ".json"))
    recipe = next(item for item in recipes if item.kernel_id == kernel_id)
    toolchain = AscendCToolchain.discover()
    resolution = dict(source_root=root, toolchain_fingerprint=toolchain.fingerprint)
    if dependency_roots is not None:
        resolution["dependency_roots"] = dependency_roots(recipe, root)
    resolved = resolve_build_recipe(recipe, **resolution)
    backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
    descriptor = descriptor_factory(recipe)
    spec = declare_kernel(
        recipe.kernel_id, backend=backend, registry=KernelRegistry((descriptor,)),
        spec_registry=JitSpecRegistry(), output_path="op-package.json",
        entry_point=recipe.host_entry, environment_fingerprint=toolchain.fingerprint,
        specialization_fingerprint=hashlib.sha256(specialization).hexdigest(),
    )
    data = root / "flashinfer/data/kernels"
    registry_path, frozen_path = data / "registry.json", data / "ascendc_specs.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    frozen = (dict(schema_version=1, kernels=[]) if allow_new_frozen and not frozen_path.exists()
              else json.loads(frozen_path.read_text(encoding="utf-8")))
    record = descriptor.to_dict()
    record["graph_capability"] = descriptor.graph_capability.value
    upsert(registry["kernels"], record, append=append)
    upsert(frozen["kernels"], dict(kernel_id=recipe.kernel_id, recipe=recipe.to_dict(), spec=spec.to_dict()),
           append=append)
    if generated_at is not None:
        registry["generated_at"] = generated_at
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    _contracts.cache_clear()
    return descriptor, recipe, spec


def release_main(*, description, freeze_contract, build_release, freeze_help, catlass_override=False):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--freeze", action="store_true", help=freeze_help)
    parser.add_argument("--output", type=Path, help="build and export into this directory")
    if catlass_override:
        parser.add_argument("--catlass-root", type=Path, help="override FLASHINFER_CATLASS_ROOT for this build")
    args = parser.parse_args()
    if not args.freeze and args.output is None:
        parser.error("supply --freeze and/or --output")
    if catlass_override and args.catlass_root is not None:
        os.environ["FLASHINFER_CATLASS_ROOT"] = str(args.catlass_root)
    result = {}
    if args.freeze:
        descriptor, recipe, spec = freeze_contract()
        result.update(kernel_id=descriptor.kernel_id, recipe=recipe.fingerprint,
                      spec=spec.fingerprint, toolchain=spec.environment_fingerprint)
    if args.output is not None:
        result["release"] = str(build_release(args.output))
    print(json.dumps(result, sort_keys=True))
