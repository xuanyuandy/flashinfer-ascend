"""Freeze the production declaration and/or build its portable AOT release.

Run from the source checkout using the supported Python/CANN environment:
  python -B tools/build_matmul_bias_silu.py --freeze --output /mnt/fi/t2-release

Freezing is a release-author action, never part of import, dispatch or an AOT
hit. The resulting static JSON is shipped with the production registry entry.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _descriptor(recipe):
    from flashinfer.runtime import (
        Backend, CapabilitySpec, CompileMode, GraphCapability, KernelDescriptor,
        KernelLaunchABI, ShapeRange, WorkspaceFormula,
    )

    bounds = {bound.name: bound for bound in recipe.shape_constraints}
    ranges = tuple(
        ShapeRange(symbol, index, axis, bounds[symbol].minimum, bounds[symbol].maximum)
        for index, parameter in enumerate(recipe.tensor_parameters)
        for axis, symbol in enumerate(parameter.shape)
    )
    return KernelDescriptor(
        recipe.kernel_id, "gemm.matmul_bias_silu", Backend.ASCENDC,
        capability=CapabilitySpec(
            op="gemm.matmul_bias_silu", dtypes=("bfloat16",),
            soc_models=("Ascend950",), physical_layouts=("ND",),
            shape_ranges=ranges,
            specializations=(("tensor_contract", ("supported",)),),
        ),
        compile_mode=CompileMode.AOT, source=recipe.source,
        target_soc=recipe.target_soc, tiling_schema_version=recipe.tiling_schema_version,
        # The validated Ascend950/CANN release reserves 16 MiB for system
        # workspace. Planning declares an upper bound over the finite shape
        # contract; eager allocation uses the exact dynamic ACLNN result.
        workspace=WorkspaceFormula(
            constant_bytes=(16 << 20) + 4 * bounds["M"].maximum * bounds["N"].maximum,
        ),
        launch_abi=KernelLaunchABI(
            recipe.launch_abi, recipe.host_entry,
            ("workspace", "workspace_bytes", "executor", "stream"),
        ),
        graph_capability=GraphCapability.NOT_CAPTURABLE,
    )


def freeze_contract():
    """Write registry and frozen recipe/spec using the current verified SDK."""
    from flashinfer.ascendc import MATMUL_BIAS_SILU_KERNEL_ID, _contracts
    from flashinfer.jit import (
        AscendCBackend, AscendCToolchain, JitSpecRegistry, declare_kernel,
        load_build_recipes, resolve_build_recipe,
    )
    from flashinfer.runtime import KernelRegistry

    recipes = load_build_recipes(ROOT / "csrc/recipes/matmul_bias_silu.json")
    recipe = next(recipe for recipe in recipes if recipe.kernel_id == MATMUL_BIAS_SILU_KERNEL_ID)
    toolchain = AscendCToolchain.discover()
    resolved = resolve_build_recipe(
        recipe, source_root=ROOT, toolchain_fingerprint=toolchain.fingerprint,
    )
    backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
    descriptor = _descriptor(recipe)
    spec = declare_kernel(
        recipe.kernel_id, backend=backend, registry=KernelRegistry((descriptor,)),
        spec_registry=JitSpecRegistry(), output_path="op-package.json",
        entry_point=recipe.host_entry, environment_fingerprint=toolchain.fingerprint,
        specialization_fingerprint=hashlib.sha256(
            b"matmul_bias_silu;contract-v2;bf16;row-major;M,N,K=16..256:16"
        ).hexdigest(),
    )
    data = ROOT / "flashinfer/data/kernels"
    registry_path = data / "registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    record = descriptor.to_dict()
    record["graph_capability"] = descriptor.graph_capability.value
    registry["kernels"] = [item for item in registry["kernels"]
                           if item["kernel_id"] != recipe.kernel_id] + [record]
    registry["generated_at"] = "2026-10-25"
    frozen_path = data / "ascendc_specs.json"
    frozen = (json.loads(frozen_path.read_text(encoding="utf-8")) if frozen_path.exists()
              else {"schema_version": 1, "kernels": []})
    frozen["kernels"] = [item for item in frozen["kernels"]
                         if item["kernel_id"] != recipe.kernel_id] + [
        {"kernel_id": recipe.kernel_id, "recipe": recipe.to_dict(), "spec": spec.to_dict()},
    ]
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    _contracts.cache_clear()
    return descriptor, recipe, spec


def build_release(root):
    """Build/export the exact frozen production identity; return its directory."""
    from flashinfer.aot.build import export_op_package
    from flashinfer.ascendc import _jit_backend, load_contract
    from flashinfer.jit import DiskJitBuildStore
    from flashinfer.jit.build import _build_record

    _, recipe, spec = load_contract()
    backend = _jit_backend(recipe, spec)
    root = Path(root)
    store = DiskJitBuildStore(root / "build-cache")
    record = _build_record(spec, backend=backend, store=store)
    return export_op_package(store.directory(record), root / "release")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="refresh packaged registry/spec JSON")
    parser.add_argument("--output", type=Path, help="build and export into this directory")
    args = parser.parse_args()
    if not args.freeze and args.output is None:
        parser.error("supply --freeze and/or --output")
    result = {}
    if args.freeze:
        descriptor, recipe, spec = freeze_contract()
        result.update(kernel_id=descriptor.kernel_id, recipe=recipe.fingerprint,
                      spec=spec.fingerprint, toolchain=spec.environment_fingerprint)
    if args.output is not None:
        result["release"] = str(build_release(args.output))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
