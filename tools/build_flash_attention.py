"""Freeze and build the bounded CATLASS FlashAttention production entry.

  FLASHINFER_CATLASS_ROOT=/mnt/fi/catlass-ref \
    /root/miniforge3/envs/flashinfer-a5/bin/python -B \
    tools/build_flash_attention.py --freeze --output /mnt/fi/t4-release

This release-author operation updates only the T4 entry. Import, dispatch and
verified AOT hits never discover a compiler or read the CATLASS checkout.
"""

import argparse
import hashlib
import json
import os
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
        recipe.kernel_id, "ascendc.single_prefill_catlass", Backend.ASCENDC,
        capability=CapabilitySpec(
            op="ascendc.single_prefill_catlass", dtypes=("float16",),
            soc_models=("Ascend950",), physical_layouts=("ND",),
            shape_ranges=ranges,
            specializations=(("tensor_contract", ("supported",)),),
        ),
        compile_mode=CompileMode.AOT, source=recipe.source,
        target_soc=recipe.target_soc, tiling_schema_version=recipe.tiling_schema_version,
        # This 950 kernel keeps intermediate tiles in L1/UB: user GM scratch
        # is zero. CANN's dynamic ACLNN workspace includes its 16 MiB system
        # prefix on the validated release; eager uses the exact callback size.
        workspace=WorkspaceFormula(constant_bytes=16 << 20),
        launch_abi=KernelLaunchABI(
            recipe.launch_abi, recipe.host_entry,
            ("workspace", "workspace_bytes", "executor", "stream"),
        ),
        graph_capability=GraphCapability.NOT_CAPTURABLE,
    )


def _upsert(records, record):
    """Replace T4 in place while preserving all other frozen records."""
    for index, previous in enumerate(records):
        if previous["kernel_id"] == record["kernel_id"]:
            records[index] = record
            return
    records.append(record)


def freeze_contract():
    """Resolve the pinned dependency and freeze only T4's descriptor/spec."""
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID, _contracts, _dependency_roots
    from flashinfer.jit import (
        AscendCBackend, AscendCToolchain, JitSpecRegistry, declare_kernel,
        load_build_recipes, resolve_build_recipe,
    )
    from flashinfer.runtime import KernelRegistry

    recipes = load_build_recipes(ROOT / "csrc/recipes/flash_attention.json")
    recipe = next(item for item in recipes if item.kernel_id == FLASH_ATTENTION_KERNEL_ID)
    toolchain = AscendCToolchain.discover()
    resolved = resolve_build_recipe(
        recipe, source_root=ROOT, toolchain_fingerprint=toolchain.fingerprint,
        dependency_roots=_dependency_roots(recipe, ROOT),
    )
    backend = AscendCBackend(recipes=(resolved,), toolchain=toolchain)
    descriptor = _descriptor(recipe)
    spec = declare_kernel(
        recipe.kernel_id, backend=backend, registry=KernelRegistry((descriptor,)),
        spec_registry=JitSpecRegistry(), output_path="op-package.json",
        entry_point=recipe.host_entry, environment_fingerprint=toolchain.fingerprint,
        specialization_fingerprint=hashlib.sha256(
            b"flash_attention;catlass-example49;contract-v2;fp16;NHD;B1;"
            b"D128;H1..8;S,T128..512:128;unmasked;fixed-length"
        ).hexdigest(),
    )
    data = ROOT / "flashinfer/data/kernels"
    registry_path = data / "registry.json"
    frozen_path = data / "ascendc_specs.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    record = descriptor.to_dict()
    record["graph_capability"] = descriptor.graph_capability.value
    _upsert(registry["kernels"], record)
    _upsert(frozen["kernels"], {
        "kernel_id": recipe.kernel_id, "recipe": recipe.to_dict(), "spec": spec.to_dict(),
    })
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    frozen_path.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    _contracts.cache_clear()
    return descriptor, recipe, spec


def build_release(root):
    """Build/export the existing frozen T4 identity using the common builder."""
    from flashinfer.aot.build import export_op_package
    from flashinfer.ascendc import FLASH_ATTENTION_KERNEL_ID, _jit_backend, load_contract
    from flashinfer.jit import DiskJitBuildStore
    from flashinfer.jit.build import _build_record

    _, recipe, spec = load_contract(FLASH_ATTENTION_KERNEL_ID)
    backend = _jit_backend(recipe, spec)
    root = Path(root)
    store = DiskJitBuildStore(root / "build-cache")
    record = _build_record(spec, backend=backend, store=store)
    return export_op_package(store.directory(record), root / "release")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true", help="refresh only T4 registry/spec JSON")
    parser.add_argument("--output", type=Path, help="build and export into this directory")
    parser.add_argument("--catlass-root", type=Path, help="override FLASHINFER_CATLASS_ROOT for this build")
    args = parser.parse_args()
    if not args.freeze and args.output is None:
        parser.error("supply --freeze and/or --output")
    if args.catlass_root is not None:
        os.environ["FLASHINFER_CATLASS_ROOT"] = str(args.catlass_root)
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
