"""Wheel families group delivery, without aliasing physical compilation targets."""

import json
import os
from pathlib import Path

from .runtime.hardware import current_hardware
from .runtime import SchemaError
from .runtime.targets import ARCH22_SOCS, canonical_soc_model


FAMILIES = {"a2a3": ARCH22_SOCS, "ascend950": ("Ascend950",)}
DISTRIBUTIONS = {"a2a3": "flashinfer_ascend910b", "ascend950": "flashinfer_ascend950"}


def canonical_target(value):
    """Return the delivery family, never a replacement runtime SoC."""
    key = value.strip().lower()
    if key in FAMILIES:
        return key
    soc = canonical_soc_model(value.strip())
    return next(name for name, socs in FAMILIES.items() if soc in socs)


def family_socs(value):
    return FAMILIES[canonical_target(value)]


def selected_families(values):
    """Reject repeated chip selections; coalesce B and C into one release."""
    identities = tuple(value.strip().lower() if value.strip().lower() in FAMILIES
                       else canonical_soc_model(value.strip()) for value in values)
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate chip targets")
    return tuple(dict.fromkeys(canonical_target(value) for value in values))


def distribution_name(target):
    return DISTRIBUTIONS[canonical_target(target)]


def prebuilt_packages(directory):
    root = Path(directory).resolve(strict=True)
    paths = ((root,) if (root / "manifest.json").is_file() else tuple(
        p for p in sorted(root.iterdir())
        if p.is_dir() and (p / "manifest.json").is_file()))
    if not paths:
        raise ValueError("FLASHINFER_AOT_PREBUILT contains no verified op-package releases")
    return paths


def wheel_target():
    requested = os.environ.get("FLASHINFER_AOT_SOCS", "")
    if requested:
        targets = selected_families(requested.split(","))
        if len(targets) != 1:
            raise ValueError("one wheel requires one family; build separate 910B and 950 release wheels")
        return targets[0]
    prebuilt = os.environ.get("FLASHINFER_AOT_PREBUILT")
    if prebuilt:
        targets = {canonical_target(json.loads((p / "manifest.json").read_text())["spec"]["target_soc"])
                   for p in prebuilt_packages(prebuilt)}
        if len(targets) != 1:
            raise ValueError("PREBUILT wheel requires one family; select 910B or 950 explicitly")
        return targets.pop()
    hardware = current_hardware()
    try:
        return canonical_target(hardware.soc_model)
    except ValueError as error:
        raise RuntimeError("cannot detect a supported local NPU; set FLASHINFER_AOT_SOCS="
                           "Ascend910B, Ascend910C or Ascend950 for an explicit release build") from error


def production_kernel_ids(target):
    from .operators.implementation import implementations

    socs = family_socs(target)
    result = tuple(d.kernel_id for d in implementations(backend="ascendc")
                   if d.native.target_soc in socs)
    if not result:
        raise SchemaError("no production Ascend C implementation for wheel target " + target)
    return result


def restore_prebuilt_specs(package, releases):
    """Bind verified exports to current author declarations without an SDK.

    A fresh checkout has generated declarations but no frozen environment.
    PREBUILT takes that environment/spec from each verified release, rather
    than depending on an earlier local compile or freezing on the packager.
    """
    from .jit.cache import verify_build_directory
    from .jit.core import JitSpec
    from .jit.op_package import verify_op_package
    from .jit.recipe import load_build_recipes
    from .runtime.manifest import load_kernel_manifest

    data = Path(package) / "data"
    descriptors = {d.kernel_id: d for d in load_kernel_manifest(data / "kernels/registry.json")}
    authored = {r.kernel_id: r for p in (data / "build/csrc/recipes").glob("*.json")
                for r in load_build_recipes(p)}
    records = []
    seen = set()
    for release in releases:
        release = Path(release)
        verify_build_directory(release)
        spec = JitSpec.from_dict(json.loads((release / "manifest.json").read_text())["spec"])
        _, recipe = verify_op_package(release, spec)
        descriptor = descriptors.get(spec.kernel_id)
        if (descriptor is None or spec.kernel_id in seen
                or authored.get(spec.kernel_id) != recipe
                or dict(spec.input_fingerprints).get("kernel") != descriptor.fingerprint):
            raise SchemaError("PREBUILT does not match the current operator declaration: " + spec.kernel_id)
        recipe.validate_descriptor(descriptor)
        spec.validate_descriptor(descriptor)
        seen.add(spec.kernel_id)
        records.append(dict(kernel_id=spec.kernel_id, recipe=recipe.to_dict(), spec=spec.to_dict()))
    if not records:
        raise SchemaError("PREBUILT contains no matching native declarations")
    (data / "kernels/ascendc_specs.json").write_text(
        json.dumps(dict(schema_version=1, kernels=records), indent=2) + "\n")


def finalize_resources(package, *, target, kernel_ids):
    """Keep the installed native catalogs exactly equal to the shipped kernels."""
    target = canonical_target(target)
    selected = set(kernel_ids)
    if not selected or len(selected) != len(kernel_ids):
        raise ValueError("wheel requires a nonempty, unique kernel inventory")
    data = Path(package) / "data"
    # A shared wheel must carry every authored counterpart of each selected
    # recipe. A B-only PREBUILT release cannot advertise the B/C family.
    recipe_files = tuple((data / "build/csrc/recipes").glob("*.json"))
    authored = [r for path in recipe_files for r in json.loads(path.read_text())]
    def variant_key(record):
        return (record.get("recipe_id", record["kernel_id"]), record.get("implementation"))
    groups = {variant_key(r) for r in authored if r["kernel_id"] in selected}
    required = {r["kernel_id"] for r in authored
                if variant_key(r) in groups and r["target_soc"] in family_socs(target)}
    if required != selected:
        raise ValueError("wheel family is missing authored recipe variants: "
                         + ", ".join(sorted(required - selected)))
    catalogs = (
        (data / "kernels/registry.json", "kernels", "kernel_id"),
        (data / "kernels/ascendc_specs.json", "kernels", "kernel_id"),
    )
    prepared = []
    for path, key, identity in catalogs:
        payload = json.loads(path.read_text())
        records = [r for r in payload[key] if r[identity] in selected]
        if {r[identity] for r in records} != selected:
            raise ValueError("wheel catalog is incomplete: " + str(path))
        for record in records:
            soc = record.get("target_soc", record.get("recipe", {}).get("target_soc"))
            if canonical_target(soc) != target:
                raise ValueError("wheel catalog contains a different chip target")
        payload[key] = records
        prepared.append((path, payload))
    path = data / "implementations.json"
    payload = json.loads(path.read_text())
    payload["implementations"] = [r for r in payload["implementations"]
        if r["backend"] != "ascendc" or r["kernel_id"] in selected]
    if {r["kernel_id"] for r in payload["implementations"] if r["backend"] == "ascendc"} != selected:
        raise ValueError("wheel implementation catalog does not match its frozen native inventory")
    prepared.append((path, payload))
    for path, payload in prepared:
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    for path in recipe_files:
        records = [r for r in json.loads(path.read_text()) if r["kernel_id"] in selected]
        if records:
            path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
        else:
            path.unlink()
    (data / "wheel.json").write_text(json.dumps(dict(schema_version=2,
        family=target, soc_models=family_socs(target), distribution=distribution_name(target),
        kernel_ids=sorted(selected)), indent=2) + "\n")
