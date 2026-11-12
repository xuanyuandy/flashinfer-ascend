"""Derive Ascend C build metadata from an implementation-owned native definition."""

import hashlib
from pathlib import Path

from flashinfer.jit.recipe import BuildRecipe, _path
from flashinfer.jit.op_package import package_target
from flashinfer.runtime import (
    Backend,
    CompileMode,
    KernelDescriptor,
    KernelLaunchABI,
    SourceRef,
    WorkspaceFormula,
    SchemaError,
)
from flashinfer.runtime.manifest import descriptor_from_dict


def prepare_native(root, directory, native):
    root = Path(root).resolve()
    source_dir = root / _path(native.source_dir)
    if (
        source_dir.is_symlink()
        or not source_dir.resolve().is_relative_to(root)
        or not source_dir.is_dir()
    ):
        raise SchemaError("native source must be an in-repository regular directory")
    build = native.build

    def reference(path, language="ascendc"):
        path = Path(path)
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root)
            or not path.is_file()
        ):
            raise SchemaError(
                "build input must be an in-directory regular file: " + str(path)
            )
        return SourceRef(
            language,
            path.relative_to(root).as_posix(),
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )

    def local(relative):
        path = source_dir / _path(relative)
        if not path.resolve().is_relative_to(source_dir.resolve()):
            raise SchemaError("native input escapes its source directory")
        return path

    source = reference(local(build.device))
    host = tuple(reference(local(p)) for p in build.host)
    selected = {source.locator} | {p.locator for p in host}
    native_files = [
        p
        for p in sorted(source_dir.rglob("*"))
        if p.is_file() and p.suffix in {".cpp", ".h", ".inc"}
    ]
    headers = tuple(
        reference(p)
        for p in native_files
        if p.relative_to(root).as_posix() not in selected
    )
    extra = []
    native_paths = {p.resolve() for p in native_files}
    for path in sorted(source_dir.rglob("*")):
        if (
            not path.is_file()
            or any(p in {"__pycache__", ".pytest_cache", ".git"} for p in path.parts)
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        if path.resolve() not in native_paths:
            extra.append(reference(path, "operator"))
    # Python adaptation and logical hooks are also part of the closed source identity.
    logical = (
        root
        / "flashinfer/operators"
        / directory.parent.name
        / native.op.partition(".")[2]
    )
    hook_directories = (directory, logical)
    for hook_directory in hook_directories:
        if hook_directory.is_dir():
            extra.extend(
                reference(p, "operator")
                for p in sorted(hook_directory.rglob("*.py"))
                if "__pycache__" not in p.parts
            )
    cmake = (
        reference(local(build.cmake), "operator").locator
        if build.cmake is not None
        else None
    )
    _, arch = package_target(native.target_soc)
    recipe = BuildRecipe(
        recipe_id=build.recipe_id or native.name,
        version=build.version,
        kernel_id=native.kernel_id,
        implementation=build.implementation,
        target_soc=native.target_soc,
        npu_arch=arch,
        source=source,
        sources=(source,),
        headers=headers,
        include_dirs=tuple(
            local(p).relative_to(root).as_posix() for p in build.includes
        ),
        dependencies=build.dependencies,
        definitions=build.definitions,
        compile_options=build.compile_options,
        link_options=build.link_options,
        device_entry=build.device_entry or "fi_" + native.name,
        host_entry=build.host_entry or "aclnn" + build.op_type,
        launch_abi=build.launch_abi or "aclnn." + native.name + ".v2",
        tiling_mode=build.tiling_mode,
        tiling_schema_version=2,
        block_dim=build.block_dim,
        workspace_bytes=build.workspace_bytes,
        task_type=build.task_type,
        sync_policy=build.sync_policy,
        host_sources=host,
        package_vendor=build.package_vendor or "flashinfer_" + native.name,
        package_op_type=build.op_type,
        tensor_parameters=native.tensors,
        shape_constraints=native.shapes,
        tiling_host_entry=build.tiling,
        workspace_host_entry=(
            build.tiling if build.workspace_bytes == "dynamic" else None
        ),
        dtype_signatures=native.dtype_signatures,
        allow_jit_fallback=native.allow_jit_fallback,
        attributes=native.attributes,
        extra_build_files=tuple(extra),
        cmake_entry=cmake,
    )
    descriptor = KernelDescriptor(
        native.kernel_id,
        native.op,
        Backend.ASCENDC,
        capability=native.capability,
        compile_mode=CompileMode.AOT,
        workspace=WorkspaceFormula(constant_bytes=native.workspace_bound),
        launch_abi=KernelLaunchABI(
            recipe.launch_abi,
            recipe.host_entry,
            ("workspace", "workspace_bytes", "executor", "stream"),
        ),
        tiling_schema_version=2,
        source=source,
        target_soc=native.target_soc,
    )
    if native.descriptor_metadata and "kernel_id" in native.descriptor_metadata:
        data = dict(
            native.descriptor_metadata,
            capability=native.capability.to_dict(),
            source=source.to_dict(),
        )
        descriptor = descriptor_from_dict(data)
    elif native.descriptor_metadata:
        from dataclasses import replace

        descriptor = replace(descriptor, **native.descriptor_metadata)
    recipe.validate_descriptor(descriptor)
    from flashinfer._resources import data_root, recipe_root

    return dict(
        descriptor=descriptor,
        recipe=recipe,
        manifest=(data_root(root) / "kernels/registry.json").relative_to(root),
        outputs={
            (recipe_root(root) / (recipe.recipe_id + ".json")).relative_to(root): [
                recipe.to_dict()
            ]
        },
    )
