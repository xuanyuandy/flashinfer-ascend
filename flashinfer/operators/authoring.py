"""Source-only discovery and deterministic catalog/build-metadata generation."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

from flashinfer.runtime import Backend, SchemaError
from .definition import OperatorDefinition
from .implementation import ImplementationDefinition


def _json(value):
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def _load(path):
    key = hashlib.sha256(str(path.resolve()).encode() + path.read_bytes()).hexdigest()[
        :24
    ]
    package = "flashinfer.operators._author_" + key
    parent = types.ModuleType(package)
    parent.__path__ = [str(path.parent)]
    sys.modules[package] = parent
    spec = importlib.util.spec_from_file_location(package + "." + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def discover(source_root):
    root = Path(source_root).resolve()
    result, seen = [], set()
    for path in sorted((root / "flashinfer/operators").glob("*/*/operator.py")):
        operator = _load(path).OPERATOR
        if (
            not isinstance(operator, OperatorDefinition)
            or operator.name != path.parent.name
            or operator.domain != path.parent.parent.name
            or operator.op in seen
        ):
            raise SchemaError(
                "logical operator directory/identity must be unique and match its declaration"
            )
        seen.add(operator.op)
        result.append((path.parent, operator))
    return tuple(result)


def discover_implementations(source_root):
    root = Path(source_root).resolve()
    result, seen = [], set()
    for backend in Backend:
        for path in sorted(
            (root / "flashinfer" / backend.value).glob("*/*/implementation.py")
        ):
            module = _load(path)
            declarations = getattr(module, "IMPLEMENTATIONS", None)
            if declarations is None:
                declarations = (module.IMPLEMENTATION,)
            for declaration in declarations:
                if (
                    not isinstance(declaration, ImplementationDefinition)
                    or declaration.backend != backend
                    or declaration.kernel_id in seen
                ):
                    raise SchemaError("duplicate or invalid implementation identity")
                seen.add(declaration.kernel_id)
                result.append((path.parent, module, declaration))
    return tuple(result)


def _runtime_files(root, directory):
    paths = set(directory.rglob("*.py"))
    # Shared primitives are authoritative inputs too, rather than untracked imports.
    for parent in (directory.parent, directory.parent.parent):
        paths.update(parent.glob("_*.py"))
    for domain in (root / "flashinfer/operators" / directory.parent.name,):
        if domain.is_dir():
            paths.update(domain.glob("_*.py"))
    return {
        p.relative_to(root / "flashinfer")
        .as_posix(): hashlib.sha256(p.read_bytes())
        .hexdigest()
        for p in sorted(paths)
        if "__pycache__" not in p.parts and p.is_file() and p.name != "__main__.py"
    }


def operator_record(root, directory, operator):
    return dict(
        name=operator.name,
        domain=operator.domain,
        op=operator.op,
        module=".".join(
            (directory / "operator.py").relative_to(root).with_suffix("").parts
        ),
        export=bool(operator.schema and operator.public_entry is None),
        cases=(directory / "cases.py").is_file(),
        schema=operator.schema,
        output_names=list(operator.output_names),
        description=operator.description,
        runtime_files=_runtime_files(root, directory),
    )


def implementation_record(root, directory, declaration):
    return dict(
        kernel_id=declaration.kernel_id,
        op=declaration.op,
        backend=declaration.backend.value,
        module=".".join(
            (directory / "implementation.py").relative_to(root).with_suffix("").parts
        ),
        scope=declaration.scope,
        origin=declaration.origin,
        files=_runtime_files(root, directory),
    )


def generate(*, source_root=None, kernel_ids=(), check=False):
    """Generate delivery metadata without editing tests, freezing or compiling."""
    root = Path(source_root or Path(__file__).resolve().parents[2]).resolve()
    logical = discover(root)
    implementations = discover_implementations(root)
    logical_by_op = {o.op: (d, o) for d, o in logical}
    selected = {
        d.kernel_id
        for _, _, d in implementations
        if not kernel_ids or d.kernel_id in kernel_ids
    }
    if set(kernel_ids) - selected:
        raise SchemaError("requested kernel is absent from implementation directories")
    from flashinfer._resources import data_root

    operator_path = data_root(root) / "operators.json"
    implementation_path = data_root(root) / "implementations.json"
    for path, key, actual in (
        (operator_path, "operators", set(logical_by_op)),
        (
            implementation_path,
            "implementations",
            {d.kernel_id for _, _, d in implementations},
        ),
    ):
        if path.exists():
            previous = json.loads(path.read_text())[key]
            required = {
                r["op" if key == "operators" else "kernel_id"] for r in previous
            }
            if required - actual:
                raise SchemaError(
                    "managed operator source was removed; explicitly review its release inventory"
                )
    if any(d.op not in logical_by_op for _, _, d in implementations):
        raise SchemaError(
            "implementation requires an independent logical operator definition"
        )
    owned = set()
    if implementation_path.exists():
        owned = {
            r["kernel_id"]
            for r in json.loads(implementation_path.read_text())["implementations"]
        }
    outputs, operator_records, implementation_records, manifests = {}, [], [], {}
    for directory, operator in logical:
        exported = bool(operator.schema and operator.public_entry is None)
        cases = (directory / "cases.py").is_file()
        if exported and not cases:
            raise SchemaError(
                "functional operator requires cases.py with independent numerical samples"
            )
        operator_records.append(operator_record(root, directory, operator))
    prepared_modules = set()
    for directory, module, declaration in implementations:
        provider = ".".join(
            (directory / "implementation.py").relative_to(root).with_suffix("").parts
        )
        implementation_records.append(
            implementation_record(root, directory, declaration)
        )
        prepare = getattr(module, "prepare_outputs", None)
        prepare_key = (provider, None)
        if prepare is None:
            prepare = getattr(declaration, "prepare_outputs", None)
            prepare_key = (provider, declaration.kernel_id)
        if (
            declaration.kernel_id not in selected
            or prepare is None
            or prepare_key in prepared_modules
        ):
            continue
        prepared_modules.add(prepare_key)
        prepared = prepare(root)
        for item in prepared if isinstance(prepared, (tuple, list)) else (prepared,):
            descriptor = item["descriptor"]
            owner = next(
                (
                    d
                    for _, _, d in implementations
                    if d.kernel_id == descriptor.kernel_id
                ),
                None,
            )
            if (
                owner is None
                or descriptor.op != owner.op
                or descriptor.backend != owner.backend
            ):
                raise SchemaError(
                    "generated descriptor differs from its implementation identity"
                )
            if descriptor.kernel_id not in selected:
                continue
            manifest = root / item["manifest"]
            if manifest not in manifests:
                payload = (
                    json.loads(manifest.read_text())
                    if manifest.exists()
                    else dict(
                        schema_version=3,
                        generated_at="generated from Python declarations",
                        kernels=[],
                    )
                )
                manifests[manifest] = (
                    payload,
                    {r["kernel_id"]: r for r in payload["kernels"]},
                )
            current_records = manifests[manifest][1]
            if (
                descriptor.kernel_id not in owned
                and descriptor.kernel_id in current_records
                and current_records[descriptor.kernel_id] != descriptor.to_dict()
            ):
                raise SchemaError(
                    "author directory cannot overwrite an unrelated production descriptor"
                )
            current_records[descriptor.kernel_id] = descriptor.to_dict()
            for relative, value in item["outputs"].items():
                target = root / relative
                from flashinfer._resources import generated_root

                if not target.resolve().is_relative_to(
                    root
                ) and not target.resolve().is_relative_to(
                    generated_root(root).resolve()
                ):
                    raise SchemaError("generated output escapes source root")
                content = _json(value)
                if target in outputs and outputs[target] != content:
                    raise SchemaError(
                        "implementations produced conflicting generated files"
                    )
                outputs[target] = content
    for path, (payload, records) in manifests.items():
        payload["kernels"] = list(records.values())
        outputs[path] = _json(payload)
    outputs[operator_path] = _json(dict(schema_version=2, operators=operator_records))
    outputs[implementation_path] = _json(
        dict(schema_version=1, implementations=implementation_records)
    )
    differences = [
        p
        for p, content in outputs.items()
        if not p.exists() or p.read_text() != content
    ]
    if check and differences:
        raise SchemaError(
            "generated operator files are stale: "
            + ", ".join(str(p.relative_to(root)) for p in differences)
        )
    if not check:
        # Validate the complete batch above before publishing any output.
        for path in differences:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(outputs[path])
        from . import catalog
        from .implementation import implementation_entries

        catalog.entries.cache_clear()
        catalog.definition.cache_clear()
        implementation_entries.cache_clear()
    return tuple(sorted(selected))
