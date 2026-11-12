"""Backend-neutral implementation declarations and installed catalog loading."""

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path

from flashinfer.runtime.schema import Backend, KernelDescriptor, SchemaError


@dataclass(frozen=True)
class ImplementationDefinition:
    """One kernel identity and its lazy descriptor/runner binding factory."""

    kernel_id: str
    op: str
    backend: Backend
    factory: object
    scope: str = "global"
    origin: str = ""

    def __post_init__(self):
        object.__setattr__(self, "backend", Backend(self.backend))
        if not self.kernel_id or not self.op or not callable(self.factory):
            raise SchemaError("implementation requires an identity, logical op and binding factory")
        if self.scope not in {"global", "experimental"}:
            raise SchemaError("invalid implementation registration scope")

    def bind(self):
        descriptor = self.factory()
        if (not isinstance(descriptor, KernelDescriptor)
                or (descriptor.kernel_id, descriptor.op, descriptor.backend)
                != (self.kernel_id, self.op, self.backend)):
            raise SchemaError("implementation factory changed its declared identity: " + self.kernel_id)
        if not callable(descriptor.runner):
            raise SchemaError("executable implementation requires a runner: " + self.kernel_id)
        return descriptor


@lru_cache(maxsize=1)
def implementation_entries():
    path = Path(__file__).resolve().parents[1] / "data/implementations.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"schema_version", "implementations"} or payload["schema_version"] != 1:
        raise SchemaError("invalid installed implementation catalog")
    records = tuple(payload["implementations"])
    seen = set()
    for record in records:
        if (set(record) != {"kernel_id", "op", "backend", "module", "scope", "origin", "files"}
                or record["kernel_id"] in seen or not isinstance(record["files"], dict)
                or not record["module"].startswith("flashinfer." + record["backend"] + ".")
                or not record["module"].endswith(".implementation")):
            raise SchemaError("invalid or duplicate installed implementation record")
        seen.add(record["kernel_id"])
    return records


def implementation_definition(record):
    root = Path(__file__).resolve().parents[1]
    for relative, digest in record["files"].items():
        path = root / relative
        if (path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
                or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest):
            raise SchemaError("implementation source digest mismatch: " + relative)
    module = importlib.import_module(record["module"])
    declarations = getattr(module, "IMPLEMENTATIONS", None)
    declarations = (module.IMPLEMENTATION,) if declarations is None else declarations
    declaration = next((d for d in declarations if d.kernel_id == record["kernel_id"]), None)
    if (not isinstance(declaration, ImplementationDefinition)
            or (declaration.kernel_id, declaration.op, declaration.backend.value,
                declaration.scope, declaration.origin)
            != tuple(record[key] for key in ("kernel_id", "op", "backend", "scope", "origin"))):
        raise SchemaError("implementation identity differs from installed catalog")
    return declaration


def implementations(*, backend=None, domain=None, scope="global", origin=None):
    backend = None if backend is None else Backend(backend).value
    return tuple(implementation_definition(record) for record in implementation_entries()
                 if (backend is None or record["backend"] == backend)
                 and (domain is None or record["op"].partition(".")[0] == domain)
                 and (scope is None or record["scope"] == scope)
                 and (origin is None or record["origin"] == origin))


def implementation_descriptors(**selection):
    return tuple(declaration.bind() for declaration in implementations(**selection))
