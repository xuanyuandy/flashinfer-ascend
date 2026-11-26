"""One author-owned interface/semantic contract, shared by generation and runtime."""

import ast
from dataclasses import dataclass
import inspect
import re

from flashinfer.runtime import SchemaError


def schema_arguments(schema):
    if (
        not isinstance(schema, str)
        or not schema.startswith("(")
        or ") -> " not in schema
    ):
        raise SchemaError("operator requires an unnamed functional Torch schema")
    arguments, result = schema[1:].split(") -> ", 1)
    declarations, keyword_only = [], False
    for value in arguments.split(","):
        value = value.strip()
        if value == "*":
            keyword_only = True
            continue
        match = re.fullmatch(
            r"(Tensor\??|float|int\??|bool|str\??) ([a-z][a-z0-9_]*)(?:=(.+))?", value
        )
        if match is None:
            raise SchemaError("unsupported author schema argument: " + value)
        dtype, name, default = match.groups()
        try:
            default = (
                inspect.Parameter.empty
                if default is None
                else ast.literal_eval(default)
            )
        except (ValueError, SyntaxError) as error:
            raise SchemaError("operator defaults must be scalar literals") from error
        declarations.append((dtype, name, default, keyword_only))
    if result != "Tensor" and not re.fullmatch(r"\(Tensor(?:, Tensor)+\)", result):
        raise SchemaError("operator returns must be one or more fresh Tensors")
    return tuple(declarations), result


def signature(declarations):
    return inspect.Signature(
        inspect.Parameter(
            name,
            (
                inspect.Parameter.KEYWORD_ONLY
                if keyword
                else inspect.Parameter.POSITIONAL_OR_KEYWORD
            ),
            default=default,
            annotation=dtype,
        )
        for dtype, name, default, keyword in declarations
    )


@dataclass(frozen=True)
class OutputSpec:
    shape: tuple
    dtype: str
    device: object


@dataclass(frozen=True)
class Invocation:
    """Semantic hook result. The framework owns selection, allocation and launch."""

    inputs: tuple
    outputs: tuple
    attributes: dict
    inference: object
    specializations: tuple = ()


@dataclass(frozen=True)
class OperatorDefinition:
    """One logical contract, independent of backend, native ABI and build mode."""

    name: str
    domain: str
    op: str
    schema: str = ""
    output_names: tuple = ()
    infer: object = None
    validate_values: object = None
    public_entry: object = None
    description: str = ""
    tensor_shapes: tuple = ()
    torch_declarations: object = None

    def __post_init__(self):
        if (
            not re.fullmatch(r"[a-z][a-z0-9_]*", self.name)
            or not re.fullmatch(r"[a-z][a-z0-9_]*", self.domain)
            or not isinstance(self.op, str)
            or not self.op
            or self.infer is not None
            and not callable(self.infer)
            or self.validate_values is not None
            and not callable(self.validate_values)
        ):
            raise SchemaError("invalid logical operator definition")
        if self.torch_declarations is not None and (
            self.schema or not self.public_entry or not callable(self.torch_declarations)
        ):
            raise SchemaError("custom Torch declarations require a public entry and cannot also generate a schema")
        if self.schema:
            declarations, returns = schema_arguments(self.schema)
            names = [name for _, name, _, _ in declarations]
            if (
                len(names) != len(set(names))
                or set(names) & {"out", "backend", "kernel_id"}
                or len(self.output_names) != returns.count("Tensor")
                or len(set(self.output_names)) != len(self.output_names)
                or set(names) & set(self.output_names)
                or not callable(self.infer)
            ):
                raise SchemaError("operator schema/output names are inconsistent")
            signature(declarations)

    @property
    def arguments(self):
        return schema_arguments(self.schema)[0]

    @property
    def public_signature(self):
        if not self.schema:
            if self.public_entry is None:
                raise SchemaError(
                    "internal operation has no direct public callable: " + self.op
                )
            import importlib

            module, attribute = self.public_entry.split(":", 1)
            return inspect.signature(
                getattr(importlib.import_module(module), attribute)
            )
        return signature(
            self.arguments
            + (
                ("object", "out", None, True),
                ("str?", "backend", None, True),
                ("str?", "kernel_id", None, True),
            )
        )

    def torch_schema(self, *, out=False):
        # Torch custom_op cannot represent keyword-only optional Tensor inputs.
        # Preserve the conventional Torch ABI for optional scalar attributes
        # as well; the Python facade retains the author's keyword-only API.
        positional = [
            p
            for p in self.arguments
            if not p[3]
            or p[0].startswith("Tensor")
            or p[2] is not inspect.Parameter.empty
        ]
        # Required Tensor inputs moved from keyword-only must precede defaults
        # in the Torch schema, even when the public signature has positional defaults.
        positional.sort(key=lambda parameter: parameter[2] is not inspect.Parameter.empty)
        keywords = [
            p
            for p in self.arguments
            if p[3]
            and not p[0].startswith("Tensor")
            and p[2] is inspect.Parameter.empty
        ]
        if out:
            split = next(
                (
                    i
                    for i, p in enumerate(positional)
                    if p[2] is not inspect.Parameter.empty
                ),
                len(positional),
            )
            positional[split:split] = [
                ("Tensor(%s!)" % chr(97 + i), name, inspect.Parameter.empty, False)
                for i, name in enumerate(self.output_names)
            ]
        keywords += [("str?", "backend", None, True), ("str?", "kernel_id", None, True)]

        def declaration(p):
            dtype, name, default, _ = p
            return (
                dtype
                + " "
                + name
                + ("" if default is inspect.Parameter.empty else "=" + repr(default))
            )

        result = "()" if out else schema_arguments(self.schema)[1]
        text = (
            "("
            + ", ".join(
                [declaration(p) for p in positional]
                + ["*"]
                + [declaration(p) for p in keywords]
            )
            + ") -> "
            + result
        )
        # Strip mutation syntax only for Python argument binding.
        binding = tuple(
            (re.sub(r"\([a-z]!\)", "", p[0]), p[1], p[2], False) for p in positional
        ) + tuple(keywords)
        return text, signature(binding)

    def bind(self, *args, **kwargs):
        bound = self.public_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return dict(bound.arguments)

    def metadata(self, *args, check_layouts=False, **kwargs):
        from .runtime import metadata

        tensors, invocation = metadata(
            self.name, self.bind(*args, **kwargs), check_layouts=check_layouts
        )
        return tensors, invocation.attributes, invocation.inference
