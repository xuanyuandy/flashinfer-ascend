"""Explicit additions to the Tensor-only ACLNN contract; no runtime imports."""

from dataclasses import dataclass
import math

from flashinfer.runtime import SchemaError


@dataclass(frozen=True)
class TensorDtypeConversion:
    """A declared storage representation, never an implicit numerical cast."""

    storage_dtype: str
    native_dtype: str
    element_multiplier: int = 1

    def __post_init__(self):
        if (self.storage_dtype, self.native_dtype, self.element_multiplier) not in {
            ("int32", "int4", 8), ("int64", "uint64", 1),
        }:
            raise SchemaError("unsupported explicit ACLNN storage representation")


@dataclass(frozen=True)
class AttributeParameter:
    """One generated ACLNN attribute, in OpDef declaration order."""

    name: str
    dtype: str
    default: object
    minimum: object = None
    maximum: object = None

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name.isidentifier()
                or self.dtype not in {"bool", "int64", "float64"}):
            raise SchemaError("invalid ACLNN attribute declaration")
        if self.dtype == "bool" and (self.minimum is not None or self.maximum is not None):
            raise SchemaError("boolean ACLNN attributes cannot declare numeric bounds")
        for bound in (self.minimum, self.maximum):
            if bound is not None and (type(bound) not in (int, float) or not math.isfinite(bound)):
                raise SchemaError("ACLNN attribute bounds must be finite numbers")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise SchemaError("ACLNN attribute bounds are empty")
        self.validate(self.default)

    def validate(self, value):
        valid = (type(value) is bool if self.dtype == "bool" else
                 type(value) is int and -(2**63) <= value < 2**63 if self.dtype == "int64" else
                 type(value) in (int, float) and math.isfinite(value))
        if (not valid or self.minimum is not None and value < self.minimum
                or self.maximum is not None and value > self.maximum):
            raise SchemaError("invalid ACLNN attribute %s=%r" % (self.name, value))
        return value
