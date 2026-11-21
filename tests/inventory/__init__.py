"""Independent review records for individual operators."""

from dataclasses import dataclass
from typing import Optional, Tuple

from flashinfer.runtime import CapabilitySpec, InputSignature


@dataclass(frozen=True)
class ReviewedKernel:
    kernel_id: str
    capability: CapabilitySpec
    backend: str
    native_recipe: Optional[Tuple[str, str]] = None  # recipe ID and implementation
    source: Optional[Tuple[str, str]] = None  # language and SoC of a non-native source
    smoke_signature: Optional[InputSignature] = None  # for constrained host dispatch
    scope: str = "global"


def shape_ranges(axes, bounds):
    from flashinfer.runtime import ShapeRange

    return tuple(ShapeRange(name, tensor, axis, *bounds[name])
                 for tensor, shape in enumerate(axes) for axis, name in enumerate(shape))
