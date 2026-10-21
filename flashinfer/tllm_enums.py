"""Serialized MoE routing values pinned to upstream 289557772."""

from enum import IntEnum


class RoutingMethodType(IntEnum):
    """Upstream routing identity; membership does not imply kernel support."""

    Default = 0
    Renormalize = 1
    DeepSeekV3 = 2
    Llama4 = 3
    RenormalizeNaive = 4
    TopK = 5
    SigmoidRenorm = 6
    MiniMax2 = 7
    Sigmoid = 8
    TopKSigmoid = 9
    Unspecified = 10

    def __repr__(self) -> str:
        return f"{type(self).__name__}.{self.name}"
