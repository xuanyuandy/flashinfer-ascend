"""Backend-neutral logical contracts and implementation declarations."""

from .definition import Invocation, OperatorDefinition, OutputSpec
from .implementation import ImplementationDefinition

__all__ = [
    "Invocation",
    "OperatorDefinition",
    "OutputSpec",
    "ImplementationDefinition",
    "Support",
    "DimRange",
    "PythonKernel",
    "CANNImplementation",
    "TritonImplementation",
    "ReferenceImplementation",
]


def __getattr__(name):
    if name in {
        "Support",
        "DimRange",
        "PythonKernel",
        "CANNImplementation",
        "TritonImplementation",
        "ReferenceImplementation",
    }:
        from . import author_api

        return getattr(author_api, name)
    if name == "BuildConfig":
        from flashinfer.ascendc.definition import BuildConfig

        return BuildConfig
    raise AttributeError(name)
