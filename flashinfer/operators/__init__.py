"""Backend-neutral logical contracts and implementation declarations."""
from .definition import Invocation, OperatorDefinition, OutputSpec
from .implementation import ImplementationDefinition

__all__ = ["Invocation", "OperatorDefinition", "OutputSpec", "ImplementationDefinition"]

def __getattr__(name):
    if name == "BuildConfig":
        from flashinfer.ascendc.definition import BuildConfig
        return BuildConfig
    raise AttributeError(name)
