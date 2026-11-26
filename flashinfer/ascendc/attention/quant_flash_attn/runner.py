"""Bind C4A4 AI Core execution and its optional LSE output."""

from dataclasses import replace

from flashinfer.ascendc._runtime import native_admission
from flashinfer.jit.tensor_adapter import package_tensor_runner


def admission(recipe, signature):
    enabled = dict(signature.specializations).get("attribute.return_softmax_lse") == "True"
    if not enabled:
        if signature.shapes[-1:] != ((0,),):
            return ("disabled softmax LSE must have shape (0,)",)
        signature = replace(signature, shapes=signature.shapes[:-1] + ((1,),))
    return native_admission(recipe, signature)


def make_runner(module, spec, recipe):
    native = package_tensor_runner(module, recipe, spec=spec)

    def run(*tensors, **attributes):
        if not attributes["return_softmax_lse"]:
            tensors = tensors[:-1] + (tensors[-1].new_empty((1,)),)
        return native(*tensors, **attributes)

    return run
