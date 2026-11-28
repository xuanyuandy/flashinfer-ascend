"""Native admission and Tensor handoff for the rich K mutation ABI."""

from dataclasses import dataclass

from flashinfer.jit.tensor_adapter import package_tensor_runner


def admission(recipe, signature):
    import math

    names = recipe.tensor_parameter_names
    if len(signature.shapes) != len(names) or len(signature.dtypes) != len(names):
        return ("rich K implementation requires its complete 10-Tensor signature",)
    try:
        recipe.validate_tensor_metadata(
            shapes=dict(zip(names, signature.shapes)),
            dtypes=dict(zip(names, signature.dtypes)),
        )
    except (ValueError, TypeError) as error:
        return (str(error),)
    if any(layout and layout != "ND" for layout in signature.physical_layouts):
        return ("rich K cache Tensors require physical ND",)
    attrs = dict(signature.specializations)
    try:
        norm_eps = float(attrs["attribute.norm_eps"])
    except (KeyError, TypeError, ValueError):
        return ("rich K requires a finite positive norm_eps",)
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        return ("rich K requires a finite positive norm_eps",)
    try:
        recipe.validate_attributes({"norm_eps": norm_eps})
    except (ValueError, TypeError) as error:
        return (str(error),)
    return ()


@dataclass(frozen=True)
class IndexerPrologueKRunner:
    native: object

    def __call__(self, latent, wk, norm_weight, rope_cos, rope_sin, indices,
                 k_cache, k_scale_cache, k_cache_out, k_scale_cache_out, *, norm_eps):
        return self.native(
            latent, wk, norm_weight, rope_cos, rope_sin, indices,
            k_cache, k_scale_cache, k_cache_out, k_scale_cache_out,
            norm_eps=norm_eps,
        )


def make_runner(module, spec, recipe):
    return IndexerPrologueKRunner(package_tensor_runner(module, recipe, spec=spec))
