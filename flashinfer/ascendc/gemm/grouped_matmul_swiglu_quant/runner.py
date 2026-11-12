"""Adapt the shared logical call only after this Ascend C kernel is selected."""

from dataclasses import dataclass

from flashinfer.jit.tensor_adapter import package_tensor_runner


def admission(recipe, signature):
    from flashinfer.ascendc._runtime import native_admission
    reasons = native_admission(recipe, signature)
    if reasons:
        return reasons
    packed = signature.dtypes[1] == 'int32'
    limit = dict(signature.specializations).get('attribute.swiglu_limit', '0.0')
    try:
        recipe.validate_attributes(dict(is_enable_weight_assistance_matrix=packed,
            dequant_mode=int(packed and len(signature.shapes[2]) == 3), limited=float(limit)))
    except (ValueError, TypeError) as error:
        return (str(error),)
    return ()


@dataclass(frozen=True)
class GroupedMatmulSwigluQuantRunner:
    native: object

    def __call__(self, x, weight, weight_scale, x_scale, assistance, group_list,
                 output, output_scale, *, swiglu_limit=0.0):
        packed = str(weight.dtype).removeprefix('torch.') == 'int32'
        attributes = dict(is_enable_weight_assistance_matrix=packed,
                          dequant_mode=int(packed and weight_scale.ndim == 3),
                          limited=swiglu_limit)
        return self.native(x, weight, weight_scale, x_scale, assistance, group_list,
                           output, output_scale, **attributes)


def make_runner(module, spec, recipe):
    return GroupedMatmulSwigluQuantRunner(package_tensor_runner(module, recipe, spec=spec))
