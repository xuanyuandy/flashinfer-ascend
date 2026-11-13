"""Map disabled public auxiliary outputs to native buffers after selection."""

from dataclasses import dataclass, replace

from flashinfer.jit.tensor_adapter import package_tensor_runner


def admission(recipe, signature):
    from flashinfer.ascendc._runtime import native_admission
    attrs = dict(signature.specializations)
    enabled = attrs.get("attribute.return_softmax_lse") == "True"
    if not enabled:
        if signature.shapes[-2:] != ((0,), (0,)):
            return ("disabled softmax outputs must have shape (0,)",)
        signature = replace(signature, shapes=signature.shapes[:-2] + ((1,), (1,)))
    reasons = native_admission(recipe, signature)
    if reasons:
        return reasons
    try:
        values = {}
        for parameter in recipe.attributes:
            value = attrs["attribute." + parameter.name]
            values[parameter.name] = (
                value == "True" if parameter.dtype == "bool" else
                int(value) if parameter.dtype == "int64" else
                float(value) if parameter.dtype == "float64" else value
            )
        recipe.validate_attributes(values)
    except (KeyError, ValueError, TypeError) as error:
        return (str(error),)
    return ()


@dataclass(frozen=True)
class SparseFlashAttentionRunner:
    native: object

    def __call__(self, query, key, value, indices, table, lengths_q, lengths_kv,
                 query_rope, key_rope, output, softmax_max, softmax_sum, **attributes):
        if not attributes["return_softmax_lse"]:
            # The upstream kernel never writes these two buffers when disabled.
            # Supplied public out tensors stay empty and preserve their identity.
            softmax_max = softmax_max.new_empty((1,))
            softmax_sum = softmax_sum.new_empty((1,))
        return self.native(query, key, value, indices, table, lengths_q, lengths_kv,
                           query_rope, key_rope, output, softmax_max, softmax_sum, **attributes)


def make_runner(module, spec, recipe):
    return SparseFlashAttentionRunner(package_tensor_runner(module, recipe, spec=spec))
