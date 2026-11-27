"""Native admission and runner for the arch22 lightning indexer."""

from dataclasses import dataclass

from flashinfer.jit.tensor_adapter import package_tensor_runner


def admission(recipe, signature):
    from flashinfer.ascendc._runtime import native_admission

    reasons = list(native_admission(recipe, signature))
    if reasons:
        return tuple(reasons)
    specializations = dict(signature.specializations)
    try:
        values = {}
        for parameter in recipe.attributes:
            # The public interface names the native block_len attribute kv_block_len.
            source = ("attribute.kv_block_len" if parameter.name == "block_len"
                      else "attribute." + parameter.name)
            value = specializations[source]
            values[parameter.name] = (
                value == "True" if parameter.dtype == "bool" else
                int(value) if parameter.dtype == "int64" else value
            )
        recipe.validate_attributes(values)
    except (KeyError, ValueError, TypeError) as error:
        return (str(error),)
    return ()


@dataclass(frozen=True)
class LightningIndexerRunner:
    native: object

    def __call__(self, query, key, weights, cur_seq_lengths_query, cur_seq_lengths_key,
                 block_table, sparse_indices, sparse_values, **attributes):
        import torch

        def _lengths(tensor):
            if tensor is not None and tensor.dtype != torch.int64:
                tensor = tensor.to(torch.int64)
            return tensor

        # Public interface names kv_block_len; the native OpDef attribute is block_len.
        attributes = dict(attributes)
        attributes["block_len"] = int(attributes.pop("kv_block_len"))
        # The kernel leaves sparse_values unwritten when return_value is unset;
        # the public contract keeps that buffer deterministic (all zeros).
        if not attributes.get("return_value", False):
            sparse_values.zero_()
        return self.native(
            query,
            key,
            weights,
            _lengths(cur_seq_lengths_query),
            _lengths(cur_seq_lengths_key),
            block_table,
            sparse_indices,
            sparse_values,
            **attributes,
        )


def make_runner(module, spec, recipe):
    return LightningIndexerRunner(package_tensor_runner(module, recipe, spec=spec))
