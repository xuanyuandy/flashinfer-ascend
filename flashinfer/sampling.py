"""FlashInfer sampling frontends; implementations are selected by the registry."""

from __future__ import annotations

from typing import Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from .runtime import build_signature, current_hardware, dispatch, get_global_registry


def _call(name, tensor, *args, **kwargs):
    signature = build_signature("sampling." + name, (tensor,), layouts=("ND",))
    decision = dispatch(signature, current_hardware(), backend=None)
    return get_global_registry().get(decision.kernel_id).runner(tensor, *args, **kwargs)

def sampling_from_logits(
    logits: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor:
    return _call(
        'sampling_from_logits',
        logits, indices, deterministic, generator,
        check_nan, seed, offset,
    )


def sampling_from_probs(
    probs: torch.Tensor,
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    return _call(
        'sampling_from_probs',
        probs, indices, deterministic, generator,
        check_nan, seed, offset, return_valid,
    )


def top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    return _call(
        'top_p_sampling_from_probs',
        probs, top_p, indices, deterministic,
        generator, check_nan, seed, offset,
        return_valid,
    )


def top_k_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    return _call(
        'top_k_sampling_from_probs',
        probs, top_k, indices, deterministic,
        generator, check_nan, seed, offset,
        return_valid,
    )


def min_p_sampling_from_probs(
    probs: torch.Tensor,
    min_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    return _call(
        'min_p_sampling_from_probs',
        probs, min_p, indices, deterministic,
        generator, check_nan, seed, offset,
        return_valid,
    )


def top_k_top_p_sampling_from_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    top_p: Union[torch.Tensor, float],
    indices: Optional[torch.Tensor] = None,
    filter_apply_order: str = "top_k_first",
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    check_nan: bool = False,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
    return_valid: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    return _call(
        'top_k_top_p_sampling_from_probs',
        probs, top_k, top_p, indices,
        filter_apply_order, deterministic, generator, check_nan,
        seed, offset, return_valid,
    )


def top_p_renorm_probs(
    probs: torch.Tensor,
    top_p: Union[torch.Tensor, float],
    is_deterministic: bool = False,
) -> torch.Tensor:
    return _call(
        'top_p_renorm_probs',
        probs, top_p, is_deterministic,
    )


def top_k_renorm_probs(
    probs: torch.Tensor,
    top_k: Union[torch.Tensor, int],
    is_deterministic: bool = False,
) -> torch.Tensor:
    return _call(
        'top_k_renorm_probs',
        probs, top_k, is_deterministic,
    )


def top_k_mask_logits(
    logits: torch.Tensor, top_k: Union[torch.Tensor, int]
) -> torch.Tensor:
    return _call(
        'top_k_mask_logits',
        logits, top_k,
    )


def chain_speculative_sampling(
    draft_probs,
    draft_token_ids,
    target_probs,
    maybe_output_accepted_token_num: Optional[torch.Tensor] = None,
    maybe_output_emitted_draft_token_num: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    generator: Optional[torch.Generator] = None,
    seed: Optional[Union[int, torch.Tensor]] = None,
    offset: Optional[Union[int, torch.Tensor]] = None,
) -> torch.Tensor:
    return _call(
        'chain_speculative_sampling',
        draft_probs, draft_token_ids, target_probs, maybe_output_accepted_token_num,
        maybe_output_emitted_draft_token_num, deterministic, generator, seed,
        offset,
    )


# Definitions and implementation choices are collected by the framework.
from flashinfer.operators.catalog import install_exports as _install_operator_exports
_install_operator_exports(globals(), domain='sampling')
