"""Sampling semantics shared by the explicit reference and CANN composites.

CANN's native top-k/top-p cuts ties by index. FlashInfer retains the entire
boundary probability group, so filtering is expressed with tensor operators
before invoking the native sampler with its own filters disabled.
"""
from functools import partial
OP_NAMES = ('sampling_from_probs', 'sampling_from_logits', 'top_p_sampling_from_probs', 'top_k_sampling_from_probs', 'top_k_top_p_sampling_from_probs', 'min_p_sampling_from_probs', 'top_p_renorm_probs', 'top_k_renorm_probs', 'top_k_mask_logits', 'chain_speculative_sampling')

def _matrix(x, *, logits=False):
    import torch
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError('sampling input must have shape [batch_size, vocab_size > 0]')
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError('sampling supports float16, bfloat16 and float32')
    if not logits and bool(((x < 0) | ~torch.isfinite(x)).any()):
        raise ValueError('probabilities must be finite and nonnegative')
    return x.float()

def _parameter(value, x, name, *, integer=False):
    import torch
    dtype = torch.int64 if integer else torch.float32
    if isinstance(value, torch.Tensor):
        if value.device != x.device or value.ndim != 1 or value.shape[0] != x.shape[0]:
            raise ValueError(f'{name} must have one value per input distribution on the same device')
        if integer and value.dtype not in (torch.int32, torch.int64):
            raise ValueError(f'{name} must contain integers')
        result = value.to(dtype)
    else:
        if integer and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f'{name} must be an integer')
        result = torch.full((x.shape[0],), value, dtype=dtype, device=x.device)
    if integer:
        if bool(((result <= 0) | (result > x.shape[1])).any()):
            raise ValueError(f'{name} must be in [1, vocab_size]')
    elif bool(((result < 0) | (result > 1) | ~torch.isfinite(result)).any()):
        raise ValueError(f'{name} must be finite and in [0, 1]')
    return result

def _top_k(x, value, fill=0.0):
    import torch
    k = _parameter(value, x, 'top_k', integer=True)
    ordered = torch.sort(x, dim=-1, descending=True).values
    threshold = ordered.gather(1, (k - 1).unsqueeze(1))
    return torch.where(x >= threshold, x, fill)

def _top_p(x, value):
    import torch
    p = _parameter(value, x, 'top_p')
    ordered = torch.sort(x, dim=-1, descending=True).values
    cumulative = ordered.cumsum(-1)
    index = (cumulative < p.unsqueeze(1)).sum(-1).clamp(max=x.shape[1] - 1)
    threshold = ordered.gather(1, index.unsqueeze(1))
    keep = (x >= threshold) | (p.unsqueeze(1) >= 1)
    return torch.where(keep, x, 0)

def _normalize(x):
    import torch
    total = x.sum(-1, keepdim=True)
    return x / torch.where(total > 1e-08, total, torch.ones_like(total))

def _rng(seed, offset):
    if seed is not None or offset is not None:
        raise NotImplementedError('CUDA Philox seed/offset row streams are not implemented on NPU; use a device-local torch.Generator (see sampling parity ledger)')

def _draw(probs, generator):
    import torch
    valid = torch.isfinite(probs).all(-1) & (probs >= 0).all(-1) & (probs.sum(-1) > 0)
    safe = torch.where(valid.unsqueeze(1), probs, torch.zeros_like(probs))
    safe[:, 0] = torch.where(valid, safe[:, 0], torch.ones_like(safe[:, 0]))
    if probs.shape[0] == 0:
        return (torch.empty(0, dtype=torch.int32, device=probs.device), valid)
    result = torch.multinomial(safe, 1, generator=generator).squeeze(1).to(torch.int32)
    return (torch.where(valid, result, torch.zeros_like(result)), valid)

def _chain(draft, ids, target, accepted_out=None, emitted_out=None, deterministic=True, generator=None, seed=None, offset=None):
    import torch
    _rng(seed, offset)
    if draft.ndim != 3 or target.ndim != 3:
        raise ValueError('draft_probs and target_probs must be 3D')
    (b, n, v) = draft.shape
    if ids.shape != (b, n) or target.shape != (b, n + 1, v):
        raise ValueError('speculative sampling shape mismatch')
    if ids.device != draft.device or target.device != draft.device:
        raise ValueError('all speculative tensors must be on the same device')
    if ids.dtype not in (torch.int32, torch.int64) or bool(((ids < 0) | (ids >= v)).any()):
        raise ValueError('draft_token_ids must contain valid integer vocabulary indices')
    _matrix(draft.reshape(-1, v))
    _matrix(target.reshape(-1, v))
    for counter in (accepted_out, emitted_out):
        if counter is not None and (counter.shape != (b,) or counter.dtype != torch.int32 or counter.device != draft.device):
            raise ValueError('output counters must be int32 [batch_size] on the input device')
    (draft, target) = (draft.float(), target.float())
    gather_ids = ids.long().unsqueeze(-1)
    q = draft.gather(-1, gather_ids).squeeze(-1)
    p = target[:, :n].gather(-1, gather_ids).squeeze(-1)
    uniform = torch.rand((b, n), device=draft.device, generator=generator)
    accepted = uniform * q < p
    accepted_count = accepted.sum(-1).to(torch.int32)
    emitted = accepted.to(torch.int32).cumprod(-1).sum(-1).to(torch.int32)
    batch = torch.arange(b, device=draft.device)
    next_target = target[batch, emitted.long()]
    if n:
        next_draft = draft[batch, emitted.long().clamp(max=n - 1)]
        residual = (next_target - next_draft).clamp_min(0)
        distribution = torch.where((emitted == n).unsqueeze(1), next_target, residual)
    else:
        distribution = next_target
    (bonus, valid) = _draw(distribution, generator)
    if not bool(valid.all()):
        raise ValueError('speculative resampling distribution is degenerate')
    output = torch.full((b, n + 1), -1, dtype=torch.int32, device=draft.device)
    if n:
        output[:, :n] = torch.where(torch.arange(n, device=draft.device) < emitted.unsqueeze(1), ids.int(), -1)
    output.scatter_(1, emitted.long().unsqueeze(1), bonus.unsqueeze(1))
    if accepted_out is None:
        accepted_out = torch.zeros(b, dtype=torch.int32, device=draft.device)
    if emitted_out is None:
        emitted_out = torch.zeros_like(accepted_out)
    accepted_out.add_(accepted_count)
    emitted_out.add_(emitted)
    return (output, accepted_out, emitted_out)

def execute(name, *args, **kwargs):
    import inspect
    import torch
    from flashinfer import sampling
    bound = inspect.signature(getattr(sampling, name)).bind(*args, **kwargs)
    bound.apply_defaults()
    options = dict(bound.arguments)
    first = options.pop(next(iter(options)))
    if name == 'chain_speculative_sampling':
        return _chain(first, options['draft_token_ids'], options['target_probs'], options['maybe_output_accepted_token_num'], options['maybe_output_emitted_draft_token_num'], options['deterministic'], options['generator'], options['seed'], options['offset'])
    is_logits = name in ('sampling_from_logits', 'top_k_mask_logits')
    if options.get('check_nan') and bool(torch.isnan(first).any()):
        raise ValueError('Input contains NaN.')
    _rng(options.get('seed'), options.get('offset'))
    x = _matrix(first, logits=is_logits)
    if name == 'top_k_mask_logits':
        return _top_k(x, options['top_k'], float('-inf')).to(first.dtype)
    if is_logits:
        x = x.softmax(-1)
        if not bool(torch.isfinite(x).all()):
            raise ValueError('logits do not define a finite probability distribution')
    if name == 'top_k_top_p_sampling_from_probs':
        order = options['filter_apply_order']
        if order not in ('top_k_first', 'joint'):
            raise ValueError(f'Invalid filter_apply_order: {order}')
        if order == 'joint':
            x = torch.minimum(_top_k(x, options['top_k']), _top_p(x, options['top_p']))
        else:
            x = _top_p(_normalize(_top_k(x, options['top_k'])), options['top_p'])
    elif 'top_k' in options:
        x = _top_k(x, options['top_k'])
    elif 'top_p' in options:
        x = _top_p(x, options['top_p'])
    elif 'min_p' in options:
        cutoff = _parameter(options['min_p'], x, 'min_p').unsqueeze(1) * x.max(-1, keepdim=True).values
        x = torch.where(x >= cutoff, x, 0)
    if name.endswith('renorm_probs'):
        return _normalize(x).to(first.dtype)
    indices = options.get('indices')
    out_dtype = torch.int32
    if indices is not None:
        if indices.device != first.device or indices.dtype not in (torch.int32, torch.int64) or indices.ndim != 1:
            raise ValueError('indices must be a 1D int32/int64 tensor on the input device')
        if bool(((indices < 0) | (indices >= x.shape[0])).any()):
            raise ValueError('indices out of range')
        x = x.index_select(0, indices.long())
        out_dtype = indices.dtype
    (samples, valid) = _draw(x, options.get('generator'))
    samples = samples.to(out_dtype)
    return (samples, valid) if options.get('return_valid', False) else samples
