"""Paged KV slot mapping and native ND scatter, without scheduler objects."""
from functools import partial

def append_execute(append_key, append_value, batch_indices, positions, paged_kv_cache, kv_indices, kv_indptr, kv_last_page_len, kv_layout='NHD'):
    import torch
    if kv_layout not in ('NHD', 'HND'):
        raise ValueError('kv_layout must be NHD or HND')
    if isinstance(paged_kv_cache, torch.Tensor):
        if paged_kv_cache.ndim != 5 or paged_kv_cache.shape[1] != 2:
            raise ValueError('packed KV must have shape [pages, 2, ...]')
        (keys, values) = (paged_kv_cache[:, 0], paged_kv_cache[:, 1])
    elif isinstance(paged_kv_cache, (tuple, list)) and len(paged_kv_cache) == 2:
        (keys, values) = paged_kv_cache
    else:
        raise TypeError('paged_kv_cache must be a packed tensor or K/V pair')
    if keys.ndim != 4 or values.ndim != 4:
        raise ValueError('K/V caches must be 4D')
    if kv_layout == 'HND':
        (keys, values) = (keys.transpose(1, 2), values.transpose(1, 2))
    if append_key.ndim != 3 or append_value.ndim != 3 or append_key.shape[:2] != append_value.shape[:2]:
        raise ValueError('append K/V must have shape [tokens, heads, dim]')
    if keys.shape[:3] != values.shape[:3] or keys.shape[2:] != append_key.shape[1:] or values.shape[2:] != append_value.shape[1:]:
        raise ValueError('append tensors do not match K/V cache shapes')
    tensors = (append_value, keys, values)
    if any((x.device != append_key.device or x.dtype != append_key.dtype for x in tensors)):
        raise ValueError('append/cache tensors must have matching dtype and device')
    indices = (batch_indices, positions, kv_indices, kv_indptr, kv_last_page_len)
    if any((x.ndim != 1 or x.dtype not in (torch.int32, torch.int64) or x.device != append_key.device for x in indices)):
        raise ValueError('metadata must be 1D int32/int64 tensors on the cache device')
    count = append_key.shape[0]
    if batch_indices.shape != (count,) or positions.shape != (count,):
        raise ValueError('batch_indices and positions must match appended token count')
    if kv_indptr.numel() < 2 or kv_indptr[0].item() != 0 or kv_indptr[-1].item() != kv_indices.numel() or bool((kv_indptr[1:] < kv_indptr[:-1]).any()):
        raise ValueError('invalid kv_indptr')
    batch = kv_indptr.numel() - 1
    if kv_last_page_len.shape != (batch,):
        raise ValueError('kv_last_page_len must have one value per request')
    page_size = keys.shape[1]
    pages = kv_indptr[1:] - kv_indptr[:-1]
    if bool(((kv_last_page_len < 0) | (kv_last_page_len > page_size) | (pages == 0) & (kv_last_page_len != 0) | (pages > 0) & (kv_last_page_len == 0)).any()):
        raise ValueError('invalid last page lengths')
    if bool(((kv_indices < 0) | (kv_indices >= keys.shape[0])).any()):
        raise ValueError('page index exceeds cache capacity')
    if bool(((batch_indices < 0) | (batch_indices >= batch)).any()):
        raise ValueError('batch_indices out of range')
    request = batch_indices.long()
    lengths = torch.where(pages == 0, 0, (pages - 1) * page_size + kv_last_page_len)
    if bool(((positions < 0) | (positions >= lengths[request])).any()):
        raise ValueError('positions exceed declared request lengths')
    page = kv_indices[kv_indptr[request].long() + positions.long() // page_size].long()
    offset = positions.long() % page_size
    slots = page * page_size + offset
    if slots.unique().numel() != count:
        raise ValueError('duplicate destination KV slots are unsupported')
    if not count:
        return None
    keys[page, offset] = append_key
    values[page, offset] = append_value
    return None
