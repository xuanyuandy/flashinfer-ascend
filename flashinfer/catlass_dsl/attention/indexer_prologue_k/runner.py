"""用 CATLASS DSL 编译 indexer K 路径 prologue，并发射两个 kernel。"""

from flashinfer import config


class CatlassDslIndexerPrologueRunner:
    """持有某一 ``(H, D, Dr)`` / 存储配置对应的发射计划。"""

    def __init__(self, _descriptor):
        self._views = {}
        self._wraps = {}
        self._artifacts = {}
        self._buffers = {}
        self._cores = {}
        self._plans = {}

    def __call__(self, latent, wk, norm_weight, rope_sin, rope_cos, k_cache,
                 k_scale_cache, cache_index, *, storage_mode, norm_eps,
                 combined_block_size, write_scale=True):
        """把 ``latent[T,H]`` 投影后把 MXFP4 行散射进 paged cache。

        ``T`` 和 cache 几何都是动态的：同一模型形状和存储配置下，一份编译产物
        服务所有 token 数。发射计划（张量绑定、artifact、grid）按输入身份缓存，
        稳态调用就是两次 artifact 发射。
        """
        if int(latent.shape[0]) == 0:
            return
        if config.jit_disabled():
            raise RuntimeError("CATLASS DSL requires JIT (FLASHINFER_NPU_JIT_DISABLE=1)")
        try:
            import catlass.tla
        except ModuleNotFoundError as error:
            if error.name != "catlass":
                raise
            raise RuntimeError(
                "CATLASS DSL requires ascend-catlass-dsl. Install the 'catlass-dsl' extra."
            ) from error

        key = (
            latent.data_ptr(),
            wk.data_ptr(),
            norm_weight.data_ptr(),
            rope_sin.data_ptr(),
            rope_cos.data_ptr(),
            k_cache.data_ptr(),
            None if k_scale_cache is None else k_scale_cache.data_ptr(),
            cache_index.data_ptr(),
            int(latent.shape[0]),
            int(latent.shape[1]),
            int(wk.shape[0]),
            int(rope_sin.shape[1]),
            tuple(k_cache.shape),
            None if k_scale_cache is None else tuple(k_scale_cache.shape),
            int(storage_mode),
            int(combined_block_size),
            float(norm_eps),
            bool(write_scale),
        )
        plan = self._plans.get(key)
        if plan is None:
            plan = self._prepare(
                latent, wk, norm_weight, rope_sin, rope_cos, k_cache,
                k_scale_cache, cache_index, int(storage_mode), float(norm_eps),
                int(combined_block_size), bool(write_scale),
            )
            if len(self._plans) > 8:
                self._plans.clear()
            self._plans[key] = plan
        proj, proj_args, proj_blocks, post, post_args, post_blocks = plan
        proj(*proj_args, block_num=proj_blocks)
        post(*post_args, block_num=post_blocks)

    def _view2d(self, tensor, cols):
        key = (tensor.data_ptr(), tuple(tensor.shape), cols)
        hit = self._views.get(key)
        if hit is not None:
            return hit
        view = tensor.view(-1, cols)
        self._views[key] = view
        if len(self._views) > 16:
            self._views.clear()
            self._views[key] = view
        return view

    def _wrap(self, buf, layout_tag, origin_shape, dynamic=True):
        """缓存 Torch buffer 的 DSL 视图；``origin_shape`` 固定 stride。"""
        from catlass.tla.runtime import from_dlpack

        key = (
            id(buf),
            buf.data_ptr(),
            tuple(buf.shape),
            str(buf.dtype),
            id(layout_tag),
            bool(dynamic),
        )
        hit = self._wraps.get(key)
        if hit is not None:
            return hit
        tensor = from_dlpack(buf, layout_tag=layout_tag, origin_shape=origin_shape)
        if dynamic:
            tensor = tensor.mark_layout_dynamic()
        self._wraps[key] = tensor
        if len(self._wraps) > 32:
            self._wraps.clear()
            self._wraps[key] = tensor
        return tensor

    def _buf2d(self, name, rows, cols, dtype, device):
        import torch

        hit = self._buffers.get(name)
        if (
            hit is not None
            and hit[0] == (rows, cols)
            and hit[1].device == device
            and hit[1].dtype == dtype
        ):
            return hit[1]
        buf = torch.empty(rows * cols, dtype=dtype, device=device).view(rows, cols)
        self._buffers[name] = ((rows, cols), buf)
        return buf

    def _compile(self, kernel, args, key):
        from catlass import tla

        art = self._artifacts.get(key)
        if art is None:
            art = tla.compile(kernel, *args, options="--npu-arch 3510")
            self._artifacts[key] = art
        return art

    def _core_counts(self, device):
        import torch

        hit = self._cores.get(device)
        if hit is not None:
            return hit
        props = torch.npu.get_device_properties(device)
        counts = (int(props.cube_core_num), int(props.vector_core_num))
        self._cores[device] = counts
        return counts

    def _prepare(self, latent, wk, norm_weight, rope_sin, rope_cos, k_cache,
                 k_scale_cache, cache_index, storage_mode, norm_eps,
                 combined_block_size, write_scale):
        """为一份输入配置构建发射计划。"""
        import torch
        from catlass import tla

        from .indexer_prologue_k import ipk_post, ipk_projection, pick_tiling

        mode = storage_mode
        group = combined_block_size
        eps = norm_eps
        total_rows, head_dim = int(latent.shape[0]), int(latent.shape[1])
        index_head_dim = int(wk.shape[0])
        rope_dim = int(rope_sin.shape[1])
        device = latent.device
        if index_head_dim % 64 != 0:
            raise ValueError("this implementation requires D % 64 == 0")
        if (index_head_dim - rope_dim) % 64 != 0:
            raise ValueError(
                "this implementation requires (D - Dr) % 64 == 0 so the RoPE tail "
                "starts on an aligned 64-element boundary"
            )

        align = 128
        head_pad = (head_dim + align - 1) // align * align

        tiling = pick_tiling(total_rows, head_pad, index_head_dim)
        split = int(tiling.split)
        m_pad = (total_rows + tiling.bm - 1) // tiling.bm * tiling.bm

        latent_in = latent
        if head_pad != head_dim or m_pad != total_rows:
            latent_in = self._buf2d("latent_pad", m_pad, head_pad, latent.dtype, device)
            latent_in.zero_()
            latent_in[:total_rows, :head_dim] = latent
        wk_in = wk
        if head_pad != head_dim:
            wk_in = self._buf2d("wk_pad", index_head_dim, head_pad, wk.dtype, device)
            wk_in.zero_()
            wk_in[:, :head_dim] = wk

        projected = self._buf2d(
            "projected", m_pad, index_head_dim * split, torch.float32, device
        )

        write_scale = bool(write_scale) and mode == 0
        scale_arg = k_scale_cache
        if scale_arg is None:
            raise ValueError("k_scale_cache must be provided for this backend")

        t_latent = self._wrap(latent_in, tla.arch.RowMajor, (m_pad, head_pad), dynamic=False)
        t_wk = self._wrap(wk_in, tla.arch.ColumnMajor, (head_pad, index_head_dim), dynamic=False)
        t_proj = self._wrap(
            projected,
            tla.arch.RowMajor,
            (m_pad, index_head_dim * split),
            dynamic=False,
        )
        t_proj_dyn = self._wrap(
            projected, tla.arch.RowMajor, (m_pad, index_head_dim * split)
        )
        t_sin = self._wrap(rope_sin, tla.arch.RowMajor, (total_rows, rope_dim))
        t_cos = self._wrap(rope_cos, tla.arch.RowMajor, (total_rows, rope_dim))
        t_gamma = self._wrap(
            self._view2d(norm_weight, index_head_dim),
            tla.arch.RowMajor,
            (1, index_head_dim),
        )
        t_idx = self._wrap(cache_index, tla.arch.RowMajor, (total_rows,))
        cache_rows = int(k_cache.shape[0]) * int(k_cache.shape[1])
        cache_cols = int(k_cache.shape[3])
        t_cache = self._wrap(
            self._view2d(k_cache, cache_cols), tla.arch.RowMajor, (cache_rows, cache_cols)
        )
        if write_scale:
            scale_cols = int(scale_arg.shape[3])
            t_scale = self._wrap(
                self._view2d(scale_arg, scale_cols),
                tla.arch.RowMajor,
                (int(scale_arg.shape[0]) * int(scale_arg.shape[1]), scale_cols),
            )
        else:
            t_scale = self._wrap(scale_arg.view(-1), tla.arch.RowMajor, (scale_arg.numel(),))

        cube_cores, _vector_cores = self._core_counts(device.index or 0)

        nm = m_pad // tiling.bm
        nn = (index_head_dim + tiling.bn - 1) // tiling.bn
        proj_blocks = min(cube_cores, nm * nn * split)
        proj_key = (
            "proj",
            head_pad,
            index_head_dim,
            tiling.bm,
            tiling.bn,
            tiling.bk,
            tiling.sk,
            split,
            m_pad,
        )
        proj = self._compile(
            ipk_projection, (t_latent, t_wk, t_proj, m_pad, tiling), proj_key
        )

        shift = 0
        post_key = (
            "post",
            index_head_dim,
            rope_dim,
            mode,
            group,
            shift,
            write_scale,
            split,
        )
        post = self._compile(
            ipk_post,
            (
                t_proj_dyn,
                t_sin,
                t_cos,
                t_gamma,
                t_idx,
                t_cache,
                t_scale,
                tla.Float32(eps),
                int(k_cache.shape[1]),
                total_rows,
                mode,
                group,
                index_head_dim,
                rope_dim,
                shift,
                write_scale,
                split,
            ),
            post_key,
        )
        post_args = (
            t_proj_dyn,
            t_sin,
            t_cos,
            t_gamma,
            t_idx,
            t_cache,
            t_scale,
            eps,
            int(k_cache.shape[1]),
            total_rows,
        )
        return (
            proj,
            (t_latent, t_wk, t_proj, m_pad),
            proj_blocks,
            post,
            post_args,
            cube_cores,
        )
