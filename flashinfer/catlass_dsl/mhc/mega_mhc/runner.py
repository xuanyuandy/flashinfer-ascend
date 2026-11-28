"""Compile the fused MHC kernels with CATLASS DSL and launch them on the calling stream."""

from threading import Lock
import weakref

from flashinfer import config
from flashinfer.catlass_dsl._launcher import PreparedLaunch


class CatlassDslMhcRunner:
    def __init__(self, _descriptor):
        self._compiled = {}
        self._lock = Lock()

    def __call__(self, x, residual, previous_mix, post_mix, comb_res_mix, fn,
                mix_scales, mix_bases, rmsnorm_weight, new_residual,
                new_post_mix, new_comb_res_mix, y, new_previous_mix):
        import torch

        if config.jit_disabled():
            raise RuntimeError("CATLASS DSL requires JIT (FLASHINFER_NPU_JIT_DISABLE=1)")
        try:
            from catlass import tla
        except ModuleNotFoundError as error:
            if error.name != "catlass":
                raise
            raise RuntimeError(
                "CATLASS DSL requires ascend-catlass-dsl. Install the 'catlass-dsl' extra."
            ) from error
        from catlass.tla.runtime import from_dlpack

        from . import kernels

        t, h = int(x.shape[0]), int(x.shape[1])
        key = (t, h, x.device.index)
        with torch.npu.device(x.device):
            state = self._compiled.get(key)
            if state is None:
                with self._lock:
                    state = self._compiled.get(key)
                    if state is None:
                        state = _ShapeState(kernels, t, h, x.device)
                        self._compiled[key] = state
            state.launch(
                kernels,
                (x, residual, previous_mix, post_mix, comb_res_mix, fn, mix_scales,
                 mix_bases, rmsnorm_weight, new_residual, new_post_mix,
                 new_comb_res_mix, y, new_previous_mix),
                tla, from_dlpack,
            )


class _ShapeState:
    """Per-shape compiled kernels and GM workspace; one instance per (T, H, device)."""

    def __init__(self, kernels, t, h, device):
        from catlass import tla

        self.plan = kernels._plan(t, h)
        workspace, self.chunk, k0, k1, k2, k3 = kernels._workspace(
            t, h, self.plan, device
        )
        self.workspace = workspace
        self.launches = tuple(
            PreparedLaunch(tla.compile(kernel, *arguments))
            for kernel, arguments in (
                (kernels.cast_kernel, k0), (kernels.mix_kernel, k1),
                (kernels.proj_gemm_kernel, k2), (kernels.post_kernel, k3),
            )
        )
        self.fn_ref = None
        self.fn_ver = None

    def launch(self, kernels, operands, tla, from_dlpack):
        (x, residual, previous_mix, post_mix, comb_res_mix, fn, mix_scales,
         mix_bases, rmsnorm_weight, new_residual, new_post_mix,
         new_comb_res_mix, y, new_previous_mix) = operands
        plan = self.plan
        workspace = self.workspace
        t, h = int(x.shape[0]), int(x.shape[1])
        # K2 reads the RMSNorm input as a `tpad x 4H` column-major matrix and K3 writes
        # the coefficients in groups of VL tokens.  When the shape needs padding the
        # kernels work on the padded workspace and the valid rows are copied out.
        direct = plan["tpad"] == t
        new_res = new_residual.view(t, 4, h) if direct else workspace["new_res"]
        if direct:
            coef_prev = new_previous_mix.view(t, 4)
            coef_post = new_post_mix.view(t, 4)
            coef_comb = new_comb_res_mix.view(t, 16)
        else:
            import torch as _t
            for _k, _s in (("cprev", (plan["tpad"], 4)),
                           ("cpost", (plan["tpad"], 4)),
                           ("ccomb", (plan["tpad"], 16))):
                if _k not in workspace:
                    workspace[_k] = _t.zeros(_s, dtype=_t.float32,
                                             device=new_residual.device)
            coef_prev, coef_post, coef_comb = (
                workspace["cprev"], workspace["cpost"], workspace["ccomb"])
        tensors = dict(
            x=x, res=residual, pm=previous_mix, post=post_mix, comb=comb_res_mix,
            new_res=new_res, pzz=workspace["pzz"], part=workspace["part"],
            puu=workspace["puu"], fn_bf=workspace["fn_bf"],
            w=rmsnorm_weight, scales=mix_scales, bases=mix_bases, y=y,
            new_prev=coef_prev, new_post=coef_post, new_comb=coef_comb,
        )
        k1, k2, k3 = kernels._desc(t, h, plan, tensors)
        cast, mix, proj, post = self.launches

        if self._fn_stale(fn):
            n = kernels._fn_len(plan)
            fn_view = fn.view(1, n)
            fn_bf = workspace["fn_bf"].view(1, n)
            cast((from_dlpack(fn_view, layout_tag=tla.arch.RowMajor),
                  from_dlpack(fn_bf, layout_tag=tla.arch.RowMajor)),
                 (fn_view, fn_bf),
                 block_num=min(kernels.NBLK_VEC, max(1, n // self.chunk)))
            self.fn_ref = weakref.ref(fn)
            self.fn_ver = getattr(fn, "_version", 0)

        mix(k1[:11],
            (x, residual, previous_mix, post_mix, comb_res_mix, new_res,
             rmsnorm_weight, y, workspace["pzz"], workspace["puu"],
             workspace["part"]),
            block_num=kernels.NBLK_VEC)
        proj(k2[:3], (workspace["fn_bf"], new_res, workspace["part"]),
             block_num=kernels.SPLIT_MAX)
        # K3 launches NCOEF coefficient blocks plus one block per cold-token group;
        # with no cold token only the coefficient chain runs.
        post(k3[:10],
             (workspace["pzz"], workspace["part"], mix_scales, mix_bases,
              coef_prev, coef_post, coef_comb, y, rmsnorm_weight,
              workspace["puu"]),
             block_num=plan["k3blk"])

        if not direct:
            new_residual.copy_(new_res[:t])
        if not direct:
            new_previous_mix.copy_(coef_prev[:t])
            new_post_mix.copy_(coef_post[:t].unsqueeze(-1))
            new_comb_res_mix.copy_(coef_comb[:t].reshape(t, 4, 4))

    def _fn_stale(self, fn):
        reference = self.fn_ref
        if reference is None:
            return True
        return (reference() is not fn
                or self.fn_ver != getattr(fn, "_version", 0))
