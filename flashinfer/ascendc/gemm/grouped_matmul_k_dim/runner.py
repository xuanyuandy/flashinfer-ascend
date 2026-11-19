"""Adapt the shared logical call only after this Ascend C kernel is selected."""

from dataclasses import dataclass

from flashinfer.jit.tensor_adapter import package_tensor_runner


@dataclass(frozen=True)
class GroupedMatmulKDimRunner:
    native: object

    def __call__(self, x, weight, group_list, y, *, trans_a=False, fp32_output=False):
        # The kernel only writes the groups it actually visits, so a zero-length
        # group would otherwise expose whatever the allocator handed out. The
        # reference operator clears its output for the same reason.
        y.zero_()
        # fp32_output only selects the output dtype, which already picked the
        # tiling key; the native ABI carries trans_a alone.
        return self.native(x, weight, group_list, y, trans_a=bool(trans_a))


def make_runner(module, spec, recipe):
    return GroupedMatmulKDimRunner(package_tensor_runner(module, recipe, spec=spec))
