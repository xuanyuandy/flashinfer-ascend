import unittest

from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.runtime import SchemaError
from tests.support.attention_provider import ProviderRuntimeTestCase


class FakeNpuWorkspace:
    def __init__(self, size=1024, *, dtype="uint8", device="npu:0"):
        self.shape = (size,)
        self.dtype = dtype
        self.device = device


def plan_wrapper(wrapper):
    wrapper.plan(
        [0, 1],
        [7],
        [64],
        8,
        2,
        128,
        128,
        q_data_type="bfloat16",
        kv_data_type="bfloat16",
        o_data_type="bfloat16",
    )


class ProviderWorkspaceResetTests(ProviderRuntimeTestCase):
    """Package-managed providers retain plans across workspace replacement."""

    def wrapper(self):
        return BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout="HND", backend="auto"
        )

    def test_reset_after_plan_updates_binding_and_preserves_active_plan(self):
        wrapper = self.wrapper()
        plan_wrapper(wrapper)
        plan = wrapper.plan_state
        initial = wrapper.workspace_contract

        wrapper.reset_workspace_buffer(
            FakeNpuWorkspace(2048), FakeNpuWorkspace(512)
        )

        rebound = wrapper.workspace_contract
        self.assertIs(wrapper.plan_state, plan)
        self.assertEqual(rebound.binding_generation, initial.binding_generation + 1)
        self.assertEqual(rebound.plan_generation, plan.generation)
        self.assertEqual(rebound.required_sizes, (0, 0))
        self.assertEqual(rebound.float_capacity_bytes, 2048)
        self.assertEqual(rebound.int_capacity_bytes, 512)
        self.assertIs(
            rebound,
            wrapper._operator_runtime.workspace_contract,
        )
        self.assertEqual(
            wrapper.run("q", ("k", "v")),
            "package-output:q",
        )

    def test_reset_before_plan_is_bound_by_the_selected_operation(self):
        wrapper = self.wrapper()
        wrapper.reset_workspace_buffer(
            FakeNpuWorkspace(256), FakeNpuWorkspace(128)
        )
        preplan = wrapper.workspace_contract
        self.assertFalse(preplan.requirements_known)

        plan_wrapper(wrapper)

        planned = wrapper.workspace_contract
        self.assertEqual(planned.binding_generation, preplan.binding_generation)
        self.assertEqual(planned.required_sizes, (0, 0))
        self.assertEqual(planned.float_capacity_bytes, 256)
        self.assertEqual(planned.int_capacity_bytes, 128)

    def test_reset_rejects_alias_dtype_and_device_drift(self):
        wrapper = self.wrapper()
        shared = FakeNpuWorkspace(64)
        with self.assertRaisesRegex(SchemaError, "cannot alias"):
            wrapper.reset_workspace_buffer(shared, shared)
        with self.assertRaisesRegex(SchemaError, "dtype must be uint8"):
            wrapper.reset_workspace_buffer(
                FakeNpuWorkspace(dtype="float16"), FakeNpuWorkspace(32)
            )
        with self.assertRaisesRegex(SchemaError, "same device"):
            wrapper.reset_workspace_buffer(
                FakeNpuWorkspace(device="npu:0"),
                FakeNpuWorkspace(32, device="npu:1"),
            )
        with self.assertRaisesRegex(SchemaError, "device cannot change"):
            wrapper.reset_workspace_buffer(
                FakeNpuWorkspace(device="npu:1"),
                FakeNpuWorkspace(32, device="npu:1"),
            )


if __name__ == "__main__":
    unittest.main()
