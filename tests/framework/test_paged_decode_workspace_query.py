import unittest

from flashinfer.attention import (
    AttentionStateError,
)
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from tests.support.attention_provider import ProviderRuntimeTestCase, package_attention
from tests.framework.test_attention_batch_providers import FakeNpuWorkspace, plan_paged_decode as plan_public_wrapper


def query_workspace(value, **kwargs):
    return value.workspace_size(
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
        **kwargs,
    )


class PagedDecodeWorkspaceQueryTests(ProviderRuntimeTestCase):
    """Paged decode shares the provider query fork without changing plan state."""

    def wrapper(self):
        return BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(), kv_layout="HND", backend="auto"
        )

    def test_query_before_plan_does_not_publish_runtime_state(self):
        value = self.wrapper()
        initial_workspace = value.workspace_contract

        self.assertEqual(query_workspace(value), (0, 0))

        self.assertFalse(value._operator_runtime.is_planned)
        self.assertIs(value.workspace_contract, initial_workspace)
        self.assertFalse(value.workspace_contract.requirements_known)
        with self.assertRaises(AttentionStateError):
            _ = value.plan_selection
        self.assertEqual(package_attention.calls, [])

    def test_query_after_plan_preserves_selection_workspace_and_executor(self):
        value = self.wrapper()
        plan_public_wrapper(value)
        original_plan = value.plan_state
        original_selection = value.plan_selection
        original_workspace = value.workspace_contract
        original_executor = value._operator_runtime._executor

        self.assertEqual(query_workspace(value, window_left=8), (0, 0))

        self.assertIs(value.plan_state, original_plan)
        self.assertEqual(value.plan_selection, original_selection)
        self.assertIs(value.workspace_contract, original_workspace)
        self.assertIs(value._operator_runtime._executor, original_executor)
        self.assertEqual(
            value.run("q", ("k", "v")),
            "package-output:q",
        )

    def test_query_rejects_unbound_controls_before_provider_resolution(self):
        value = self.wrapper()

        with self.assertRaisesRegex(NotImplementedError, "split-K"):
            query_workspace(value, fixed_split_size=1)

        self.assertEqual(self.components["loader"].resolve_calls, 0)
        self.assertEqual(self.components["authority"].calls, 0)

    def test_failed_query_preserves_existing_plan(self):
        value = self.wrapper()
        plan_public_wrapper(value)
        original_plan = value.plan_state
        original_selection = value.plan_selection
        self.components["authority"].fail = True

        with self.assertRaisesRegex(RuntimeError, "authority failure"):
            query_workspace(value, window_left=8)

        self.assertIs(value.plan_state, original_plan)
        self.assertEqual(value.plan_selection, original_selection)


if __name__ == "__main__":
    unittest.main()
