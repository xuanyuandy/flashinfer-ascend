"""attention batch providers regression coverage."""


import inspect

from flashinfer.attention import (
    AttentionMode,
    AttentionOperatorRuntimeResolverRegistry,
    install_attention_operator_runtime_resolvers,
)
from flashinfer.prefill import BatchPrefillWithPagedKVCacheWrapper
from tests.support.attention_provider import ProviderRuntimeTestCase, package_attention


class FakeNpuWorkspace:
    shape = (1024,)
    dtype = "uint8"
    device = "npu:0"


def plan_paged_prefill(wrapper):
    return wrapper.plan(
        [0, 2],
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


class PublicPagedPrefillProviderRuntimeTests(ProviderRuntimeTestCase):
    """The FlashInfer facade owns provider selection and plan/run state."""

    def wrapper(self):
        return BatchPrefillWithPagedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout="HND",
            backend="auto",
        )

    def test_public_plan_selects_provider_and_run_hides_lse_by_default(self):
        wrapper = self.wrapper()

        self.assertIsNone(plan_paged_prefill(wrapper))
        output = wrapper.run("q", ("k-cache", "v-cache"))
        output_with_lse = wrapper.run(
            "q-lse", ("k-cache", "v-cache"), return_lse=True
        )

        self.assertEqual(output, "package-output:q")
        self.assertEqual(
            output_with_lse,
            ("package-output:q-lse", "package-lse:0.25"),
        )
        self.assertEqual(wrapper.plan_state.spec.mode, AttentionMode.BATCH_PREFILL_PAGED)
        self.assertEqual(self.components["loader"].resolve_calls, 1)
        self.assertEqual(self.components["authority"].calls, 1)
        self.assertEqual(len(package_attention.calls), 2)
        self.assertFalse(package_attention.calls[0][5])
        self.assertTrue(package_attention.calls[1][5])

    def test_wrapper_freezes_the_registry_snapshot_at_construction(self):
        wrapper = self.wrapper()
        install_attention_operator_runtime_resolvers(
            AttentionOperatorRuntimeResolverRegistry()
        )

        plan_paged_prefill(wrapper)
        self.assertEqual(
            wrapper.run("q", ("k", "v")),
            "package-output:q",
        )

    def test_unbound_public_options_fail_before_provider_execution(self):
        wrapper = self.wrapper()
        plan_paged_prefill(wrapper)

        with self.assertRaisesRegex(NotImplementedError, "query-scale"):
            wrapper.run("q", ("k", "v"), q_scale="scale")
        with self.assertRaisesRegex(NotImplementedError, "attention-sink"):
            wrapper.run("q", ("k", "v"), sinks="sink")
        with self.assertRaisesRegex(NotImplementedError, "SP-compression"):
            wrapper.run("q", ("k", "v"), uses_spcompress=True)
        with self.assertRaisesRegex(NotImplementedError, "LSE-buffer"):
            wrapper.run("q", ("k", "v"), lse="lse-buffer")
        with self.assertRaisesRegex(NotImplementedError, "split-K"):
            wrapper.plan(
                [0, 2], [0, 1], [7], [64], 8, 2, 128, 128,
                fixed_split_size=1,
            )
        self.assertEqual(package_attention.calls, [])

    def test_provider_graph_fails_and_workspace_query_remains_non_executing(self):
        with self.assertRaisesRegex(NotImplementedError, "graph resources"):
            BatchPrefillWithPagedKVCacheWrapper(
                FakeNpuWorkspace(),
                kv_layout="HND",
                use_cuda_graph=True,
                backend="auto",
            )
        wrapper = self.wrapper()
        self.assertEqual(
            wrapper.workspace_size(
                [0, 2], [0, 1], [7], [64], 8, 2, 128, 128
            ),
            (0, 0),
        )
        self.assertEqual(package_attention.calls, [])

    def test_public_signature_exposes_no_provider_runtime_handle(self):
        for callable_value in (
            BatchPrefillWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper.plan,
            BatchPrefillWithPagedKVCacheWrapper.run,
        ):
            parameters = inspect.signature(callable_value).parameters
            self.assertNotIn("provider", parameters)
            self.assertNotIn("runtime", parameters)
            self.assertNotIn("plan_handle", parameters)


from flashinfer.attention.batch import HostBatchReferenceWrapper
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper




class UnconnectedBatchWrapper(HostBatchReferenceWrapper):
    def __init__(self):
        self._init_host_wrapper(
            mode=AttentionMode.BATCH_PREFILL_RAGGED,
            float_workspace_buffer=FakeNpuWorkspace(),
            backend="auto",
            graph_enabled=False,
            fixed_batch_size=None,
        )




def plan_paged_decode(wrapper, **kwargs):
    return wrapper.plan(
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


class PublicPagedDecodeProviderRuntimeTests(ProviderRuntimeTestCase):
    """The paged-decode facade owns provider selection and plan/run state."""

    def wrapper(self):
        return BatchDecodeWithPagedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout="HND",
            backend="auto",
        )

    def test_public_plan_selects_provider_and_run_preserves_return_shape(self):
        wrapper = self.wrapper()

        self.assertIsNone(plan_paged_decode(wrapper))
        output = wrapper.run("q", ("k-cache", "v-cache"))
        output_with_lse = wrapper.run(
            "q-lse", ("k-cache", "v-cache"), return_lse=True
        )

        self.assertEqual(output, "package-output:q")
        self.assertEqual(
            output_with_lse,
            ("package-output:q-lse", "package-lse:0.25"),
        )
        self.assertEqual(wrapper.plan_state.spec.mode, AttentionMode.BATCH_DECODE_PAGED)
        self.assertEqual(wrapper.workspace_contract.required_sizes, (0, 0))
        self.assertEqual(
            wrapper.workspace_contract.plan_generation,
            wrapper.plan_state.generation,
        )
        self.assertEqual(self.components["loader"].resolve_calls, 1)
        self.assertEqual(self.components["authority"].calls, 1)
        self.assertEqual(len(package_attention.calls), 2)

    def test_wrapper_freezes_registry_and_catalog_at_construction(self):
        wrapper = self.wrapper()
        install_attention_operator_runtime_resolvers(
            AttentionOperatorRuntimeResolverRegistry()
        )

        plan_paged_decode(wrapper)
        self.assertEqual(
            wrapper.run("q", ("k", "v")),
            "package-output:q",
        )

    def test_unbound_options_fail_before_provider_execution(self):
        wrapper = self.wrapper()
        plan_paged_decode(wrapper)

        with self.assertRaisesRegex(NotImplementedError, "query-scale"):
            wrapper.run("q", ("k", "v"), q_scale="scale")
        with self.assertRaisesRegex(NotImplementedError, "attention-sink"):
            wrapper.run("q", ("k", "v"), sinks="sink")
        with self.assertRaisesRegex(NotImplementedError, "skip-softmax"):
            wrapper.run(
                "q",
                ("k", "v"),
                skip_softmax_threshold_scale_factor=1.0,
            )
        with self.assertRaisesRegex(NotImplementedError, "split-K"):
            plan_paged_decode(wrapper, fixed_split_size=1)
        self.assertEqual(package_attention.calls, [])

    def test_provider_resource_controls_and_non_executing_workspace_query(self):
        with self.assertRaisesRegex(NotImplementedError, "graph resources"):
            BatchDecodeWithPagedKVCacheWrapper(
                FakeNpuWorkspace(),
                use_cuda_graph=True,
                backend="auto",
            )
        with self.assertRaisesRegex(NotImplementedError, "matrix-core"):
            BatchDecodeWithPagedKVCacheWrapper(
                FakeNpuWorkspace(),
                use_tensor_cores=True,
                backend="auto",
            )
        wrapper = self.wrapper()
        self.assertEqual(
            wrapper.workspace_size([0, 1], [7], [64], 8, 2, 128, 128),
            (0, 0),
        )
        self.assertEqual(package_attention.calls, [])

    def test_shared_base_never_enables_provider_resolution_implicitly(self):
        with self.assertRaisesRegex(
            NotImplementedError, "batch_prefill_ragged public wrapper"
        ):
            UnconnectedBatchWrapper()
        self.assertEqual(self.components["loader"].resolve_calls, 0)
        self.assertEqual(self.components["authority"].calls, 0)

    def test_public_signature_exposes_no_provider_runtime_handle(self):
        for callable_value in (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchDecodeWithPagedKVCacheWrapper.plan,
            BatchDecodeWithPagedKVCacheWrapper.run,
        ):
            parameters = inspect.signature(callable_value).parameters
            self.assertNotIn("provider", parameters)
            self.assertNotIn("runtime", parameters)
            self.assertNotIn("plan_handle", parameters)


from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper






def plan_ragged_prefill(wrapper, **kwargs):
    return wrapper.plan(
        [0, 2],
        [0, 2],
        8,
        2,
        128,
        q_data_type="bfloat16",
        kv_data_type="bfloat16",
        o_data_type="bfloat16",
        **kwargs,
    )


class PublicRaggedPrefillProviderRuntimeTests(ProviderRuntimeTestCase):
    """The ragged-prefill facade owns provider selection and plan/run state."""

    def wrapper(self):
        return BatchPrefillWithRaggedKVCacheWrapper(
            FakeNpuWorkspace(),
            kv_layout="HND",
            backend="auto",
        )

    def test_public_plan_selects_provider_and_run_preserves_return_shape(self):
        wrapper = self.wrapper()

        self.assertIsNone(plan_ragged_prefill(wrapper))
        output = wrapper.run("q", "k", "v")
        output_with_lse = wrapper.run("q-lse", "k", "v", return_lse=True)

        self.assertEqual(output, "package-output:q")
        self.assertEqual(
            output_with_lse,
            ("package-output:q-lse", "package-lse:0.25"),
        )
        self.assertEqual(
            wrapper.plan_state.spec.mode,
            AttentionMode.BATCH_PREFILL_RAGGED,
        )
        self.assertEqual(wrapper.workspace_contract.required_sizes, (0, 0))
        self.assertEqual(
            wrapper.workspace_contract.plan_generation,
            wrapper.plan_state.generation,
        )
        self.assertEqual(self.components["loader"].resolve_calls, 1)
        self.assertEqual(self.components["authority"].calls, 1)
        self.assertEqual(len(package_attention.calls), 2)

    def test_wrapper_freezes_registry_and_catalog_at_construction(self):
        wrapper = self.wrapper()
        install_attention_operator_runtime_resolvers(
            AttentionOperatorRuntimeResolverRegistry()
        )

        plan_ragged_prefill(wrapper)
        self.assertEqual(wrapper.run("q", "k", "v"), "package-output:q")

    def test_unbound_options_fail_before_provider_execution(self):
        wrapper = self.wrapper()
        plan_ragged_prefill(wrapper)

        with self.assertRaisesRegex(NotImplementedError, "query-scale"):
            wrapper.run("q", "k", "v", q_scale="scale")
        with self.assertRaisesRegex(NotImplementedError, "o_scale"):
            wrapper.run("q", "k", "v", o_scale="scale")
        with self.assertRaisesRegex(NotImplementedError, "split-K"):
            plan_ragged_prefill(wrapper, fixed_split_size=1)
        with self.assertRaisesRegex(NotImplementedError, "custom-mask"):
            plan_ragged_prefill(wrapper, custom_mask=[True, True, True, True])
        self.assertEqual(package_attention.calls, [])

    def test_provider_graph_resources_fail_explicitly(self):
        with self.assertRaisesRegex(NotImplementedError, "graph resources"):
            BatchPrefillWithRaggedKVCacheWrapper(
                FakeNpuWorkspace(),
                use_cuda_graph=True,
                backend="auto",
            )

    def test_public_signature_exposes_no_provider_runtime_handle(self):
        for callable_value in (
            BatchPrefillWithRaggedKVCacheWrapper,
            BatchPrefillWithRaggedKVCacheWrapper.plan,
            BatchPrefillWithRaggedKVCacheWrapper.run,
        ):
            parameters = inspect.signature(callable_value).parameters
            self.assertNotIn("provider", parameters)
            self.assertNotIn("runtime", parameters)
            self.assertNotIn("plan_handle", parameters)
