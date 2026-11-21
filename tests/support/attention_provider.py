"""Test provider, isolated registry and synthetic package fixtures for attention."""

import unittest
from dataclasses import dataclass

from flashinfer.attention import (
    AttentionDispatchReceipt,
    AttentionFrameworkSession,
    AttentionLoweredOperatorCall,
    AttentionMode,
    AttentionObservedCallableSignature,
    AttentionObservedOperatorCallable,
    AttentionOperatorBatchRuntime,
    AttentionOperatorOperationCatalog,
    AttentionOperatorOperationSpec,
    AttentionOperatorPackageCompatibility,
    AttentionOperatorPackageResolver,
    AttentionOperatorPackageRuntimeImplementation,
    AttentionOperatorProviderProbe,
    AttentionOperatorProviderSelection,
    AttentionOperatorRuntimeAuthority,
    AttentionOperatorRuntimeImplementationRegistry,
    AttentionOperatorRuntimeResolverRegistry,
    AttentionOperatorTensorPlan,
    AttentionPlanSpec,
    AttentionPreparedOperatorPlan,
    KVLayout,
    MixedPagedKVMetadata,
    attention_operator_runtime_registry_snapshot,
    bind_attention_operator_callable,
    install_attention_operator_runtime_resolvers,
    load_packaged_attention_operator_catalog,
)
from flashinfer.runtime import Backend, SchemaError
from tests.support import hash_value


def package_attention(
    query,
    key,
    value,
    *,
    table=None,
    scale=1.0,
    return_softmax_lse=False,
    key_scale=None,
    value_scale=None,
    runtime_key_scale=None,
    runtime_value_scale=None,
):
    package_attention.calls.append(
        (
            query,
            key,
            value,
            table,
            scale,
            return_softmax_lse,
            key_scale,
            value_scale,
            runtime_key_scale,
            runtime_value_scale,
        )
    )
    output = "package-output:%s" % query
    if return_softmax_lse:
        return (output, "package-lse:%s" % scale)
    return output


package_attention.calls = []


def fake_operation():
    return AttentionOperatorOperationSpec(
        operation_id="cann.checkpoint_019_attention@v1",
        provider_id="cann",
        package_name="checkpoint-019-package",
        callable_path="checkpoint_019_package.attention",
        api_version="v1",
        candidate_modes=(
            AttentionMode.SINGLE_PREFILL,
            AttentionMode.SINGLE_DECODE,
            AttentionMode.BATCH_MIXED_PAGED,
            AttentionMode.BATCH_PREFILL_PAGED,
            AttentionMode.BATCH_PREFILL_RAGGED,
            AttentionMode.BATCH_DECODE_PAGED,
        ),
        positional_arguments=("query", "key", "value"),
        keyword_arguments=(
            "table",
            "scale",
            "return_softmax_lse",
            "key_scale",
            "value_scale",
            "runtime_key_scale",
            "runtime_value_scale",
        ),
        return_names=("output", "softmax_lse"),
        paged_table_argument="table",
        lse_control_argument="return_softmax_lse",
        quant_arguments=(
            "key_scale",
            "value_scale",
            "runtime_key_scale",
            "runtime_value_scale",
        ),
        source_url="https://example.com/checkpoint-019-attention-v1",
    )


class FakePackageLoader:
    loader_id = "checkpoint-019-loader-v1"

    def __init__(self, events, version="1.0.0"):
        self.events = events
        self.version = version
        self.version_calls = 0
        self.resolve_calls = 0

    def package_version(self, package_name):
        self.version_calls += 1
        self.events.append("package_version")
        return self.version

    def resolve_callable(self, callable_path):
        self.resolve_calls += 1
        self.events.append("resolve_callable")
        return package_attention


class FakePlanGate:
    provider_id = "cann"
    operation_id = fake_operation().operation_id

    def __init__(self, events, reasons=()):
        self.events = events
        self.reasons = tuple(reasons)
        self.calls = 0

    def rejection_reasons(self, plan, device):
        self.calls += 1
        self.events.append("plan_gate")
        return self.reasons


def receipt_for(plan):
    return AttentionDispatchReceipt(
        mode=plan.spec.mode,
        plan_fingerprint=plan.fingerprint,
        admission_fingerprint=plan.admission_fingerprint,
        workload_fingerprint=plan.workload.fingerprint,
        numerics_policy_fingerprint=hash_value("1"),
        profile_id="checkpoint.019.profile.v1",
        profile_fingerprint=hash_value("2"),
        rule_id="checkpoint_019_rule_v1",
        environment_fingerprint=hash_value("3"),
        evidence_id="checkpoint-019-evidence",
        evidence_result_digest=hash_value("4"),
        kernel_id="checkpoint-019-kernel",
        kernel_fingerprint=hash_value("5"),
        artifact_fingerprint=hash_value("6"),
        launch_abi_fingerprint=hash_value("7"),
        binary_abi_fingerprint=hash_value("8"),
        backend=Backend.ACLNN,
        float_workspace_bytes=0,
        int_workspace_bytes=0,
        float_workspace_alignment=1,
        int_workspace_alignment=1,
        selection_source="priority",
        requested_backend="auto",
    )


class FakeAuthorityResolver:
    provider_id = "cann"
    operation_id = fake_operation().operation_id

    def __init__(self, events):
        self.events = events
        self.calls = 0
        self.fail = False
        self.drift = False

    def authorize(self, plan, device, operation, provider_probe):
        self.calls += 1
        self.events.append("authorize")
        if self.fail:
            raise RuntimeError("checkpoint-019 authority failure")
        receipt = receipt_for(plan)
        selection = AttentionOperatorProviderSelection(
            provider_id="cann",
            provider_probe_fingerprint=provider_probe.fingerprint,
            provider_record_fingerprint=hash_value("9"),
            dispatch_receipt_fingerprint=receipt.fingerprint,
            profile_id=receipt.profile_id,
            profile_fingerprint=receipt.profile_fingerprint,
            backend=receipt.backend,
        )
        return AttentionOperatorRuntimeAuthority(
            framework_plan_fingerprint=plan.fingerprint,
            device=device,
            provider_probe_fingerprint=provider_probe.fingerprint,
            operation_id=operation.operation_id,
            operation_fingerprint=(
                hash_value("a") if self.drift else operation.fingerprint
            ),
            receipt=receipt,
            selection=selection,
        )


@dataclass(frozen=True)
class FakeLogicalState:
    table: AttentionOperatorTensorPlan


class FakeLogicalFactory:
    provider_id = "cann"
    operation_id = fake_operation().operation_id

    def __init__(self, events):
        self.events = events
        self.calls = 0

    def prepare(self, plan, receipt, selection):
        self.calls += 1
        self.events.append("prepare")
        state = FakeLogicalState(
            table=AttentionOperatorTensorPlan(
                role="table",
                shape=(2,),
                dtype="int32",
                materialization="explicit_int32",
                values=(7, 9),
            )
        )
        return AttentionPreparedOperatorPlan(
            provider_id=self.provider_id,
            provider_selection_fingerprint=selection.fingerprint,
            framework_plan_fingerprint=plan.fingerprint,
            framework_plan_generation=plan.generation,
            implementation_id=self.operation_id,
            opaque_plan_token="checkpoint-019-logical-%d" % plan.generation,
            opaque_state=state,
        )


class FakeLogicalRunAdapter:
    provider_id = "cann"

    def lower(self, active_plan, request):
        state = active_plan.prepared_plan.opaque_state
        if not isinstance(state, FakeLogicalState):
            raise SchemaError("fake logical state is missing")
        key, value = request.kv_cache
        return AttentionLoweredOperatorCall(
            provider_id=self.provider_id,
            operation_id=fake_operation().operation_id,
            active_plan_fingerprint=active_plan.fingerprint,
            positional_arguments=(
                ("query", request.query),
                ("key", key),
                ("value", value),
            ),
            keyword_arguments=(
                ("table", state.table),
                ("scale", 0.25),
                ("return_softmax_lse", request.return_lse),
            ),
            return_names=(
                ("output", "softmax_lse")
                if request.return_lse
                else ("output",)
            ),
            consumed_request_fields=request.consumed_fields,
        )


class FakeTensorMaterializer:
    provider_id = "cann"
    materializer_id = "checkpoint-019-materializer-v1"

    def __init__(self, events):
        self.events = events
        self.calls = []

    def materialize(self, tensor_plan, device):
        self.events.append("materialize")
        tensor = ("fake-tensor", tensor_plan.role, tensor_plan.values, device)
        self.calls.append((tensor_plan, device, tensor))
        return tensor


def framework_inputs(kv_length=128):
    spec = AttentionPlanSpec(
        mode=AttentionMode.BATCH_MIXED_PAGED,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim_qk=128,
        head_dim_vo=128,
        kv_layout=KVLayout.HND,
        causal=True,
        q_dtype="bfloat16",
        kv_dtype="bfloat16",
        o_dtype="bfloat16",
    )
    metadata = MixedPagedKVMetadata(
        qo_indptr=(0, 1),
        kv_indptr=(0, 1),
        kv_indices=(7,),
        kv_len_arr=(kv_length,),
        page_size=128,
    )
    return spec, metadata


def build_components(*, package_version="1.0.0", gate_reasons=()):
    events = []
    operation = fake_operation()
    catalog = AttentionOperatorOperationCatalog(
        name="checkpoint-019-catalog", operations=(operation,)
    )
    loader = FakePackageLoader(events, version=package_version)
    package_resolver = AttentionOperatorPackageResolver(
        catalog,
        AttentionOperatorPackageCompatibility(
            provider_id="cann",
            operation_id=operation.operation_id,
            adapter_version="checkpoint-019-adapter-v1",
            supported_package_versions=("1.0.0",),
        ),
        loader,
    )
    gate = FakePlanGate(events, gate_reasons)
    authority = FakeAuthorityResolver(events)
    factory = FakeLogicalFactory(events)
    materializer = FakeTensorMaterializer(events)
    implementation = AttentionOperatorPackageRuntimeImplementation(
        priority=100,
        package_resolver=package_resolver,
        plan_gate=gate,
        authority_resolver=authority,
        logical_factory=factory,
        logical_run_adapter=FakeLogicalRunAdapter(),
        tensor_materializer=materializer,
    )
    return {
        "events": events,
        "operation": operation,
        "catalog": catalog,
        "loader": loader,
        "gate": gate,
        "authority": authority,
        "factory": factory,
        "materializer": materializer,
        "implementation": implementation,
    }


def framework_plan():
    spec, metadata = framework_inputs()
    return AttentionFrameworkSession(spec.mode).plan(spec, metadata)


def callable_authority(operation_id):
    operation = load_packaged_attention_operator_catalog().get(operation_id)
    probe = AttentionOperatorProviderProbe(
        provider_id=operation.provider_id,
        adapter_version="test-adapter",
        available=True,
        package_versions=((operation.package_name, "test-package"),),
    )
    observed = AttentionObservedOperatorCallable(
        provider_id=operation.provider_id,
        package_name=operation.package_name,
        package_version="test-package",
        callable_path=operation.callable_path,
        api_version=operation.api_version,
        available=True,
        signature=AttentionObservedCallableSignature(
            operation.positional_arguments,
            operation.keyword_arguments,
            observation_kind="test-adapter",
        ),
    )
    return probe, bind_attention_operator_callable(probe, operation, observed)


def runtime_registry(components):
    implementations = AttentionOperatorRuntimeImplementationRegistry((components["implementation"],))
    return AttentionOperatorRuntimeResolverRegistry((("npu", implementations),))


def batch_runtime(components):
    return AttentionOperatorBatchRuntime("npu:0", runtime_registry(components), components["catalog"])


def install_provider(test, registry, catalog):
    previous = attention_operator_runtime_registry_snapshot()
    test.addCleanup(install_attention_operator_runtime_resolvers, previous.registry,
                    operation_catalog=previous.operation_catalog)
    install_attention_operator_runtime_resolvers(registry, operation_catalog=catalog)


class ProviderRuntimeTestCase(unittest.TestCase):
    def setUp(self):
        self.components = build_components()
        install_provider(self, runtime_registry(self.components), self.components["catalog"])
        package_attention.calls.clear()
