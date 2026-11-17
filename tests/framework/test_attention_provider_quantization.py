"""attention provider quantization regression coverage."""


import unittest
from dataclasses import replace

from flashinfer.attention import (
    AttentionFrameworkSession,
    AttentionOperatorQuantArgumentBinding,
    AttentionOperatorQuantizationBinding,
    AttentionOperatorQuantizationPlanGate,
    build_attention_operator_package_runtime,
    validate_attention_operator_quantization_bindings,
)
from flashinfer.runtime import SchemaError
from tests.framework.test_attention_capability import functional_profile, group_plan
from tests.framework.test_attention_runtime_bootstrap import bootstrap_components


def argument(source, name):
    return AttentionOperatorQuantArgumentBinding(source, name)


def binding_for(quant_spec, operation, *, arguments=None, **policies):
    if arguments is None:
        arguments = (
            argument("kv.key.scale", "key_scale"),
            argument("kv.value.scale", "value_scale"),
        )
    return AttentionOperatorQuantizationBinding(
        provider_id=operation.provider_id,
        operation_id=operation.operation_id,
        quant_spec=quant_spec,
        argument_bindings=arguments,
        **policies,
    )


class OperatorQuantizationBindingTests(unittest.TestCase):
    """Checkpoint 024: capability QuantSpec must close over catalog arguments."""

    def test_exact_symmetric_binding_is_canonical_and_order_independent(self):
        values = bootstrap_components()
        quant_spec = functional_profile().rules[0].quant_specs[0]
        first = binding_for(quant_spec, values["operation"])
        second = binding_for(
            quant_spec,
            values["operation"],
            arguments=tuple(reversed(first.argument_bindings)),
        )

        self.assertEqual(first, second)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(
            first.arguments_by_source,
            {"kv.key.scale": "key_scale", "kv.value.scale": "value_scale"},
        )
        self.assertEqual(len(first.fingerprint), 64)

    def test_asymmetric_binding_requires_independent_key_and_value_zero_points(self):
        values = bootstrap_components()
        symmetric = functional_profile().rules[0].quant_specs[0]
        asymmetric = replace(symmetric, scheme="asymmetric", has_zero_point=True)

        with self.assertRaisesRegex(SchemaError, "missing source.*zero_point"):
            binding_for(asymmetric, values["operation"])

        complete = binding_for(
            asymmetric,
            values["operation"],
            arguments=(
                argument("kv.key.scale", "key_scale"),
                argument("kv.value.scale", "value_scale"),
                argument("kv.key.zero_point", "key_zero"),
                argument("kv.value.zero_point", "value_zero"),
            ),
        )
        self.assertIn("kv.key.zero_point", complete.arguments_by_source)
        self.assertIn("kv.value.zero_point", complete.arguments_by_source)

    def test_runtime_scale_policy_must_match_explicit_argument_source(self):
        values = bootstrap_components()
        quant_spec = functional_profile().rules[0].quant_specs[0]
        with self.assertRaisesRegex(SchemaError, "policy does not match"):
            binding_for(
                quant_spec,
                values["operation"],
                runtime_k_scale_policy="argument",
            )
        with self.assertRaisesRegex(SchemaError, "policy does not match"):
            binding_for(
                quant_spec,
                values["operation"],
                arguments=(
                    argument("kv.key.scale", "key_scale"),
                    argument("kv.value.scale", "value_scale"),
                    argument("run.k_scale", "runtime_key_scale"),
                ),
            )

    def test_profile_and_operation_binding_validate_as_an_exact_set(self):
        values = bootstrap_components()
        validated = validate_attention_operator_quantization_bindings(
            values["operation"],
            values["spec"].profiles,
            values["spec"].quantization_bindings,
        )

        self.assertEqual(validated, values["spec"].quantization_bindings)
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_bootstrap_rejects_quant_capability_without_api_binding(self):
        values = bootstrap_components()
        incomplete = replace(values["spec"], quantization_bindings=())

        with self.assertRaisesRegex(
            SchemaError, "capability QuantSpec has no API argument binding"
        ):
            build_attention_operator_package_runtime(
                incomplete,
                operation_catalog=values["catalog"],
                package_loader=values["loader"],
            )
        self.assertEqual(values["events"], [])

    def test_non_catalog_argument_and_extra_quantspec_are_rejected(self):
        values = bootstrap_components()
        quant_spec = functional_profile().rules[0].quant_specs[0]
        non_catalog = binding_for(
            quant_spec,
            values["operation"],
            arguments=(
                argument("kv.key.scale", "not_in_catalog"),
                argument("kv.value.scale", "value_scale"),
            ),
        )
        with self.assertRaisesRegex(SchemaError, "non-catalog argument"):
            validate_attention_operator_quantization_bindings(
                values["operation"], values["spec"].profiles, (non_catalog,)
            )

        other_spec = replace(
            quant_spec, granularity="tensor", group_size=None, axis=None
        )
        extra = binding_for(other_spec, values["operation"])
        with self.assertRaisesRegex(
            SchemaError, "binding has no capability QuantSpec"
        ):
            validate_attention_operator_quantization_bindings(
                values["operation"],
                values["spec"].profiles,
                values["spec"].quantization_bindings + (extra,),
            )
        foreign = replace(
            values["spec"].quantization_bindings[0],
            provider_id="flash_attention_npu",
        )
        with self.assertRaisesRegex(SchemaError, "binding identity differs"):
            AttentionOperatorQuantizationPlanGate(
                values["gate"], values["operation"], (foreign,)
            )
        with self.assertRaisesRegex(SchemaError, "duplicate QuantSpec"):
            AttentionOperatorQuantizationPlanGate(
                values["gate"],
                values["operation"],
                (
                    values["spec"].quantization_bindings[0],
                    values["spec"].quantization_bindings[0],
                ),
            )

    def test_composed_gate_accepts_only_the_exact_bound_quantspec(self):
        values = bootstrap_components()
        implementation = build_attention_operator_package_runtime(
            values["spec"],
            operation_catalog=values["catalog"],
            package_loader=values["loader"],
        )
        exact_plan = group_plan()
        original = exact_plan.spec.kv_quant_spec
        different = replace(
            original, granularity="tensor", group_size=None, axis=None
        )
        different_spec = replace(exact_plan.spec, kv_quant_spec=different)
        different_plan = AttentionFrameworkSession(different_spec.mode).plan(
            different_spec, exact_plan.metadata
        )

        self.assertEqual(
            implementation.rejection_reasons(exact_plan, "npu:0"), ()
        )
        reasons = implementation.rejection_reasons(different_plan, "npu:0")
        self.assertTrue(
            any("no exact KV QuantSpec argument binding" in item for item in reasons)
        )
        self.assertEqual(values["loader"].resolve_calls, 0)

    def test_quantization_gate_preserves_base_reasons_deterministically(self):
        values = bootstrap_components(gate_reasons=("layout rejected",))
        gate = AttentionOperatorQuantizationPlanGate(
            values["gate"],
            values["operation"],
            values["spec"].quantization_bindings,
        )
        plan = group_plan()

        self.assertEqual(
            gate.rejection_reasons(plan, "npu"), ("layout rejected",)
        )
        self.assertEqual(values["loader"].version_calls, 0)
        self.assertEqual(values["loader"].resolve_calls, 0)


from flashinfer.attention import (
    AttentionOperatorOperationCatalog,
    AttentionOperatorQuantArgumentBinding,
    AttentionOperatorQuantizationBinding,
    AttentionOperatorQuantizedKVInput,
    AttentionOperatorWrapperSession,
    TensorView,
    build_attention_operator_package_runtime,
    contiguous_strides,
    dtype_itemsize,
    infer_quant_scale_shape,
    infer_quant_storage_shape,
)
from tests.framework.test_attention_capability import group_plan
from tests.framework.test_package_runtime_integration import package_attention


def active_session(values, *, spec=None, catalog=None, plan=None):
    spec = values["spec"] if spec is None else spec
    catalog = values["catalog"] if catalog is None else catalog
    implementation = build_attention_operator_package_runtime(
        spec,
        operation_catalog=catalog,
        package_loader=values["loader"],
    )
    plan = group_plan() if plan is None else plan
    resolved = implementation.resolve(plan, "npu:0")
    session = AttentionOperatorWrapperSession(catalog)
    session.plan(
        resolved.factory,
        resolved.run_adapter,
        plan,
        resolved.receipt,
        resolved.selection,
        resolved.callable_binding,
    )
    return plan, session


class MetadataTensor:
    def __init__(self, tensor_view):
        self.tensor_view = tensor_view


def metadata_tensor(name, shape, dtype, *, device="npu:0", strides=None):
    shape = tuple(shape)
    numel = 1
    for dim in shape:
        numel *= dim
    return MetadataTensor(
        TensorView(
            shape=shape,
            strides=contiguous_strides(shape) if strides is None else tuple(strides),
            dtype=dtype,
            device=device,
            storage_id="checkpoint-025:" + name,
            storage_nbytes=numel * dtype_itemsize(dtype),
        )
    )


def quantized_input(quant_spec, **overrides):
    key_shape = (2, 2, 1, 3)
    value_shape = (2, 2, 1, 2)
    storage_dtype = (
        "uint8"
        if quant_spec.storage_dtype in {"int4_packed", "uint4_packed"}
        else quant_spec.storage_dtype
    )
    values = {
        "quant_spec": quant_spec,
        "key_storage": metadata_tensor(
            "key-storage",
            infer_quant_storage_shape(key_shape, quant_spec),
            storage_dtype,
        ),
        "value_storage": metadata_tensor(
            "value-storage",
            infer_quant_storage_shape(value_shape, quant_spec),
            storage_dtype,
        ),
        "key_scale": metadata_tensor(
            "key-scale",
            infer_quant_scale_shape(key_shape, quant_spec),
            quant_spec.scale_dtype,
        ),
        "value_scale": metadata_tensor(
            "value-scale",
            infer_quant_scale_shape(value_shape, quant_spec),
            quant_spec.scale_dtype,
        ),
    }
    values.update(overrides)
    return AttentionOperatorQuantizedKVInput(**values)


class QuantizedProviderRunLoweringTests(unittest.TestCase):
    """Checkpoint 025: bound quant inputs lower without executing a package."""

    def setUp(self):
        package_attention.calls[:] = []

    def test_symmetric_input_requires_independent_storage_and_scale_objects(self):
        quant_spec = group_plan().spec.kv_quant_spec
        value = quantized_input(quant_spec)

        self.assertIsNot(value.key_storage, value.value_storage)
        self.assertIsNot(value.key_scale, value.value_scale)
        for field_name in (
            "key_storage",
            "value_storage",
            "key_scale",
            "value_scale",
        ):
            with self.subTest(field_name=field_name):
                with self.assertRaisesRegex(SchemaError, "requires %s" % field_name):
                    quantized_input(quant_spec, **{field_name: None})

    def test_zero_point_presence_must_match_exact_quant_spec(self):
        symmetric = group_plan().spec.kv_quant_spec
        with self.assertRaisesRegex(SchemaError, "cannot carry zero points"):
            quantized_input(symmetric, key_zero_point=object())

        asymmetric = replace(symmetric, scheme="asymmetric", has_zero_point=True)
        with self.assertRaisesRegex(SchemaError, "requires independent zero points"):
            quantized_input(asymmetric)
        value = quantized_input(
            asymmetric,
            key_zero_point=object(),
            value_zero_point=object(),
        )
        self.assertIsNotNone(value.key_zero_point)
        self.assertIsNotNone(value.value_zero_point)

    def test_bootstrap_lowering_unwraps_storage_and_injects_bound_scales(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        kv_input = quantized_input(plan.spec.kv_quant_spec)

        lowered = session.run("query", kv_input)

        positional = dict(lowered.positional_arguments)
        keywords = dict(lowered.keyword_arguments)
        self.assertIs(positional["key"], kv_input.key_storage)
        self.assertIs(positional["value"], kv_input.value_storage)
        self.assertIs(keywords["key_scale"], kv_input.key_scale)
        self.assertIs(keywords["value_scale"], kv_input.value_scale)
        self.assertEqual(
            lowered.consumed_request_fields,
            ("query", "kv_cache", "return_lse", "logits_soft_cap"),
        )
        self.assertEqual(package_attention.calls, [])

    def test_plain_or_different_quantspec_input_is_rejected_before_base_lowering(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        with self.assertRaisesRegex(SchemaError, "QuantizedKVInput"):
            session.run("query", ("key", "value"))

        different = replace(
            plan.spec.kv_quant_spec,
            granularity="tensor",
            group_size=None,
            axis=None,
        )
        with self.assertRaisesRegex(SchemaError, "does not match"):
            session.run("query", quantized_input(different))
        self.assertEqual(len(values["materializer"].calls), 1)
        self.assertEqual(package_attention.calls, [])

    def test_default_binding_rejects_public_runtime_scale_multipliers(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        kv_input = quantized_input(plan.spec.kv_quant_spec)

        for name in ("k_scale", "v_scale"):
            with self.subTest(name=name):
                with self.assertRaisesRegex(SchemaError, "rejects run-time"):
                    session.run("query", kv_input, **{name: object()})
        self.assertEqual(package_attention.calls, [])

    def test_explicit_runtime_scale_policy_injects_separate_arguments(self):
        values = bootstrap_components()
        original = values["spec"].quantization_bindings[0]
        extended = replace(
            original,
            argument_bindings=original.argument_bindings
            + (
                AttentionOperatorQuantArgumentBinding(
                    "run.k_scale", "runtime_key_scale"
                ),
                AttentionOperatorQuantArgumentBinding(
                    "run.v_scale", "runtime_value_scale"
                ),
            ),
            runtime_k_scale_policy="argument",
            runtime_v_scale_policy="argument",
        )
        spec = replace(values["spec"], quantization_bindings=(extended,))
        plan, session = active_session(values, spec=spec)
        kv_input = quantized_input(plan.spec.kv_quant_spec)
        runtime_k_scale = object()
        runtime_v_scale = object()

        lowered = session.run(
            "query",
            kv_input,
            k_scale=runtime_k_scale,
            v_scale=runtime_v_scale,
        )

        keywords = dict(lowered.keyword_arguments)
        self.assertIs(keywords["runtime_key_scale"], runtime_k_scale)
        self.assertIs(keywords["runtime_value_scale"], runtime_v_scale)
        self.assertEqual(
            lowered.consumed_request_fields,
            (
                "query",
                "kv_cache",
                "return_lse",
                "k_scale",
                "v_scale",
                "logits_soft_cap",
            ),
        )
        self.assertEqual(package_attention.calls, [])

    def test_injected_argument_cannot_overwrite_base_provider_lowering(self):
        values = bootstrap_components()
        operation = replace(
            values["operation"],
            quant_arguments=values["operation"].quant_arguments + ("scale",),
        )
        catalog = AttentionOperatorOperationCatalog(
            name="checkpoint-025-collision", operations=(operation,)
        )
        original = values["spec"].quantization_bindings[0]
        collision = replace(
            original,
            argument_bindings=(
                AttentionOperatorQuantArgumentBinding("kv.key.scale", "scale"),
                AttentionOperatorQuantArgumentBinding(
                    "kv.value.scale", "value_scale"
                ),
            ),
        )
        spec = replace(values["spec"], quantization_bindings=(collision,))
        plan, session = active_session(values, spec=spec, catalog=catalog)

        with self.assertRaisesRegex(SchemaError, "collides.*scale"):
            session.run("query", quantized_input(plan.spec.kv_quant_spec))
        self.assertEqual(package_attention.calls, [])

    def test_lowering_reuses_one_plan_materialization_without_callable_execution(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        kv_input = quantized_input(plan.spec.kv_quant_spec)

        first = session.run("query-1", kv_input)
        second = session.run("query-2", kv_input)

        self.assertEqual(len(values["materializer"].calls), 1)
        self.assertIs(
            dict(first.keyword_arguments)["table"],
            dict(second.keyword_arguments)["table"],
        )
        self.assertEqual(package_attention.calls, [])


from flashinfer.attention import (
    AttentionOperatorPackageRuntimeSpec,
    AttentionOperatorQuantizedKVInput,
    TensorView,
    inspect_attention_operator_quantized_kv_input,
)


def with_view(tensor, **changes):
    return type(tensor)(replace(tensor.tensor_view, **changes))


class InvalidMetadataInspector:
    def to_view(self, tensor, *, name, writable=False):
        return object()


class QuantizedProviderTensorMetadataTests(unittest.TestCase):
    """Checkpoint 026: provider KV metadata is proven before lowering."""

    def test_valid_input_is_inspected_before_non_executing_lowering(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        kv_input = quantized_input(plan.spec.kv_quant_spec)

        lowered = session.run("query", kv_input)

        calls = values["tensor_metadata_inspector"].calls
        self.assertEqual(
            tuple(name for _, name, _ in calls),
            (
                "kv.key.storage",
                "kv.value.storage",
                "kv.key.scale",
                "kv.value.scale",
            ),
        )
        self.assertTrue(all(writable is False for _, _, writable in calls))
        self.assertIs(dict(lowered.positional_arguments)["key"], kv_input.key_storage)

    def test_storage_dtype_and_logical_shape_must_match_plan(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        valid = quantized_input(plan.spec.kv_quant_spec)

        wrong_dtype = with_view(valid.key_storage, dtype="uint8")
        with self.assertRaisesRegex(SchemaError, "storage view dtype"):
            session.run("query", replace(valid, key_storage=wrong_dtype))

        too_few_pages = metadata_tensor(
            "too-few-pages", (1, 2, 1, 3), "int8"
        )
        with self.assertRaisesRegex(SchemaError, "every referenced page"):
            session.run("query", replace(valid, key_storage=too_few_pages))

    def test_scale_shape_and_dtype_must_match_exact_quant_spec(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        valid = quantized_input(plan.spec.kv_quant_spec)

        wrong_shape = metadata_tensor("wrong-scale-shape", (2, 2, 1, 1), "float32")
        with self.assertRaisesRegex(SchemaError, "scale view shape"):
            session.run("query", replace(valid, key_scale=wrong_shape))

        wrong_dtype = with_view(valid.value_scale, dtype="float16", storage_nbytes=8)
        with self.assertRaisesRegex(SchemaError, "scale view dtype"):
            session.run("query", replace(valid, value_scale=wrong_dtype))

    def test_asymmetric_zero_points_require_int32_exact_shapes(self):
        plan = group_plan()
        asymmetric = replace(
            plan.spec.kv_quant_spec, scheme="asymmetric", has_zero_point=True
        )
        asymmetric_plan = replace(
            plan, spec=replace(plan.spec, kv_quant_spec=asymmetric)
        )
        value = quantized_input(
            asymmetric,
            key_zero_point=metadata_tensor(
                "key-zero", (2, 2, 1, 2), "int32"
            ),
            value_zero_point=metadata_tensor(
                "value-zero", (2, 2, 1, 1), "int32"
            ),
        )
        inspector = bootstrap_components()["tensor_metadata_inspector"]

        inspected = inspect_attention_operator_quantized_kv_input(
            asymmetric_plan, value, inspector, "npu:0"
        )
        self.assertTrue(inspected.quantized)

        wrong = with_view(value.key_zero_point, dtype="int16")
        with self.assertRaisesRegex(SchemaError, "zero_point view dtype"):
            inspect_attention_operator_quantized_kv_input(
                asymmetric_plan,
                replace(value, key_zero_point=wrong),
                inspector,
                "npu:0",
            )

    def test_non_contiguous_components_are_rejected_before_lowering(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        valid = quantized_input(plan.spec.kv_quant_spec)
        non_contiguous = with_view(
            valid.key_storage,
            strides=(12, 6, 3, 1),
            storage_nbytes=21,
        )

        with self.assertRaisesRegex(SchemaError, "must be contiguous"):
            session.run("query", replace(valid, key_storage=non_contiguous))

    def test_every_component_must_use_the_resolved_device(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        valid = quantized_input(plan.spec.kv_quant_spec)
        foreign = with_view(valid.value_scale, device="npu:1")

        with self.assertRaisesRegex(SchemaError, "share device"):
            session.run("query", replace(valid, value_scale=foreign))

        all_foreign = {
            name: with_view(getattr(valid, name), device="npu:1")
            for name in ("key_storage", "value_storage", "key_scale", "value_scale")
        }
        with self.assertRaisesRegex(SchemaError, "device does not match"):
            session.run("query", replace(valid, **all_foreign))

    def test_component_and_kv_storage_aliases_are_rejected(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        valid = quantized_input(plan.spec.kv_quant_spec)

        aliased_scale = with_view(
            valid.key_scale,
            storage_id=valid.key_storage.tensor_view.storage_id,
        )
        with self.assertRaisesRegex(SchemaError, "components.*alias"):
            session.run("query", replace(valid, key_scale=aliased_scale))

        aliased_value = with_view(
            valid.value_storage,
            storage_id=valid.key_storage.tensor_view.storage_id,
        )
        with self.assertRaisesRegex(SchemaError, "K and V.*alias"):
            session.run("query", replace(valid, value_storage=aliased_value))

    def test_bootstrap_requires_a_real_inspector_and_rejects_invalid_output(self):
        values = bootstrap_components()
        with self.assertRaisesRegex(SchemaError, "require.*metadata inspector"):
            replace(values["spec"], tensor_metadata_inspector=None)

        invalid = replace(
            values["spec"], tensor_metadata_inspector=InvalidMetadataInspector()
        )
        plan, session = active_session(values, spec=invalid)
        with self.assertRaisesRegex(TypeError, "must return TensorView"):
            session.run("query", quantized_input(plan.spec.kv_quant_spec))


from flashinfer.attention import (
    AttentionFrameworkSession,
    AttentionDispatchError,
    AttentionOperatorQuantizedKVInput,
    QuantPhysicalAxisTransform,
    QuantPhysicalLayoutCatalog,
    QuantPhysicalLayoutDescriptor,
    attention_kernel_binary_abi,
    attention_kernel_binary_abi_v2,
    inspect_attention_operator_quantized_kv_input,
    validate_attention_operator_quant_physical_layouts,
)
from tests.framework.test_attention_capability import attention_launch_abi, bound_kernel, functional_profile, group_plan


def physical_descriptor(*, layout_id="checkpoint027.blocked", page_block=1):
    page_axes = ("o0", "i0") if page_block > 1 else ("o0",)
    return QuantPhysicalLayoutDescriptor(
        layout_id=layout_id,
        storage_dtypes=("int8",),
        storage_transform=QuantPhysicalAxisTransform(
            4,
            (page_block, 1, 1, 2),
            page_axes + ("o1", "o2", "o3", "i3"),
            required_alignment=16,
        ),
        storage_converter_id="checkpoint027.to_blocked",
        storage_inverse_converter_id="checkpoint027.from_blocked",
        required_features=("int8-group-dequant",),
    )


def aligned_tensor(name, shape, dtype):
    value = metadata_tensor(name, shape, dtype)
    return type(value)(replace(value.tensor_view, data_ptr_alignment=16))


def physical_input(quant_spec, descriptor, **overrides):
    key_shape = (2, 2, 1, 3)
    value_shape = (2, 2, 1, 2)
    key_shapes = descriptor.physical_shapes(key_shape, quant_spec)
    value_shapes = descriptor.physical_shapes(value_shape, quant_spec)
    values = {
        "quant_spec": quant_spec,
        "key_storage": aligned_tensor("physical-key", key_shapes.storage, "int8"),
        "value_storage": aligned_tensor(
            "physical-value", value_shapes.storage, "int8"
        ),
        "key_scale": metadata_tensor(
            "physical-key-scale", key_shapes.scale, quant_spec.scale_dtype
        ),
        "value_scale": metadata_tensor(
            "physical-value-scale", value_shapes.scale, quant_spec.scale_dtype
        ),
    }
    values.update(overrides)
    return AttentionOperatorQuantizedKVInput(**values)


def physical_runtime(*, descriptor=None, binary_abi=None):
    descriptor = physical_descriptor() if descriptor is None else descriptor
    values = bootstrap_components()
    base_plan = group_plan()
    quant_spec = replace(
        base_plan.spec.kv_quant_spec, physical_layout=descriptor.layout_id
    )
    plan = AttentionFrameworkSession(base_plan.spec.mode).plan(
        replace(base_plan.spec, kv_quant_spec=quant_spec), base_plan.metadata
    )
    base_profile = functional_profile()
    rule = replace(base_profile.rules[0], quant_specs=(quant_spec,))
    profile = replace(base_profile, rules=(rule,))
    selected_binary_abi = (
        attention_kernel_binary_abi_v2()
        if binary_abi is None
        else binary_abi
    )
    kernel = bound_kernel(
        profile,
        launch_abi=replace(
            attention_launch_abi(),
            abi_name=(
                "flashinfer.attention.v2"
                if selected_binary_abi.fingerprint
                == attention_kernel_binary_abi_v2().fingerprint
                else "flashinfer.attention.v1"
            ),
        ),
        binary_abi=selected_binary_abi,
    )
    binding = replace(
        values["spec"].quantization_bindings[0], quant_spec=quant_spec
    )
    spec = replace(
        values["spec"],
        profiles=(profile,),
        descriptors=(kernel,),
        observed_environment=profile.environment,
        quantization_bindings=(binding,),
        quant_physical_layout_catalog=QuantPhysicalLayoutCatalog((descriptor,)),
    )
    return values, plan, spec, quant_spec, descriptor


class ProviderPhysicalLayoutBindingTests(unittest.TestCase):
    """Checkpoint 027: non-logical KV needs exact descriptor and evidence."""

    def setUp(self):
        package_attention.calls[:] = []

    def test_catalog_set_must_exactly_cover_nonlogical_bindings(self):
        values, _, spec, _, descriptor = physical_runtime()
        binding = spec.quantization_bindings[0]
        with self.assertRaisesRegex(SchemaError, "no physical descriptor"):
            validate_attention_operator_quant_physical_layouts(
                (binding,), QuantPhysicalLayoutCatalog()
            )
        with self.assertRaisesRegex(SchemaError, "unbound descriptor"):
            validate_attention_operator_quant_physical_layouts(
                values["spec"].quantization_bindings,
                QuantPhysicalLayoutCatalog((descriptor,)),
            )

    def test_catalog_descriptor_must_support_bound_storage_dtype(self):
        _, _, spec, _, descriptor = physical_runtime()
        unsupported = replace(descriptor, storage_dtypes=("uint8",))
        with self.assertRaisesRegex(SchemaError, "bound storage dtype"):
            validate_attention_operator_quant_physical_layouts(
                spec.quantization_bindings,
                QuantPhysicalLayoutCatalog((unsupported,)),
            )

    def test_bootstrap_rejects_nonlogical_binding_without_catalog(self):
        values, _, spec, _, _ = physical_runtime()
        missing = replace(
            spec, quant_physical_layout_catalog=QuantPhysicalLayoutCatalog()
        )
        with self.assertRaisesRegex(SchemaError, "no physical descriptor"):
            active_session(values, spec=missing)

    def test_valid_physical_kv_metadata_is_inspectable_but_not_authorized(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime()
        value = physical_input(quant_spec, descriptor)

        inspected = inspect_attention_operator_quantized_kv_input(
            plan,
            value,
            values["tensor_metadata_inspector"],
            "npu:0",
            descriptor,
        )

        self.assertTrue(inspected.quantized)
        self.assertEqual(len(values["tensor_metadata_inspector"].calls), 4)
        with self.assertRaisesRegex(
            AttentionDispatchError, "requires exactly one physical-layout evidence"
        ):
            active_session(values, spec=spec, plan=plan)
        self.assertEqual(package_attention.calls, [])

    def test_physical_storage_shape_and_alignment_are_enforced(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime()
        value = physical_input(quant_spec, descriptor)
        wrong_shape = aligned_tensor("wrong-physical-shape", (2, 2, 1, 3), "int8")
        with self.assertRaisesRegex(SchemaError, "storage view shape"):
            inspect_attention_operator_quantized_kv_input(
                plan,
                replace(value, key_storage=wrong_shape),
                values["tensor_metadata_inspector"],
                "npu:0",
                descriptor,
            )

        misaligned = type(value.key_storage)(
            replace(value.key_storage.tensor_view, data_ptr_alignment=8)
        )
        with self.assertRaisesRegex(SchemaError, "16-byte aligned"):
            inspect_attention_operator_quantized_kv_input(
                plan,
                replace(value, key_storage=misaligned),
                values["tensor_metadata_inspector"],
                "npu:0",
                descriptor,
            )

    def test_paged_layout_must_preserve_an_exact_unblocked_page_axis(self):
        descriptor = physical_descriptor(
            layout_id="checkpoint027.blocked_page", page_block=2
        )
        values, plan, spec, quant_spec, descriptor = physical_runtime(
            descriptor=descriptor
        )
        with self.assertRaisesRegex(SchemaError, "preserve an exact page axis"):
            inspect_attention_operator_quantized_kv_input(
                plan,
                physical_input(quant_spec, descriptor),
                values["tensor_metadata_inspector"],
                "npu:0",
                descriptor,
            )

    def test_kernel_native_layout_requires_v2_binary_abi_evidence(self):
        values, plan, spec, quant_spec, descriptor = physical_runtime(
            binary_abi=attention_kernel_binary_abi()
        )
        with self.assertRaisesRegex(SchemaError, "requires KV POD v2"):
            active_session(values, spec=spec, plan=plan)
        self.assertEqual(package_attention.calls, [])

    def test_logical_binding_still_uses_empty_catalog_path(self):
        values = bootstrap_components()
        plan, session = active_session(values)
        lowered = session.run("query", quantized_input(plan.spec.kv_quant_spec))
        self.assertEqual(lowered.provider_id, "cann")
        self.assertEqual(package_attention.calls, [])
