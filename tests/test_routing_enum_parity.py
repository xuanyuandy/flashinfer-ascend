import pytest

from flashinfer import RoutingMethodType
from flashinfer.runtime.reference_kernels import _route_topk
from flashinfer.tllm_enums import RoutingMethodType as CanonicalRoutingMethodType


def test_enum_members_and_eval_repr_match_pinned_upstream():
    assert RoutingMethodType is CanonicalRoutingMethodType
    assert {m.name: m.value for m in RoutingMethodType} == {
        "Default": 0, "Renormalize": 1, "DeepSeekV3": 2, "Llama4": 3,
        "RenormalizeNaive": 4, "TopK": 5, "SigmoidRenorm": 6, "MiniMax2": 7,
        "Sigmoid": 8, "TopKSigmoid": 9, "Unspecified": 10,
    }
    for member in RoutingMethodType:
        assert eval(repr(member), {"RoutingMethodType": RoutingMethodType}) is member
    assert RoutingMethodType(5) is RoutingMethodType.TopK


@pytest.mark.parametrize("member", list(RoutingMethodType)[5:])
def test_valid_but_unimplemented_routes_fail_before_tensor_computation(member):
    with pytest.raises(NotImplementedError, match=member.name):
        _route_topk(None, None, 2, 1, 1, 1.0, member)
