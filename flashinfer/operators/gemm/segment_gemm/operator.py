"""Backend-independent declaration for gemm.segment_gemm."""

from flashinfer.operators.definition import OperatorDefinition

OPERATOR = OperatorDefinition(name='segment_gemm', domain='gemm', op='gemm.segment_gemm', public_entry='flashinfer.gemm:SegmentGEMMWrapper')
