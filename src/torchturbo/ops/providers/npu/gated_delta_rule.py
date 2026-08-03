"""NPU (AscendC) provider for the chunk gated-delta-rule op.

Wraps the vendored AscendC kernel in ``ops/providers/npu/_fla_npu_gdn`` which is a
drop-in for fla's ``chunk_gated_delta_rule`` (same ``[B,T,H,D]`` q/k/v /
``[B,T,H]`` g/beta layout, same ``head_first=False`` return). priority=100 so
`auto` on NPU picks it over the fla native (priority=0) — this is the SP32
fla-Ascend crash mitigation: dense always uses AscendC on NPU.
"""

from __future__ import annotations

from torchturbo.runtime import TurboDevice

from ...base import OpImpl, OpImplKind
from ...registry import register_op


@register_op(
    op_name="gated_delta_rule",
    impl_name="ascendc_gated_delta_rule",
    kind=OpImplKind.CUSTOM,
    priority=100,
    devices=(TurboDevice.NPU,),
)
class AscendCGatedDeltaRule(OpImpl):
    def available(self) -> bool:
        try:
            import fla_npu  # noqa: F401
            from torchturbo.ops.providers.npu._fla_npu_gdn import (  # noqa: F401
                ascendc_chunk_gated_delta_rule,
            )
        except Exception as exc:  # pragma: no cover - diagnostic
            import traceback

            from torchturbo.logging import get_logger

            get_logger(__name__).warning(
                "ascendc_gated_delta_rule unavailable (will fall back to fla "
                "under auto): %r\n%s",
                exc,
                traceback.format_exc(),
            )
            return False
        return True

    def __call__(
        self,
        q,
        k,
        v,
        g,
        beta,
        *,
        initial_state=None,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
        cu_seqlens=None,
        **kwargs,
    ):
        from torchturbo.ops.providers.npu._fla_npu_gdn import (
            ascendc_chunk_gated_delta_rule,
        )

        return ascendc_chunk_gated_delta_rule(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            cu_seqlens=cu_seqlens,
        )