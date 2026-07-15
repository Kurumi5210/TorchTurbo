import os
from copy import deepcopy
from functools import wraps

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from .moe_config import MoEConfig
from torchturbo.utils.device import is_npu_available

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

try:
    import transformer_engine.pytorch as te
    from transformer_engine.pytorch import GroupedLinear
except ImportError:
    GroupedLinear = None


def grouped_gemm_is_available():
    """Check if grouped_gemm is available."""
    return grouped_gemm is not None


def transformer_engine_is_available():
    """Check if transformer_engine is available."""
    return GroupedLinear is not None


def assert_grouped_gemm_is_available():
    """Assert that grouped_gemm is available."""
    assert grouped_gemm_is_available(), (
        "Grouped GEMM is not available. Please run "
        "`pip install git+https://github.com/fanshiqing/grouped_gemm@v1.1.4`."
    )


if is_npu_available:
    from mindspeed.core.fusions.grouped_matmul import Ops

    gg_ops = Ops
else:
    assert_grouped_gemm_is_available()
    gg_ops = grouped_gemm.ops


def conditional_jit_fuser(use_fused_swiglu=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        if not is_npu_available and use_fused_swiglu:
            print("Using JIT fuser for fused swiglu.")
            jit_fuser = torch.compile
            wrapper = jit_fuser(wrapper)
        return wrapper

    return decorator


class GroupedMLP(torch.nn.Module):
    """An efficient implementation of the Experts layer using GroupedGEMM.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(
        self,
        num_local_experts: int,
        use_fused_swiglu: bool,
        use_te: bool,
        config: MoEConfig,
        layer_number: int = None,
    ):
        super().__init__()
        self.config = config
        self.num_local_experts = num_local_experts
        self.layer_number = layer_number
        limit_list = getattr(config, "moe_swiglu_limit", None)
        if (
            limit_list is not None
            and layer_number is not None
            and 0 <= layer_number < len(limit_list)
        ):
            self.swiglu_limit = float(limit_list[layer_number])
        else:
            self.swiglu_limit = 0.0

        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError(
                    "Activation function must be silu or gelu when using GroupedMLP."
                )

            @conditional_jit_fuser(use_fused_swiglu)
            def glu(x, probs):
                if use_fused_swiglu:
                    # use npu fused kernel for glu
                    import torch_npu

                    return torch_npu.npu_swiglu(x, dim=-1)
                gate, up = torch.chunk(x, 2, dim=-1)
                # Apply swiglu_limit if configured (DeepSeek V4 style).
                if self.swiglu_limit > 0:
                    up = torch.clamp(
                        up.float(), min=-self.swiglu_limit, max=self.swiglu_limit
                    )
                    gate = torch.clamp(gate.float(), max=self.swiglu_limit)
                return (self.config.activation_func(gate) * up).to(x.dtype)

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        self.activation_func_with_probs = self.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        if config.gated_linear_unit:
            fc1_output_size = self.config.moe_ffn_hidden_size * 2
        else:
            fc1_output_size = self.config.moe_ffn_hidden_size
        fc1_output_size_per_partition = fc1_output_size

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = fc2_input_size

        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        self.use_te = use_te
        if transformer_engine_is_available() and self.use_te:
            if torch.distributed.get_rank() == 0:
                print("Using TE GroupedLinear.")
            self.fc1_linear = GroupedLinear(
                num_gemms=self.num_local_experts,
                in_features=self.config.hidden_size,
                out_features=fc1_output_size_per_partition,
                bias=False,  # bias not supported in original implementation
                params_dtype=config.params_dtype,
                device="cuda",
            )

            self.fc2_linear = GroupedLinear(
                num_gemms=self.num_local_experts,
                in_features=fc2_input_size_per_partition // self.num_local_experts,
                out_features=self.config.hidden_size,
                bias=False,  # bias not supported in original implementation
                params_dtype=config.params_dtype,
                device="cuda",
            )
        else:
            if torch.distributed.get_rank() == 0:
                print("Using fused op grouped_matmul.")
            self.weight1 = Parameter(
                torch.empty(
                    self.config.hidden_size * self.num_local_experts,
                    fc1_output_size_per_partition,
                    dtype=config.params_dtype,
                )
            )
            self.weight2 = Parameter(
                torch.empty(
                    fc2_input_size_per_partition,
                    self.config.hidden_size,
                    dtype=config.params_dtype,
                )
            )

    def forward(
        self,
        permuted_local_hidden_states: torch.Tensor,
        tokens_per_expert: torch.Tensor,
        permuted_probs: torch.Tensor,
    ):
        """Forward step of the GroupedMLP."""
        if self.config.moe_apply_probs_on_input:
            assert (
                self.config.moe_router_topk == 1
            ), "`moe_apply_probs_on_input` only works with `moe_router_topk`=1."
            original_dtype = permuted_local_hidden_states.dtype
            permuted_local_hidden_states = (
                permuted_probs.unsqueeze(-1) * permuted_local_hidden_states
            )
            permuted_local_hidden_states = permuted_local_hidden_states.to(
                original_dtype
            )
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        # `tokens_per_expert` is the per-expert token count used as the grouped-GEMM
        # `batch_sizes`. The grouped_gemm C++ kernel hard-requires it on CPU
        # (`assert batch_sizes.is_cpu()`). During normal forward the AlltoAll token
        # dispatcher moves it to CPU via an async side-stream + sync point; but under
        # activation-checkpoint recompute (driven by autograd in backward) that side
        # stream/sync timing is not reliably hit, so the GPU tensor can leak through
        # and trip the is_cpu() assert. Pin it to CPU here at the call site instead of
        # relying on the dispatcher's async DtoH, so forward and recompute behave the
        # same. (The TE branch already does the equivalent via .tolist() below.)
        if isinstance(tokens_per_expert, torch.Tensor) and tokens_per_expert.is_cuda:
            tokens_per_expert = tokens_per_expert.cpu()

        if permuted_local_hidden_states.nelement() != 0:

            if transformer_engine_is_available() and self.use_te:
                # Use TransformerEngine GroupedLinear if available, otherwise fall back to grouped_gemm
                m_splits = tokens_per_expert.tolist()
                # First grouped linear: hidden_size -> fc1_output_size
                fc1_output = self.fc1_linear(permuted_local_hidden_states, m_splits)

                # Apply activation function
                intermediate_parallel = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )

                # Second grouped linear: fc2_input_size -> hidden_size
                fc2_output = self.fc2_linear(intermediate_parallel, m_splits)
            else:
                # Fall back to original grouped_gemm implementation
                # Reshape the weights for the grouped GEMMs.
                w1 = self.weight1.view(
                    self.num_local_experts, self.config.hidden_size, -1
                )
                w2 = self.weight2.view(
                    self.num_local_experts, -1, self.config.hidden_size
                )

                fc1_output = gg_ops.gmm(
                    permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
                )
                intermediate_parallel = self.activation_func_with_probs(
                    fc1_output, permuted_probs.unsqueeze(-1)
                )
                fc2_output = gg_ops.gmm(
                    intermediate_parallel, w2, tokens_per_expert, trans_b=False
                )
        else:
            # No token is allocated for local experts.
            assert torch.count_nonzero(tokens_per_expert) == 0

            # Make sure params of experts still have gradients even given zero tokens.
            w1 = self.weight1.view(self.config.hidden_size, -1)
            w2 = self.weight2.view(-1, self.config.hidden_size)
            h = torch.matmul(permuted_local_hidden_states, w1)
            h = self.activation_func_with_probs(h, permuted_probs.unsqueeze(-1))
            fc2_output = torch.matmul(h, w2)

        return fc2_output, None
