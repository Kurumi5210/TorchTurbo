from copy import deepcopy
from functools import wraps

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from torch_turbo.transformer.transformer_config import TransformerConfig
from torch_turbo.transformer.utils import divide
from torch_turbo.distributed.parallel import get_expert_tensor_parallel_world_size
from torch_turbo.utils.device import is_npu_available

try:
    import grouped_gemm
except ImportError:
    grouped_gemm = None

def grouped_gemm_is_available():
    """Check if grouped_gemm is available."""
    return grouped_gemm is not None

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

def conditional_jit_fuser(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    if not is_npu_available:
        jit_fuser = torch.compile
        wrapper = jit_fuser(wrapper)
    return wrapper


class GroupedMLP(torch.nn.Module):
    """An efficient implementation of the Experts layer using GroupedGEMM.

    Executes multiple experts in parallel to maximize computational efficiency.
    """

    def __init__(self, num_local_experts: int, use_fused_swiglu: bool, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.num_local_experts = num_local_experts
        assert (
            config.add_bias_linear == False
        ), "bias not supported in Grouped GEMM yet, please set '--disable-bias-linear' instead."

        self.expert_parallel = config.expert_model_parallel_size > 1
        if self.config.gated_linear_unit:
            if self.config.activation_func not in (F.silu, F.gelu):
                raise ValueError("Activation function must be silu or gelu when using GroupedMLP.")

            @conditional_jit_fuser
            def glu(x, probs):
                if use_fused_swiglu:
                    # use npu fused kernel for glu
                    print("fused swiglu activated...")
                    import torch_npu
                    return torch_npu.npu_swiglu(x, dim=-1)
                x = torch.chunk(x, 2, dim=-1)
                return self.config.activation_func(x[0]) * x[1]

            self.activation_func = glu
        else:
            self.activation_func = self.config.activation_func

        self.activation_func_with_probs = self.activation_func

        # How many feature each rank holds for fc1 and fc2, respectively.
        tp_size = get_expert_tensor_parallel_world_size()

        if config.gated_linear_unit:
            fc1_output_size = self.config.moe_ffn_hidden_size * 2
        else:
            fc1_output_size = self.config.moe_ffn_hidden_size
        fc1_output_size_per_partition = divide(fc1_output_size, tp_size)

        fc2_input_size = self.config.moe_ffn_hidden_size * self.num_local_experts
        fc2_input_size_per_partition = divide(fc2_input_size, tp_size)

        self.num_local_experts = config.num_moe_experts // config.expert_model_parallel_size
        # Note: The current kernel implementations of grouped_gemm
        # does not support transposition with CUTLASS grouped GEMM
        # (https://github.com/fanshiqing/grouped_gemm/blob/main/csrc/grouped_gemm.cu#L355-L358)
        # and as a result we avoid allocate the transpose of weights.
        # Initialize weight.
        if config.use_cpu_initialization:
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
                    dtype=config.params_dtype
                )
            )
        setattr(self.weight1, 'allreduce', not self.expert_parallel)
        setattr(self.weight2, 'allreduce', not self.expert_parallel)

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
            permuted_local_hidden_states = permuted_local_hidden_states.to(original_dtype)
            # Probs already applied, so reset to 1.
            permuted_probs = torch.ones_like(permuted_probs)

        if permuted_local_hidden_states.nelement() != 0:
            # Reshape the weights for the grouped GEMMs.
            w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
            w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

            fc1_output = gg_ops.gmm(
                permuted_local_hidden_states, w1, tokens_per_expert, trans_b=False
            )
            intermediate_parallel = self.activation_func_with_probs(
                fc1_output, permuted_probs.unsqueeze(-1)
            )
            fc2_output = gg_ops.gmm(intermediate_parallel, w2, tokens_per_expert, trans_b=False)
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
