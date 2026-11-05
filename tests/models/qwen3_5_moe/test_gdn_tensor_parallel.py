"""Qwen3.5 GatedDeltaNet tensor parallelism through the shared GDN forward."""

import copy
from datetime import timedelta
from types import MethodType, SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.models.modules.attention.gdn.test_forward import _cpu_dispatch


def _worker(rank, path):
    import torchturbo.ops as turbo_ops
    from torchturbo.engine.backends.fsdp2.parallel_context import ParallelContext
    from torchturbo.models.modules.attention.gdn import build_gdn_sp_forward
    from torchturbo.models.qwen3_5_moe import modeling_qwen3_5_moe as moe
    from torchturbo.models.qwen3_5_moe.tensor_parallel import _parallelize_gdn
    from torchturbo.ops.dispatcher import OpDispatcher

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{path}", rank=rank,
                            world_size=2, timeout=timedelta(seconds=120))
    try:
        # Only the accelerator kernels are replaced; DTensor collectives stay real.
        dispatcher = OpDispatcher.__new__(OpDispatcher)
        dispatcher._dispatch = _cpu_dispatch
        turbo_ops.ops = dispatcher
        moe.FusedRMSNormGated = None
        context = ParallelContext("cpu", sp_size=1, ep_size=1, tp_size=2)
        config = SimpleNamespace(
            hidden_size=8, linear_num_value_heads=4, linear_num_key_heads=2,
            linear_key_head_dim=3, linear_value_head_dim=2, linear_conv_kernel_dim=3,
            hidden_act="silu", rms_norm_eps=1e-6, dtype=None,
        )
        torch.manual_seed(11)
        reference = moe.Qwen3_5MoeGatedDeltaNet(config, layer_idx=0)
        with torch.no_grad():
            for parameter in reference.parameters():
                parameter.normal_(std=.3)
        module = copy.deepcopy(reference)
        _parallelize_gdn(module, context.tp_mesh)
        assert (module.num_k_heads, module.num_v_heads) == (1, 2)
        forward = build_gdn_sp_forward(None, 1, 0)
        for target in (module, reference):
            target.forward = MethodType(forward, target)

        x = torch.randn(1, 6, 8, generator=torch.Generator().manual_seed(1))
        boundaries = torch.tensor([0, 2, 6], dtype=torch.int32)
        local_x, reference_x = x.clone().requires_grad_(), x.clone().requires_grad_()
        output = module(local_x, cu_seq_lens=boundaries)
        expected = reference(reference_x, cu_seq_lens=boundaries)
        torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-6)
        probe = torch.randn(expected.shape, generator=torch.Generator().manual_seed(2))
        (output * probe).sum().backward()
        (expected * probe).sum().backward()
        torch.testing.assert_close(local_x.grad, reference_x.grad, rtol=1e-5, atol=1e-6)
        # Replicated conv/norm weights accumulate partial head-shard gradients.
        for name, parameter in module.named_parameters():
            torch.testing.assert_close(parameter.grad.full_tensor(), reference.get_parameter(name).grad,
                                       rtol=1e-5, atol=1e-6, msg=name)
    finally:
        dist.destroy_process_group()


def test_gdn_tensor_parallel_matches_dense_forward_and_gradients(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "store"),), nprocs=2, join=True)
