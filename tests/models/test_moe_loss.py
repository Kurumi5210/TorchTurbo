import logging
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from torchturbo.engine.metrics import MetricTracker
from torchturbo.models.modules.moe.moe_config import MoEConfig
from torchturbo.models.modules.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    compute_routing_scores_for_aux_loss,
)
from torchturbo.models.modules.moe.router import TopKRouter


def _router_config(**kwargs):
    values = {
        "hidden_size": 3,
        "num_moe_experts": 4,
        "perform_initialization": False,
        "moe_aux_loss_coeff": 0.1,
        "moe_router_topk": 2,
        "moe_router_pre_softmax": True,
    }
    values.update(kwargs)
    return MoEConfig(**values)


def _compute_token_aux_loss(router, logits, padding_mask=None):
    routing_map, scores = compute_routing_scores_for_aux_loss(
        logits,
        router.topk,
        router.score_function,
        padding_mask,
    )
    tokens_per_expert = routing_map.sum(dim=0)
    if router.sp_group is not None and dist.get_world_size(router.sp_group) > 1:
        dist.all_reduce(tokens_per_expert, group=router.sp_group)
    total_num_tokens = tokens_per_expert.sum() / router.topk
    return router._token_aux_loss(scores, tokens_per_expert, total_num_tokens)


def test_token_aux_loss_matches_switch_formula():
    logits = torch.tensor(
        [[2.0, 0.0, -1.0], [0.0, 1.0, 2.0], [1.0, 3.0, 0.0]],
        requires_grad=True,
    )
    scores = torch.softmax(logits, dim=-1)
    top_indices = torch.topk(scores, 2, dim=-1).indices
    routing_map = torch.zeros_like(scores).scatter_(1, top_indices, 1)
    expected = (
        (scores.sum(dim=0) * routing_map.sum(dim=0)).sum()
        * scores.shape[-1]
        / (scores.shape[0] ** 2 * 2)
    )

    router = TopKRouter(
        _router_config(
            num_moe_experts=3,
            moe_router_topk=2,
            moe_router_score_function="softmax",
        )
    )
    actual = _compute_token_aux_loss(router, logits)

    torch.testing.assert_close(actual, expected)


def test_router_aux_uses_raw_logits_not_dispatch_map():
    tracker = MetricTracker()
    router = TopKRouter(
        _router_config(
            hidden_size=4,
            moe_router_topk=1,
            moe_router_score_function="sigmoid",
            moe_router_num_groups=2,
            moe_router_group_topk=1,
            moe_router_enable_expert_bias=True,
        )
    )
    router.expert_bias.copy_(torch.tensor([-10.0, 0.0, 0.0, 10.0]))
    logits = torch.tensor([[[4.0, 3.0, 2.0, 1.0]], [[3.0, 2.0, 1.0, 0.0]]])

    _, dispatch_map = router.routing(logits)

    raw_scores = torch.sigmoid(logits.reshape(-1, 4))
    raw_scores = raw_scores / raw_scores.sum(dim=-1, keepdim=True)
    raw_top1 = torch.topk(raw_scores, 1, dim=-1).indices
    raw_map = torch.zeros_like(raw_scores).scatter_(1, raw_top1, 1)
    expected = _compute_token_aux_loss(router, logits)
    tracker.collect(router)
    logs = {}
    tracker.report(logs, lambda value: value.reshape(1))

    assert not torch.equal(dispatch_map, raw_map)
    torch.testing.assert_close(torch.tensor(logs["aux_loss"]), expected)


def test_router_aux_does_not_change_forward_routing():
    aux_router = TopKRouter(_router_config())
    plain_router = TopKRouter(
        _router_config(
            moe_aux_loss_coeff=0.0,
        )
    )
    weight = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 10
    aux_router.weight.data.copy_(weight)
    plain_router.weight.data.copy_(weight)
    hidden_states = torch.randn(2, 3, 3)

    aux_probs, aux_map, _ = aux_router(hidden_states)
    plain_probs, plain_map, _ = plain_router(hidden_states)

    torch.testing.assert_close(aux_probs, plain_probs)
    torch.testing.assert_close(aux_map, plain_map)


def test_aux_loss_autoscaler_adds_scaled_gradient():
    logits = torch.randn(5, 4, requires_grad=True)
    main_loss = logits.square().mean()
    router = TopKRouter(_router_config())
    aux_loss = _compute_token_aux_loss(router, logits) * 0.1
    expected_grad = torch.autograd.grad(
        main_loss + aux_loss * 0.25, logits, retain_graph=True
    )[0]

    MoEAuxLossAutoScaler.set_loss_scale(0.25)
    attached = MoEAuxLossAutoScaler.apply(main_loss, aux_loss)
    attached.backward()

    torch.testing.assert_close(attached, main_loss)
    torch.testing.assert_close(logits.grad, expected_grad)
    MoEAuxLossAutoScaler.set_loss_scale(1.0)


def test_token_aux_loss_ignores_padding_tokens():
    logits = torch.tensor(
        [
            [2.0, 0.0, -1.0, 0.5],
            [8.0, -4.0, -3.0, -2.0],
            [0.0, 1.0, 2.0, -0.5],
            [-5.0, 7.0, -6.0, -4.0],
        ],
        requires_grad=True,
    )
    padding_mask = torch.tensor([False, True, False, True])
    router = TopKRouter(_router_config(hidden_size=4))

    loss = _compute_token_aux_loss(router, logits, padding_mask)
    loss.backward()

    reference_logits = logits.detach()[~padding_mask].requires_grad_()
    reference_loss = _compute_token_aux_loss(router, reference_logits)
    reference_loss.backward()

    torch.testing.assert_close(loss, reference_loss)
    torch.testing.assert_close(logits.grad[~padding_mask], reference_logits.grad)
    torch.testing.assert_close(
        logits.grad[padding_mask], torch.zeros_like(logits.grad[padding_mask])
    )


def test_aux_loss_tracker_averages_layers(caplog):
    tracker = MetricTracker()
    model = torch.nn.ModuleList([TopKRouter(_router_config()) for _ in range(2)])
    model[0](torch.randn(2, 3, 3))
    model[1](torch.randn(2, 3, 3))

    layer_expected = [router._metrics["aux_loss"] for router in model]
    expected = sum(layer_expected) / len(model)
    with caplog.at_level(logging.INFO, logger="torchturbo"):
        tracker.collect(model)
    logs = {}
    tracker.report(logs, lambda value: value.reshape(1))

    torch.testing.assert_close(torch.tensor(logs["aux_loss"]), expected)
    torch.testing.assert_close(torch.tensor(logs["aux_loss_scaled"]), expected * 0.1)
    assert set(logs) == {"aux_loss", "aux_loss_scaled"}


def test_aux_loss_tracker_reports_mtp_module_path_in_debug(caplog):
    class MetricModule(torch.nn.Module):
        def pop_metrics(self):
            return {"aux_loss": torch.tensor(2.0)}

    model = torch.nn.Module()
    model.mtp = torch.nn.Module()
    model.mtp.layers = torch.nn.ModuleList([MetricModule()])
    tracker = MetricTracker()

    with caplog.at_level(logging.DEBUG, logger="torchturbo"):
        tracker.collect(model)
    logs = {}
    tracker.report(logs, lambda value: value.reshape(1))

    assert logs == {
        "aux_loss": 2.0,
        "aux_loss/mtp.layers.0": 2.0,
    }


def test_aux_loss_logging_averages_microbatches():
    tracker = MetricTracker()
    for _ in range(4):
        tracker.record(
            {
                "aux_loss": torch.tensor(2.0),
                "aux_loss_scaled": torch.tensor(0.002),
            }
        )

    logs = {}
    tracker.report(logs, lambda value: value.reshape(1))

    assert logs == {"aux_loss": 2.0, "aux_loss_scaled": 0.002}
    assert tracker.counts == {"aux_loss": 0, "aux_loss_scaled": 0}


def _run_sp_aux_loss_test(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        hidden_states = torch.tensor(
            [
                [1.0, 0.5, -0.25],
                [-0.5, 2.0, 0.75],
                [0.25, -1.0, 1.25],
                [1.5, 0.75, -0.5],
                [-1.25, 0.5, 2.0],
            ]
        )
        initial_weight = torch.tensor(
            [
                [0.2, -0.1, 0.3, 0.6],
                [-0.4, 0.5, 0.1, -0.2],
                [0.7, -0.3, 0.4, 0.1],
            ]
        )
        local_hidden_states = hidden_states.chunk(world_size)[rank]

        for score_function, topk in (
            ("softmax", 1),
            ("sigmoid", 2),
            ("sqrtsoftplus", 2),
        ):
            router = TopKRouter(
                _router_config(
                    hidden_size=3,
                    moe_router_topk=topk,
                    moe_router_score_function=score_function,
                ),
                sp_group=dist.group.WORLD,
            )
            weight = initial_weight.clone().requires_grad_()
            loss = _compute_token_aux_loss(router, local_hidden_states @ weight)
            loss.backward()
            dist.all_reduce(weight.grad, op=dist.ReduceOp.SUM)
            weight.grad.div_(world_size)

            reported_loss = loss.detach().clone()
            dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM)
            reported_loss.div_(world_size)

            reference_router = TopKRouter(
                _router_config(
                    hidden_size=3,
                    moe_router_topk=topk,
                    moe_router_score_function=score_function,
                )
            )
            reference_weight = initial_weight.clone().requires_grad_()
            reference_loss = _compute_token_aux_loss(
                reference_router, hidden_states @ reference_weight
            )
            reference_loss.backward()

            torch.testing.assert_close(reported_loss, reference_loss)
            torch.testing.assert_close(weight.grad, reference_weight.grad)

        padding_mask = torch.tensor([False, True, False, False, True])
        local_padding_mask = padding_mask.chunk(world_size)[rank]
        router = TopKRouter(
            _router_config(hidden_size=3),
            sp_group=dist.group.WORLD,
        )
        weight = initial_weight.clone().requires_grad_()
        loss = _compute_token_aux_loss(
            router,
            local_hidden_states @ weight,
            local_padding_mask,
        )
        loss.backward()
        dist.all_reduce(weight.grad, op=dist.ReduceOp.SUM)
        weight.grad.div_(world_size)

        reported_loss = loss.detach().clone()
        dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM)
        reported_loss.div_(world_size)

        reference_router = TopKRouter(_router_config(hidden_size=3))
        reference_weight = initial_weight.clone().requires_grad_()
        reference_loss = _compute_token_aux_loss(
            reference_router,
            hidden_states @ reference_weight,
            padding_mask,
        )
        reference_loss.backward()

        torch.testing.assert_close(reported_loss, reference_loss)
        torch.testing.assert_close(weight.grad, reference_weight.grad)

        router = TopKRouter(
            _router_config(hidden_size=3),
            sp_group=dist.group.WORLD,
        )
        router.weight.data.copy_(initial_weight.T)
        scores, _, _ = router(local_hidden_states.unsqueeze(1))
        (scores * 0.0).sum().backward()
        dist.all_reduce(router.weight.grad, op=dist.ReduceOp.SUM)
        router.weight.grad.div_(world_size)

        reference_router = TopKRouter(_router_config(hidden_size=3))
        reference_weight = initial_weight.T.clone().requires_grad_()
        reference_aux = _compute_token_aux_loss(
            reference_router, hidden_states @ reference_weight.T
        )
        (reference_aux * 0.1).backward()

        torch.testing.assert_close(router.weight.grad, reference_weight.grad)
    finally:
        dist.destroy_process_group()


def test_sp_aux_loss_matches_unsharded(tmp_path):
    mp.spawn(
        _run_sp_aux_loss_test,
        args=(2, str(tmp_path / "sp_aux_loss")),
        nprocs=2,
        join=True,
    )


def _run_sp_moe_runtime_inputs_test(
    rank: int, world_size: int, init_file: str
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from torchturbo.models.qwen3_5_moe.model import (
            TurboQwen3_5MoeForConditionalGeneration,
        )

        model = SimpleNamespace(
        )
        attention_mask = torch.tensor([[1, 0, 1, 1, 0, 0]])
        inputs = {"attention_mask": attention_mask}
        parallel_context = {
            "sp": {
                "sp_group": dist.group.WORLD,
                "sp_size": world_size,
                "sp_rank": rank,
            }
        }
        runtime_inputs = (
            TurboQwen3_5MoeForConditionalGeneration.get_moe_runtime_inputs(
                model, inputs, parallel_context
            )
        )
        expected = attention_mask.eq(0).chunk(world_size, dim=1)[rank]

        assert inputs["attention_mask"] is attention_mask
        torch.testing.assert_close(runtime_inputs.padding_mask, expected)
    finally:
        dist.destroy_process_group()


def test_qwen_moe_runtime_inputs_match_sp_shard(tmp_path):
    mp.spawn(
        _run_sp_moe_runtime_inputs_test,
        args=(2, str(tmp_path / "sp_moe_runtime_inputs")),
        nprocs=2,
        join=True,
    )


def test_qwen_moe_runtime_inputs_use_attention_mask_for_lf():
    from torchturbo.models.qwen3_5_moe.model import (
        TurboQwen3_5MoeForConditionalGeneration,
    )

    model = SimpleNamespace()
    attention_mask = torch.tensor([[1, 1, 2, 2, 0, 0]])
    inputs = {"attention_mask": attention_mask}
    parallel_context = {
        "sp": {"sp_group": None, "sp_size": 1, "sp_rank": 0}
    }

    runtime_inputs = (
        TurboQwen3_5MoeForConditionalGeneration.get_moe_runtime_inputs(
            model, inputs, parallel_context
        )
    )

    assert inputs["attention_mask"] is attention_mask
    torch.testing.assert_close(runtime_inputs.padding_mask, attention_mask.eq(0))


def _run_sp_group_isolation_test(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
        group_index = rank // 2
        group = groups[group_index]
        group_rank = dist.get_rank(group)
        group_logits = (
            torch.tensor(
                [
                    [2.0, 0.0, -1.0, 0.5],
                    [0.0, 1.0, 2.0, -0.5],
                    [1.0, 3.0, 0.0, 0.25],
                    [-1.0, 0.5, 2.5, 1.0],
                    [0.75, -0.25, 1.5, 2.0],
                ]
            )
            + group_index * torch.tensor([0.0, 1.5, -0.5, 0.75])
        )
        local_logits = group_logits.chunk(2)[group_rank].clone().requires_grad_()
        router = TopKRouter(
            _router_config(
                hidden_size=4,
                moe_router_score_function="softmax",
            ),
            sp_group=group,
        )
        loss = _compute_token_aux_loss(router, local_logits)
        loss.backward()

        reported_loss = loss.detach().clone()
        dist.all_reduce(reported_loss, op=dist.ReduceOp.SUM, group=group)
        reported_loss.div_(2)

        reference_router = TopKRouter(
            _router_config(
                hidden_size=4,
                moe_router_score_function="softmax",
            )
        )
        reference_logits = group_logits.clone().requires_grad_()
        reference_loss = _compute_token_aux_loss(reference_router, reference_logits)
        reference_loss.backward()
        reference_local_grad = reference_logits.grad.chunk(2)[group_rank]

        torch.testing.assert_close(reported_loss, reference_loss)
        torch.testing.assert_close(local_logits.grad / 2, reference_local_grad)
    finally:
        dist.destroy_process_group()


def test_sp_aux_loss_is_scoped_to_sp_group(tmp_path):
    mp.spawn(
        _run_sp_group_isolation_test,
        args=(4, str(tmp_path / "sp_aux_loss_groups")),
        nprocs=4,
        join=True,
    )
