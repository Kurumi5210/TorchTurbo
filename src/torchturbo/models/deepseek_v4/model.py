import math

import torch
from transformers import AutoModelForCausalLM

from torchturbo.models.modules.attention.ulysses import all_gather_sp
from torchturbo.models.modules.moe.moe_layer import MoELayer
from torchturbo.models.modules.moe.token_dispatcher import MoEAlltoAllTokenDispatcher
from torchturbo.ops import npu_fused_ops
from torchturbo.utils.device import is_npu_available

from ..registry import register_model
from ..weight_converter import BaseWeightConverter, WeightConverterMixin
from .configuration_deepseek_v4 import DeepseekV4Config
from .modeling_deepseek_v4 import DeepseekV4ForCausalLM


class DeepseekV4WeightConverter(BaseWeightConverter):
    def get_expert_gate_name(self):
        return "w1"

    def get_expert_up_name(self):
        return "w3"

    def get_expert_down_name(self):
        return "w2"

    def get_hf_expert_mlp_name(self):
        return "ffn"

    def get_hf_gate_bias_name(self):
        return "gate.bias"

    def get_hf_gate_hash_name(self):
        return "gate.tid2eid"

    def get_router_hash_name(self):
        return "router.tid2eid"

    def get_special_param_names(self):
        return [
            "hc_head_base",
            "hc_head_fn",
            "hc_head_scale",
            "hc_attn_base",
            "hc_attn_fn",
            "hc_attn_scale",
            "hc_ffn_base",
            "hc_ffn_fn",
            "hc_ffn_scale",
            "attn.attn_sink",
            "compressor.ape",
        ]

    def get_default_hf_dtype(self, param_name):
        if self.get_hf_gate_hash_name() in param_name:
            return torch.int64
        if self.get_hf_gate_bias_name() in param_name:
            return torch.float32
        for name in self.get_special_param_names():
            if name in param_name:
                return torch.float32
        return self.dtype

    def is_persist_buffer(self, name):
        if self.get_router_bias_name() in name:
            return True
        if self.get_router_hash_name() in name:
            return True
        return False

    def convert_persist_buffer(self, name):
        if self.get_router_hash_name() in name:
            return name.replace(self.get_router_hash_name(), self.get_hf_gate_hash_name())
        return super().convert_persist_buffer(name)


@register_model(
    "deepseek_v4", config_cls=DeepseekV4Config, auto_class=AutoModelForCausalLM
)
class TurboDeepseekV4ForCausalLM(WeightConverterMixin, DeepseekV4ForCausalLM):
    config_class = DeepseekV4Config
    base_model_prefix = ""

    def __init__(self, config: DeepseekV4Config):
        self._pre_init_patches(config)
        super().__init__(config)
        self._post_init_patches()

    @classmethod
    def get_parallel_plan(cls, **kwargs):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan(**kwargs)

    def get_weight_converter(self, mesh):
        return DeepseekV4WeightConverter(self.config, self.get_moe_config(), mesh)

    @staticmethod
    def _pre_init_patches(config):
        from . import modeling_deepseek_v4 as deepseek_v4_model

        origin_moe_forward = getattr(
            MoELayer, "_torchturbo_deepseek_v4_origin_forward", MoELayer.forward
        )
        MoELayer._torchturbo_deepseek_v4_origin_forward = origin_moe_forward

        def deepseek_moe_forward(self, hidden_states: torch.Tensor, input_ids=None):
            output, _ = origin_moe_forward(self, hidden_states, input_ids)
            return output

        MoELayer.forward = deepseek_moe_forward

        if is_npu_available:
            deepseek_v4_model.DeepseekV4RMSNorm.forward = (
                npu_fused_ops.fused_rms_norm_forward_npu
            )
            MoEAlltoAllTokenDispatcher.token_permutation = (
                npu_fused_ops.fused_token_permutation_npu
            )
            MoEAlltoAllTokenDispatcher.token_unpermutation = (
                npu_fused_ops.fused_token_unpermutation_npu
            )

        config._attn_implementation = "eager"

    def _post_init_patches(self):
        self._use_flash_attention_2 = True
        self.config._attn_implementation = "flash_attention_2"

    def post_weight_load(self):
        """Rebuild runtime buffers cleared by FSDP2 meta-device loading."""
        from . import modeling_deepseek_v4 as dsv4

        for layer in self.layers:
            attn = layer.attn
            if attn.compress_ratio > 1:
                original_seq_len = attn.config.original_max_position_embeddings
                rope_theta = attn.config.compress_rope_theta
            else:
                original_seq_len = 0
                rope_theta = attn.config.rope_theta

            freqs_cis = attn._precompute_freqs_cis(
                attn.rope_head_dim,
                attn.config.max_position_embeddings,
                original_seq_len,
                rope_theta,
                attn.config.rope_factor,
                attn.config.beta_fast,
                attn.config.beta_slow,
            ).to(attn.attn_sink.device)
            attn.register_buffer("freqs_cis", freqs_cis, persistent=False)

            if attn.compress_ratio == 4:
                hadamard_dim = 2 ** math.ceil(math.log2(attn.config.index_head_dim))
                hadamard_mat = dsv4.generate_hadamard_matrix(hadamard_dim).to(
                    attn.attn_sink.device
                )
                attn.register_buffer("hadamard_mat", hadamard_mat, persistent=False)
            else:
                attn.hadamard_mat = None

    def setup_sp(self, sp_info):
        sp_group = sp_info["sp_group"]
        sp_size = sp_info["sp_size"]
        sp_rank = sp_info["sp_rank"]
        assert sp_size >= 1, "must be"

        self.patch_deepseek_v4_attention_for_sp(sp_group, sp_size, sp_rank)

    @classmethod
    def patch_deepseek_v4_attention_for_sp(cls, sp_group, sp_size, sp_rank):
        from . import modeling_deepseek_v4 as dsv4

        def _save_origin(module_cls):
            origin_attr = "_torchturbo_deepseek_v4_origin_forward"
            if getattr(module_cls, origin_attr, None) is None:
                setattr(module_cls, origin_attr, module_cls.forward)
            return getattr(module_cls, origin_attr)

        patched_classes = [
            dsv4.DeepseekV4Compressor,
            dsv4.DeepseekV4Indexer,
            dsv4.GetWindowTopkIdxs,
            dsv4.GetCompressTopkIdxs,
            dsv4.LiCompute,
            dsv4.PreAttentionOps,
            dsv4.PostAttentionOps,
        ]
        origins = {module_cls: _save_origin(module_cls) for module_cls in patched_classes}

        if sp_size <= 1:
            for module_cls, origin_forward in origins.items():
                module_cls.forward = origin_forward
            return

        def get_sp_local_freqs_cis(local_seq_len, freqs_cis):
            start = sp_rank * local_seq_len
            end = (sp_rank + 1) * local_seq_len
            return freqs_cis[start:end]

        def compressor_forward(self, x: torch.Tensor, freqs_cis: torch.Tensor):
            _, local_seqlen, _ = x.size()
            ratio, overlap = self.compress_ratio, self.overlap
            dtype = x.dtype

            if local_seqlen * sp_size < ratio:
                return None

            x = x.float()
            kv = self.wkv(x)
            score = self.wgate(x)

            seqlen = local_seqlen
            kv = all_gather_sp(kv, dim=1, sp_group=sp_group, sp_size=sp_size)
            score = all_gather_sp(score, dim=1, sp_group=sp_group, sp_size=sp_size)
            seqlen = seqlen * sp_size
            assert seqlen == kv.size(1), "must be"
            freqs_cis = freqs_cis[:seqlen]

            remainder = seqlen % ratio
            cutoff = seqlen - remainder
            freqs_cis = freqs_cis[:cutoff:ratio]

            if remainder > 0:
                kv = kv[:, :cutoff]
                score = score[:, :cutoff]

            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape

            if overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))

            kv = (kv * score.softmax(dim=2)).sum(dim=2)
            kv = self.norm(kv.to(dtype))

            kv_rot = dsv4.apply_rotary_emb(kv[..., -self.rope_head_dim :], freqs_cis)
            kv = torch.cat([kv[..., : -self.rope_head_dim], kv_rot], dim=-1)
            return kv

        def indexer_forward(
            self,
            x: torch.Tensor,
            qr: torch.Tensor,
            freqs_cis: torch.Tensor,
            hadamard_mat: torch.Tensor,
            offset: int,
        ):
            bsz, seqlen, _ = x.size()
            rd = self.rope_head_dim
            local_freq_cis = get_sp_local_freqs_cis(seqlen, freqs_cis)

            q = self.wq_b(qr)
            q = q.view(bsz, seqlen, self.n_heads, self.head_dim)
            q = q.clone()
            q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
            q_rope = dsv4.apply_rotary_emb(q_rope, local_freq_cis)
            q = torch.cat([q_nope, q_rope], dim=-1)
            q = dsv4.rotate_activation(q, hadamard_mat)

            k = self.compressor(x, freqs_cis)
            if k is None:
                return None, None, None
            k = dsv4.rotate_activation(k, hadamard_mat)

            weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

            return q, k, weights

        def window_topk_forward(self, window_size: int, bsz: int, seqlen: int, device):
            base = torch.arange(
                seqlen * sp_rank, seqlen * (sp_rank + 1), device=device
            ).unsqueeze(1)
            global_seqlen = seqlen * sp_size
            window_topk = (base - window_size + 1).clamp(0) + torch.arange(
                min(global_seqlen, window_size), device=device
            )
            window_topk = torch.where(window_topk > base, -1, window_topk)
            return window_topk.unsqueeze(0).expand(bsz, -1, -1)

        def compress_topk_forward(self, x: torch.Tensor, offset: int):
            bsz, seqlen = x.shape[0], x.shape[1]
            matrix = torch.arange(
                seqlen * sp_size // self.ratio, device=x.device
            ).repeat(seqlen, 1)
            mask = matrix >= torch.arange(
                1 + seqlen * sp_rank,
                1 + seqlen * (sp_rank + 1),
                device=x.device,
            ).unsqueeze(1) // self.ratio
            compress_topk = torch.where(mask, -1, matrix + offset)
            return compress_topk.unsqueeze(0).expand(bsz, -1, -1)

        def li_compute_forward(
            self,
            q_indexer: torch.Tensor,
            k_indexer: torch.Tensor,
            weights: torch.Tensor,
            seqlen: int,
            offset: int,
        ):
            end_pos = seqlen * sp_size
            index_score = torch.einsum("bshd,btd->bsht", q_indexer, k_indexer)
            index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

            device = index_score.device
            base = torch.arange(seqlen, device=device).unsqueeze(1) + seqlen * sp_rank
            mask = (
                torch.arange(seqlen * sp_size // self.ratio, device=device).unsqueeze(0)
                >= (base + 1) // self.ratio
            )
            index_score = index_score + torch.where(
                mask, torch.finfo(q_indexer.dtype).min, 0
            )

            index_score, topk_idxs = index_score.topk(
                min(self.index_topk, end_pos // self.ratio), dim=-1
            )

            mask = topk_idxs >= (base + 1) // self.ratio
            compress_topk_idxs = torch.where(mask, -1, topk_idxs + offset)
            return compress_topk_idxs, index_score

        def pre_attention_forward(
            self,
            x: torch.Tensor,
            freqs_cis: torch.Tensor,
            hadamard_mat: torch.Tensor,
            wq_a,
            q_norm,
            wq_b,
            wkv,
            kv_norm,
            compressor,
            indexer,
        ):
            rd = self.rope_head_dim
            local_seq_len = x.size(1)
            local_freq_cis = get_sp_local_freqs_cis(local_seq_len, freqs_cis)

            qr = q_norm(wq_a(x))
            q = wq_b(qr).unflatten(-1, (self.n_heads, self.head_dim))
            q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
            q_nope, q_rope = torch.split(q, [self.head_dim - rd, rd], dim=-1)
            q_rope = dsv4.apply_rotary_emb(q_rope, local_freq_cis)
            q = torch.cat([q_nope, q_rope], dim=-1)

            kv = wkv(x)
            kv = kv_norm(kv)
            kv_nope, kv_rope = torch.split(kv, [self.head_dim - rd, rd], dim=-1)
            kv_rope = dsv4.apply_rotary_emb(kv_rope, local_freq_cis)
            kv = torch.cat([kv_nope, kv_rope], dim=-1)
            kv = all_gather_sp(kv, dim=1, sp_group=sp_group, sp_size=sp_size)

            offset = kv.size(1)

            q_indexer = k_indexer = weights = None
            if self.compress_ratio > 1 and indexer is not None:
                q_indexer, k_indexer, weights = indexer(
                    x.detach(), qr.detach(), freqs_cis, hadamard_mat, offset
                )

            kv_compress = None
            if self.compress_ratio > 1 and compressor is not None:
                kv_compress = compressor(x, freqs_cis)

            return q, kv, kv_compress, q_indexer, k_indexer, weights, offset

        def post_attention_forward(
            self,
            o: torch.Tensor,
            freqs_cis: torch.Tensor,
            bsz: int,
            seqlen: int,
            n_local_groups: int,
            wo_a,
            wo_b,
        ):
            freqs_cis = get_sp_local_freqs_cis(o.size(1), freqs_cis)
            rd = self.rope_head_dim
            o_nope, o_rope = torch.split(o, [self.head_dim - rd, rd], dim=-1)
            o_rope = dsv4.apply_rotary_emb(o_rope, freqs_cis, inverse=True)
            o = torch.cat([o_nope, o_rope], dim=-1)

            o = o.view(bsz, seqlen, n_local_groups, -1)
            wo_a_weight = wo_a.weight.view(n_local_groups, self.o_lora_rank, -1)
            o = torch.einsum("bsgd,grd->bsgr", o, wo_a_weight)
            return wo_b(o.reshape(bsz, seqlen, -1))

        dsv4.DeepseekV4Compressor.forward = compressor_forward
        dsv4.DeepseekV4Indexer.forward = indexer_forward
        dsv4.GetWindowTopkIdxs.forward = window_topk_forward
        dsv4.GetCompressTopkIdxs.forward = compress_topk_forward
        dsv4.LiCompute.forward = li_compute_forward
        dsv4.PreAttentionOps.forward = pre_attention_forward
        dsv4.PostAttentionOps.forward = post_attention_forward
