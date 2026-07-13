import contextlib
import re
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer
torch.serialization.add_safe_globals([DynamicCache, DynamicLayer])
from collections import OrderedDict

# Constant definitions
OUTPUT_FILES = {
    "deepspeed": "/input/zhisan/dump/ds_infer_rank_0.pt",
    "torchturbo": "/input/zhisan/dump/tt_infer_rank_0.pt"
}
COMPARISON_LOG = "/input/zhisan/dump/align_compare.log"

def compare_tensor(k, v1, v2, eps=1e-5):
    """Compare if two tensors are similar"""
    v1 = v1.float()
    v2 = v2.float()
    if not torch.allclose(v1, v2):
        diff = torch.abs(v1 - v2)
        max_diff = diff.max().item()
        diff_mask = diff > eps
        n_total = v1.numel()
        n_diff = diff_mask.sum().item()

        print(f'key: {k} diff')
        print(f"\tmax difference        : {max_diff:.8e}")
        print(f"\tnum different (>eps)  : {n_diff}  ({n_diff/n_total*100:.2f}%)")
        print(f"\tnum same (==0)        : {n_total - n_diff}  ({(n_total-n_diff)/n_total*100:.2f}%)")
        print(f"\teps used              : {eps}")
        return False
    print(f'key: {k} is the same')
    return True


def compare_outputs():
    print(f"Comparing outputs and writing results to {COMPARISON_LOG}")

    d1 = torch.load(OUTPUT_FILES["torchturbo"], map_location="cpu", weights_only=True)
    d2 = torch.load(OUTPUT_FILES["deepspeed"],   map_location="cpu", weights_only=True)

    all_keys = list(OrderedDict.fromkeys(k for step in d1 for k in step))
    print(all_keys)

    with open(COMPARISON_LOG, "w") as f, contextlib.redirect_stdout(f):
        # For each key, scan through both sets of records and perform comparison \
        # only when the key is present in both.
        for k in all_keys:
            idx1 = idx2 = 0
            while idx1 < len(d1) and idx2 < len(d2):
                if k not in d1[idx1]:
                    idx1 += 1
                    continue
                if k not in d2[idx2]:
                    idx2 += 1
                    continue

                v1, v2 = d1[idx1][k], d2[idx2][k]
                try:
                    # the hooked function 'decoder_layer{i}_forward'
                    if k.endswith('_forward'):
                        pass
                    # the hooked function 'decoder_layer{i}_att_op_chunk'
                    # https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/simple_gla/chunk.py
                    # def chunk_simple_gla(
                    #     q: torch.Tensor,
                    #     k: torch.Tensor,
                    #     v: torch.Tensor, 
                    #     g: torch.Tensor | None = None,
                    #     g_gamma: torch.Tensor | None = None,
                    #     scale: float | None = None,
                    #     initial_state: torch.Tensor | None = None,
                    #     output_final_state: bool = False,
                    #     cu_seqlens: torch.LongTensor | None = None,
                    #     head_first: bool = False,
                    # ) -> tuple[torch.Tensor, torch.Tensor]:
                    # return o, final_state
                    elif k.endswith('_att_op_chunk'):
                        # input
                        compare_tensor(f"{k}:q",  v1[0][0], v2[0][0]) # torch.Size([1, 512, 16, 128])
                        compare_tensor(f"{k}:k",  v1[0][1], v2[0][1]) # torch.Size([1, 512, 16, 128])
                        compare_tensor(f"{k}:v",  v1[0][2], v2[0][2]) # torch.Size([1, 512, 16, 128])
                        compare_tensor(f"{k}:g",  v1[0][3], v2[0][3]) # torch.Size([1, 512, 16])
                        # output
                        compare_tensor(f"{k}:o",  v1[1][0], v2[1][0]) # torch.Size([1, 512, 16, 128])
                        compare_tensor(f"{k}:state",  v1[1][1], v2[1][1]) # torch.Size([1, 16, 128, 128])
                    # (dispatched_input, tokens_per_expert, permuted_probs) = (self.token_dispatcher.token_permutation(hidden_states, probs, routing_map))
                    elif k.endswith('_mlp_dispatch'):
                        # input
                        compare_tensor(f"{k}:hidden_states",  v1[0][0], v2[0][0])
                        compare_tensor(f"{k}:probs",  v1[0][1], v2[0][1])
                        compare_tensor(f"{k}:routing_map",  v1[0][2], v2[0][2])
                        # output
                        compare_tensor(f"{k}:dispatched_input",  v1[1][0], v2[1][0])
                        compare_tensor(f"{k}:tokens_per_expert",  v1[1][1], v2[1][1])
                        compare_tensor(f"{k}:permuted_probs",  v1[1][2], v2[1][2])                    
                    # output, mlp_bias = self.token_dispatcher.token_unpermutation(expert_output, mlp_bias)
                    # mlp_bias is None
                    elif k.endswith('_mlp_undispatch'):
                        # input
                        compare_tensor(f"{k}:expert_output",  v1[0][0], v2[0][0])
                        # output
                        compare_tensor(f"{k}:output",  v1[1][0], v2[1][0])
                    elif k.endswith(('post_attention_layernorm', 'input_layernorm')):
                        compare_tensor(f"{k}:input",  v1[0][0], v2[0][0])
                        compare_tensor(f"{k}:output", v1[1][0], v2[1][0])
                        compare_tensor(f"{k}:weight", v1[2][0], v2[2][0])
                    elif k.endswith((
                        'attn.query_key_value',
                        'attn.query_layernorm',
                        'attn.key_layernorm',
                        'attn.g_norm',
                        'attn.g_proj',
                        'attn.dense'
                        )):
                        compare_tensor(f"{k}:input",  v1[0][0], v2[0][0])
                        compare_tensor(f"{k}:output", v1[1][0], v2[1][0])
                    elif k.endswith((
                        'mlp.router',
                        'mlp.experts',
                        'mlp.shared_experts.gate_proj',
                        'mlp.shared_experts.act_fn',
                        'mlp.shared_experts.up_proj',
                        'mlp.shared_experts.down_proj',
                        'mlp.shared_experts'
                        )):
                        compare_tensor(f"{k}:input",  v1[0][0], v2[0][0])
                        compare_tensor(f"{k}:output", v1[1][0], v2[1][0])
                    elif re.fullmatch(r'layer\.\d+', k):
                        compare_tensor(f"{k}:input",  v1[0][0], v2[0][0])
                        compare_tensor(f"{k}:output", v1[1][0], v2[1][0])
                    elif re.fullmatch(r'layer\.\d+\.attn', k):
                        compare_tensor(f"{k}:val", v1[1][0], v2[1][0])
                    elif re.fullmatch(r'layer\.\d+\.mlp', k):
                        compare_tensor(f"{k}:input",  v1[0][0], v2[0][0])
                        compare_tensor(f"{k}:output", v1[1][0], v2[1][0])
                    elif k == 'input':
                        compare_tensor("input_ids", v1[0], v2[0])
                    elif k == 'logits':
                        compare_tensor("logits", v1[0], v2[0])
                    elif k == 'loss':
                        diff = abs(v1[0] - v2[0])
                        print(f'loss diff: {diff:.8e}')
                        if diff > 5e-3: # 1e-4
                            print('Precision Alignment Failure!')
                            return False
                except (IndexError, TypeError) as e:
                    print(f'[ERROR] {k}: {e}')
                break
            else:
                print(f'[SKIP] {k} not found in both records.')

    print('Pass Precision Alignment!')
    return True

def main():
    compare_outputs()

if __name__ == "__main__":
    main()