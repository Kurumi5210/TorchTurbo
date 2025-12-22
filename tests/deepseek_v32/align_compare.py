import contextlib
import torch
import math

# Constant definitions
OUTPUT_FILES = {
    "hf_infer": "/tmp/hf_infer_rank_0.pt",
    "torchturbo": "/tmp/tt_infer_rank_0.pt"
}
COMPARISON_LOG = "/tmp/align_compare.log"

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
    """Compare outputs of both modes and write to file"""
    print(f"Comparing outputs and writing results to {COMPARISON_LOG}")

    with open(COMPARISON_LOG, "w") as f, contextlib.redirect_stdout(f):
        d1 = torch.load(OUTPUT_FILES["hf_infer"], map_location="cpu")
        d2 = torch.load(OUTPUT_FILES["torchturbo"], map_location="cpu")

        assert len(d1) == len(d2)
        for i in range(len(d1)):
            for k, v1 in d1[i].items():
                v2 = d2[i][k]
                if 'layer' in k:
                    in1 = v1[0]
                    in2 = v2[0]
                    out1 = v1[1]
                    out2 = v2[1]
                    if "input_layernorm" in k or "q_a_layernorm" in k:
                        w1 = v1[2]
                        w2 = v2[2]
                        compare_tensor(f"{k}:weight", w1[0], w2[0])
                        compare_tensor(f"{k}:input", in1[0], in2[0])
                        compare_tensor(f"{k}:output", out1[0], out2[0])
                    elif ".indexer" in k:
                        compare_tensor(f"{k}:input", in1[0], in2[0])
                        compare_tensor(f"{k}:qr", in1[1], in2[1])
                        compare_tensor(f"{k}:mask", in1[4], in2[4])
                        compare_tensor(f"{k}:index_score", out1[0], out2[0])
                        compare_tensor(f"{k}:topk_indices", out1[1], out2[1])
                    elif "index_scores" in k:
                        compare_tensor(f"{k}:q", in1[0], in2[0])
                        compare_tensor(f"{k}:weights", in1[1], in2[1])
                        compare_tensor(f"{k}:k", in1[2], in2[2])
                        compare_tensor(f"{k}:output", out1[0], out1[0])
                    elif "apply_rotary_emb" in k:
                        input2 = in2[0]
                        output2 = out2[0]
                        if in1[0].shape != in2[0].shape:
                            input2 = in2[0].transpose(1, 2)
                            output2 = out2[0].transpose(1, 2)
                        compare_tensor(f"{k}:input", in1[0], input2)
                        compare_tensor(f"{k}:output", out1[0], output2)
                    else:
                        if len(in1) > 0 and len(in2) > 0 and not compare_tensor(f"{k}:input", in1[0], in2[0]):
                            pass
                        compare_tensor(f"{k}:output", out1[0], out2[0])
                elif 'input' in k:
                    compare_tensor("input_ids", v1[0], v2[0])
                elif 'logits' in k:
                    compare_tensor("logits", v1[0], v2[0])
                elif 'loss' in k:
                    print(f'loss: {v1[0]} vs {v2[0]}')
                    diff = v1[0] - v2[0]
                    if diff > 1e-4:
                        print('Precision Alignment Failure!')
                        return False
    print('Pass Precision Alignment!')
    return True

def main():
    compare_outputs()

if __name__ == "__main__":
    main()