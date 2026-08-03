# Qwen3.5 Dense
## 支持特性

| 特性 | 项目 |
| --- | --- |
| **SFT** | 基于 transformers 4.57.3 接入 qwen3.5 modeling |
| | 基于 H20/A3，对齐开源 deepspeed 0.19.2 + transformers 5.7.0 精度 |
| | 支持 ulysses SP，最大支持 SP=32/48 |
| | 对齐 turbo infer v.s. vllm-ascend infer |
| | 对齐 turbo v.s. megatron (via mbridge+mindspeed) infer，涵盖 packing 场景 |

## TODO LIST

- [ ] 高 grad_norm 问题定位 @cjs02503376 @执伞
- [ ] 接入 Ascend C 算子后验证性能/精度 @执伞 @cjs02503376
- [ ] 27B dense 模型 scaling 到 256K，更新长跑实验 @cjs02503376

## 运行脚本

| 任务 | 脚本 |
| --- | --- |
| TEXT SFT | `scripts/runs/lf_qwen3_5_dense_text_sft.sh` |
| TEXT CPT | `scripts/runs/lf_qwen3_5_dense_text_cpt.sh` |
| MLLM SFT | `scripts/runs/lf_qwen3_5_dense_mllm_sft.sh` |
| MLLM CPT | `scripts/runs/lf_qwen3_5_dense_mllm_cpt.sh` |

---

## 精度对齐报告

基于 Qwen3.5-4B:与 deepspeed 对齐用单机 8 卡,与 vllm_infer / megatron 对齐用单机单卡。

**结论:训练侧 `loss/grad` 精度达标,可交付;前向 logp 差异来自 GDN 算子跨框架实现差异。**

### 与 deepspeed 精度对齐

实验:Qwen3.5-4B,单机,8K 序列长度,science 数据集。loss / grad_norm 对齐曲线:

![loss](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/222456468/1784793830977-c8754d01-5036-4ec3-b582-b1fe9dfc37fd.png)

![grad_norm](https://intranetproxy.alipay.com/skylark/lark/0/2026/png/222456468/1784794420346-e1aaf533-a497-4668-ace0-71d9fde7ec96.png)

### 与 vllm_infer 精度对齐

vllm 是推理引擎,仅做前向图比对(无 loss/grad):

```bash
# turbo 训练侧 vs vllm 推理侧(前向图)
bash scripts/rl/run_turbo_vs_vllm.sh
```

| 指标 | turbo vs vllm (seq=122) |
| --- | --- |
| 前向 logp mean abs diff | 3.08e-2 |
| **前向 logp max abs diff** | **1.36e-1 ✅** |
| 达 ≤1e-3 位置占比 | 24% |

### 与 megatron 精度对齐

```bash
# turbo vs megatron,前向图 + loss/grad
bash scripts/rl/run_turbo_vs_megatron.sh           # 非 packed
bash scripts/rl/run_turbo_vs_megatron.sh --packed  # 3 序列 packed
```

| 阶段 | 产物 | 说明 |
| --- | --- | --- |
| [1/4] gen_input | `input_spec.pt` | 共享输入 |
| [2/4] megatron | `{graph.pt,loss_grad.json,logp.npy}` | MCore via mbridge + mindspeed |
| [3/4] turbo | 同上 | TorchTurboEngine |
| [4/4] compare | `report.json` | 前向逐 token logp + 模块图比对 |
| [cmp] loss_grad_compare | `loss_grad_report.json` | loss/grad 差异 |

| 指标 | turbo vs megatron 非 packed (seq=122) | turbo vs megatron packed (seq=31) |
| --- | --- | --- |
| 前向 logp mean abs diff | 3.58e-2 | 3.71e-2 |
| 前向 logp max abs diff | 1.29e-1 | 2.40e-1 |
| 达 ≤1e-3 位置占比 | 14% | 20% |
| loss abs diff | 4.7e-4 | 4.9e-3 |
| **loss rel diff** | **1.5e-4 ✅** | **1.4e-3 ✅** |
| grad abs diff | 1.96e-2 | 0.93 |
| **grad rel diff** | **3.9e-4 ✅** | **9.1e-3 ✅** |
| grad ratio (t/m) | 0.9996 | 0.991 |
| 有梯度参数 (turbo/megatron) | 426 / 306 | 426 / 306 |
| 有效 token 数 | 121 | 28 |
