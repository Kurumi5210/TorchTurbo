# LLaDA 2.1 SFT 训练说明（torchturbo V3 + NPU）

> 推理相关见同目录 `README_gen.md`。

LLaDA2.1 是**离散块扩散（Block Diffusion, BD）**模型，不是自回归。训练入口仍是 LlamaFactory + torchturbo 的 `run_lf.py`，BD/CAP 逻辑按 `model_type=llada2` 自动接入（`run_lf.py` 走 BD 数据/loss，`--tt_cap_lambda>0` 开 CAP）。

## 一、快速跑通

```bash
# 1. 装依赖（NPU）
bash scripts/runs/npu_deps/qwen3_5_moe_deps.sh

# 2. 训练（单机 16 卡 / sp2 ep8 / 8K / gsm8k_bd）
bash scripts/runs/llada2/lf_text_sft.sh
```

脚本：`scripts/runs/llada2/lf_text_sft.sh`，模型默认 `/mnt/sfs_turbo/models/LLaDA2.1-mini`，数据集默认 `gsm8k_bd`，也可使用 `science`。

## 二、关键训练参数

| 参数 | 值/说明 |
|---|---|
| `--stage sft` | SFT 阶段 |
| `--finetuning_type full` | 全参微调 |
| `--template default` | LLaDA2 chat 模板 |
| `--cutoff_len` | 单样本截断长度（= `seq_length`）；**BD beam 长度 = 2×cutoff_len**（noisy + clean 拼接） |
| `--flash_attn sdpa` | **目前 attention 用 PyTorch 原生 SDPA，尚未上 block/flash 高性能算子**（见第四节） |
| `--neat_packing` | 多样本拼进一个 beam，提升短样本算力利用率（gsm8k 填充率 ~98%） |
| `--bf16` | 计算精度 |
| `--tt_sp_size` / `--tt_ep_size` | 序列并行 / 专家并行（SP1EP1 纯 FSDP2 走精度对齐基线） |
| `--tt_bd_mask_low` / `--tt_bd_mask_high` | **BD 加噪比例 sigma 采样区间** `U[low,high]`，默认 0.3 / 0.8；每条样本独立采一个 sigma，按比例把 maskable token 替换成 `<|mask|>` |
| `--tt_bd_block_size` | **BD 块大小（block-diagonal attention 粒度）**，默认 32（dFactory/LLaDA2 默认值）；`cutoff_len` 会向上对齐到 `bd_block_size` 的整数倍，BD beam 内按此粒度切块双向可见、跨块不可见 |
| `--tt_cap_lambda` | **CAP 置信度损失权重**；`>0` 开启（推荐 2.0），默认 0 = 关闭 |

示例（脚本里的实际配置）：

```bash
torchrun $DISTRIBUTED_ARGS scripts/runs/run_lf.py \
    --tt_sp_size 2 --tt_ep_size 8 \
    --tt_bd_mask_low 0.3 --tt_bd_mask_high 0.8 \
    --stage sft --model_name_or_path $model_name_or_path \
    --flash_attn sdpa --dataset gsm8k_bd --template default \
    --finetuning_type full --cutoff_len 4096 --neat_packing \
    --lr_scheduler_type cosine --warmup_ratio 0.03 \
    --learning_rate 1e-5 --weight_decay 0.1 --max_grad_norm 1.0 \
    --max_steps 50 --bf16 --seed 42 \
    --output_dir $output_dir ...
```

## 三、训练特性的实际效果

| 特性 | 开关 | 结论 |
|---|---|---|
| **BD 加噪阈值** | `--tt_bd_mask_low/high` | sigma 区间越激进模型越难学；loss/grad_norm 与阈值强绑定 |
| **排除 padding 的 maskable** |（默认行为）只对 `[response]` 部分可 mask | 避免 padding 过多导致受迫预测 `<pad>`；单机首 loss `0.139→0.928`、首 grad_norm `20.8→9.93`，精度回到合理区间 |
| **neat_packing** | `--neat_packing` | 多样本平均 → 梯度方差小、更平滑；首 grad_norm `9.93→4.96` |
| **CAP** | `--tt_cap_lambda 2.0` | 辅助损失压熵；`L_conf` 从 0.33 降到 0.07；推理 TPF 提升（详见 `README_gen.md`） |

> maskable 只取 `[response]`、不含 `[response][padding]`，这点对精度影响显著——padding 多时若误把 padding 也设为 maskable，loss 会因被稀释而偏低。

## 四、Attention 算子现状（重要）

**当前 LLaDA2 训练的 attention 走 PyTorch 原生 SDPA（`--flash_attn sdpa`），尚未接入 block-attention / flash 类高性能算子。** 后果：

- **长序列（8K 以上）性能受限**，无法享受 block-attention 在长序列上的 tiling 收益。
- BD 的 block-diagonal attention mask（beam 内多样本间不可见）目前靠 SDPA 通用路径构造，灵活性够、但不是最优。

这与交付清单第 6 项「接入 block attention 算子 ⚠️（8K 以上性能明显提升）」状态一致——**block-attention 接入尚未完成**，是当前主要待补的算子缺口。短序列（≤16K）训练已验证可用，长序列/超长序列效果待 block 接入后压测。

## 五、SP/EP 并行支持

LLaDA2 head 配置 `Q=16, KV=4`（GQA）。Ulysses SP 在 head 数 < sp_size 时按需 repeat head（自动），已验证 SP1/2/4/8/16/32 精度对齐：

| ulysses_sp | repeat_q | repeat_kv | a2a 后 Q / KV | eff_gqa |
|---|---|---|---|---|
| 2 | 1 | 1 | 8 / 2 | 4 |
| 4 | 1 | 1 | 4 / 1 | 4 |
| 8 | 1 | 2（4→8） | 2 / 1 | 2 |
| 16 | 1 | 4（4→16） | 1 / 1 | 1 |
| 32 | 2（16→32） | 8（4→32） | 1 / 1 | 1 |

- **SP>8** 必须 `repeat_kv`，**SP>16** 必须 `repeat_q`（框架自动处理）。
- 单机（gbs16，SP1/2/4/8/16 × EP1/8）与双机（gbs32，SP1/32 × EP8）均已端到端精度对齐，证明 EP 与 repeat_q/repeat_kv 正确性。

## 六、精度对齐基线

- **单步精度对齐**：torchturbo+LlamaFactory（Stack B）vs dFactory+VeOmni（Stack A），共享 prepared batch（禁用两边的随机掩码 RNG 差异），hook/msprobe dump tensor 逐项比对 → 已对齐。
- **端到端 loss 对齐**：固定 `lr_scheduler_type=constant`、mask、dataloader shuffle 三个变量，50 步 loss vs dFactory golden，趋势与绝对值对齐。

## 七、长跑实验参考

> 以下均：关 CAP、开 neat_packing、gsm8k、100 steps。

**mini（2 机 32 卡）：**

| 序列 | SP | EP | gbs | 秒/step |
|---|---|---|---|---|
| 8K | 1 | 8 | 32 | 6.9 |
| 16K | 2 | 8 | 32 | 26.8 |
| 32K | 4 | 8 | 32 | 216.8 |

**flash 100B（4 机 32 卡）：**

| 序列 | SP | EP | gbs | 秒/step |
|---|---|---|---|---|
| 16K | 1 | 8 | 64 | 53.3 |
| 16K | 4 | 8 | 64 | 71.6 |
| 32K | 8 | 8 | 64 | 464.5 |

> 32K 长序列 step 时长陡增（mini 从 8K 的 6.9s → 32K 的 216.8s），正是第四节「缺高性能 attention 算子」的直接体现——长序列下 SDPA 通用路径开销大。block-attention 接入后预期有显著改善。

## 八、TODO

**block-attention 接入**，待接入完备后，将补充更长序列的长跑实验。
