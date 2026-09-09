# Offline Tokenizer 离线数据处理指南

> **核心**：在 CPU 节点**离线处理一次**，产物写入共享存储；所有训练节点**秒级直接加载**，彻底解决海量数据在线 Tokenize 触发的集群启动超时。

---

## 核心机制差异（速查）

| 维度 | LLaMA-Factory (LF) | ms-swift |
| :--- | :--- | :--- |
| **产物内容** | **全量 Token 级张量**（`input_ids`、`labels` 等） | **原始文本/多模态字段 + 预计算 `lengths`** |
| **编码时机** | 离线一次性完成，训练时原样返回 | 训练取数据时懒加载（Lazy Encode） |
| **`cutoff_len`** | ⚠️ **必须在离线定死**（训练传了会被忽略） | ⚠️ **以训练脚本为准**（离线仅做统计估算） |
| **`packing`** | ⚠️ **在离线执行**（加 `--neat_packing` 拼满序列） | ⚠️ **在训练执行**（离线严禁传 `--packing`） |
| **多模态支持** | 离线需校验媒体占位符环境变量 | 完整保留媒体字段，Lazy 实时读图，完全支持 |

---

## 快速使用

### 1. LLaMA-Factory 路径

- **离线处理**：改好参数直接运行脚本
  ```bash
  bash scripts/offline_tokenizer/lf_tokenize_run.sh
  ```
  *(注：脚本已配置 `--preprocessing_num_workers 32` 多进程及媒体占位符环境变量)*
- **训练加载**：训练脚本中把 `--dataset` 替换为 `--tokenized_path /path/to/science_lf`。

### 2. ms-swift 路径

- **离线处理**：改好参数直接运行脚本
  ```bash
  bash scripts/offline_tokenizer/swift_tokenize_run.sh
  ```
  *(注：脚本已配置 `--dataset_num_proc 32` 加速，离线切勿传 `--packing`)*
- **训练加载**：训练脚本中**删掉 `--dataset`**，替换为 `--cached_dataset /path/to/science_swift/train`（训练命令照常带 `--packing`）。

### 3. 精度抽样验证（可选）

```bash
# LF 验证
python scripts/offline_tokenizer/verify_tokenized.py lf train_sft.yaml --tokenized_path /path/to/science_lf --num_samples 500

# Swift 验证
python scripts/offline_tokenizer/verify_tokenized.py swift --model_type turbo_qwen3_5_moe --model /path/to/model --template qwen3_5 --max_length 4096 --cached_dataset /path/to/science_swift/train --num_samples 500
```

---

## 实测对齐与效果（Benchmark）

在真实昇腾 NPU 环境下，分别针对 **LLaMA-Factory (纯文本 4K)** 与 **ms-swift (多模态 4K)** 实测对比：

### 1. LLaMA-Factory 实测（Qwen3.5-4B，Science SFT 纯文本 4K）
| 模式 | Step 1 Loss | Step 2 Loss | Step 3 Loss | Step 4 Loss | Step 5 Loss | 结论 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **JSONL 在线** | `0.4763` | `0.4740` | `0.4490` | `0.4728` | `0.4713` | 基准 |
| **Tokenized 离线** | `0.4763` | `0.4733` | `0.4491` | `0.4728` | `0.4709` | **位级完全对齐**（误差 $\le 0.0007$） |

### 2. ms-swift 实测（Qwen3.5-4B，500 条 MLLM 多模态 4K）
| 模式 | Step 1 Loss | Grad Norm | MFU / HFU | Step 耗时 (s/it) | 结论 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **JSONL 在线** | `2.02233171` | `577.7180` | `3.34%` / `4.45%` | `16.02s` | 基准 |
| **Cached 离线** | `2.02233171` | `577.6985` | `5.01%` / `6.68%` | `7.16s` | **Loss 8 位完全相同，吞吐提升 2.24x** |
