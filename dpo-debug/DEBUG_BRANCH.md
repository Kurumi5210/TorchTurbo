本分支 `debug/dpo-history-think` 从 `docs/dpo-debug-scripts@26d78e57897d426dc005a6853fb602ea02d70c8b` 独立分出。验证历史 thinking 输入对精度的影响：在 RLHF 的 last_round 模式下保留历史 thinking，并允许所有 assistant 轮次补空 think。监督边界仍使用原来的最后 user。

对照基线为上述提交的 NPU 配置。另一实验分支与本分支是并列关系，应分别和同一基线比较。

根因、DPO/RPO 影响、公开 Swift 提交记录和验证边界见 [DPO_PRECISION_ANALYSIS.md](DPO_PRECISION_ANALYSIS.md)。

| 实验 | chosen / rejected 输入 token | chosen / rejected 有效监督 token |
| --- | ---: | ---: |
| 原 NPU 基线 | 77119 / 77103 | 2295 / 2279 |
| 本分支 history_think | 77127 / 77111 | 2295 / 2279 |

真实日志中两个历史 assistant 各增加 4 个空 think token，新增位置的 labels 为 -100；其他有效监督 token 的内容和顺序与原 NPU 完全相同。包含非空历史 thinking 的构造样本也验证了保留行为。本分支强制启用此历史保留策略，只适用于本次对照实验。

改动位于 [Qwen3.5 Swift 模板包装](../src/torchturbo/models/qwen3_5/swift_template.py)。仍使用 `--template qwen3_5`、`--loss_scale last_round+ignore_empty_think`、`--add_non_thinking_prefix true` 和 Swift **4.4.2**。只在 `mode=rlhf` 且 base strategy 为 `last_round` 时生效。SFT、推理和其他 loss strategy 的原行为已经过 CPU 对照；Qwen3.5 dense / MoE 共享此模板包装。

**本地 CPU 验证。**

[verify_dpo_debug_cpu.py](verify_dpo_debug_cpu.py) 执行实际的 Swift 模板与本分支注册函数。为避免导入整个加速引擎，仅将模板文件中的包日志导入替换为标准日志适配；未加载权重或执行 forward。它逐项核对两端日志的完整数组，并检查多次工具调用、非空历史 thinking、序列化及其他训练模式。mask 分支还检查 rejected_response 的字符串/列表形式和显式 loss 覆盖。

在本分支目录中执行，tokenizer 目录应包含与日志样本匹配的 processor/tokenizer 元数据：

```bash
python dpo-debug/verify_dpo_debug_cpu.py \
  --tokenizer /path/to/tokenizer \
  --output /tmp/dpo-history_think-verification.json
```

如果使用源码版 Swift，增加 `--swift-source /path/to/ms-swift-4.4.2`。本次本地验证使用 CPU PyTorch 2.7.0 / Transformers 5.9.0 / Swift 4.4.2；正式训练继续使用原 NPU 依赖环境，CPU 验证不覆盖设备前反向。

**在原 NPU 提交包中运行。**

原 [NPU 脚本](npu_torchturbo_dpo.sh) 会解压固定的 `torchturbo-c9c1ad2.tar.gz`。本分支脚本在解压后，将当前分支的单个模板文件覆盖到运行目录，校验实验标记并记录 SHA256，避免切了 Git 分支却继续训练旧代码。

在已有提交包的副本中放入下面两个文件，再沿用原来的任务提交流程和环境变量：

```bash
dpo_package=/path/to/package-copy
cp src/torchturbo/models/qwen3_5/swift_template.py \
  "$dpo_package/dpo_debug_swift_template.py"
cp dpo-debug/npu_torchturbo_dpo.sh \
  "$dpo_package/run_qwen38_27b_dpo.sh"
```

两台节点必须拿到同一份包，并为本次实验指定独立的 OUTPUT_DIR。原代码归档、数据 manifest 和归档校验继续使用原包；覆盖操作仅作用于解压后的本次运行目录。脚本会拒绝错误实验的模板文件及非 4.4.2 的 Swift。

启动时应出现：

```text
DPO debug overlay: {"experiment": "history_think", ...}
DPO debug template registered: history_think (qwen3_5, mode=rlhf, last_round)
```

rank 0 的输出目录额外保存 `dpo-debug-overlay.json` 和实际使用的 `dpo-debug-swift-template.py`，`submission-config.json` 的 `dpo_debug` 字段记录相同信息。原 `torchturbo-commit.txt` 仍表示基础归档；完整实验代码由该归档和模板 SHA256 一起确定。

**对比方式。**

输入变长可能改变超长样本过滤与 packing 分组。全量训练前核对保留的原始样本 ID、每包有序样本列表和每步 batch；严格隔离上下文影响时，基线和本分支应使用同一份包清单。不能仅凭相同 seed 认定分组相同。

先确认日志中的 input_ids/labels 符合上表，再按原训练时间轴完成训练，用相同推理引擎、推理模板、解码参数和评测集评测对应 checkpoint。这里不切换推理模板。CPU 编码检查通过并不代表业务精度已经恢复。

本分支不同时修改另一项实验行为，也不调整 RPO、beta、学习率、优化器或算子实现。
