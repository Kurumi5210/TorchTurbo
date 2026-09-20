#!/usr/bin/env python3
"""Offline dataset tokenization for the LLaMA-Factory path (CPU-only, no engine, no GPU).

Runs LlamaFactory's full data pipeline (template + tokenizer + processors +
multiprocess ``dataset.map``) WITHOUT building the model/engine, and saves the
tokenized dataset to ``--tokenized_path``. Training then loads it in seconds
instead of re-tokenizing millions of samples at startup::

    # Step 1 (once, on any CPU node — can run offline / in data-prep jobs):
    python scripts/offline_tokenizer/lf_tokenize.py train_sft.yaml \\
        --tokenized_path /shared_fs/tokenized/my_sft_v1

    # Step 2 (training, unchanged yaml + the same --tokenized_path):
    torchrun ... scripts/runs/run_lf.py train_sft.yaml \\
        --tokenized_path /shared_fs/tokenized/my_sft_v1 --tt_sp_size 8 ...

torchturbo integration
----------------------
The script activates the llamafactory_v0 adapter FIRST, so torchturbo model
templates (``lf_template.register()`` per model) exist during tokenization.
For multimodal PT (stage=pt with image inputs), the adapter's
``PretrainProcessorPatch`` — normally applied after the engine is built — is
applied here explicitly so offline artifacts match the online patched
processor. ``--tt_*`` args are accepted and ignored (they only affect the
engine, which tokenization never builds).
"""

from __future__ import annotations

import os
import shutil
import sys

from _common import ensure_repo_src, pop_flag, pop_value, strip_distributed_env

ensure_repo_src()

_ENGINE_CONFIG_HINT = """
[lf_tokenize] 传入参数错误。
正确姿势二选一:
  A. 传完整 LF 训练 yaml(含 model_name_or_path/dataset/template/cutoff_len 等)
  B. 把 LF 参数原样搬成命令行 flags，例如:
       lf_tokenize.py --stage sft --do_train --model_name_or_path <模型路径> \\
           --dataset_dir <dir> --dataset <名> --template <模板> \\
           --cutoff_len <长度> --tokenized_path <产物目录>
(--tt_config / --tt_* 与 tokenize 无关,会被自动剥离)"""


def _looks_like_engine_config(raw: object) -> bool:
    """torchturbo engine yamls (scripts/runs/**/lf_*.yaml) have a single root key."""
    return isinstance(raw, dict) and set(raw) == {"engine_config"}


def main() -> None:
    argv = sys.argv[1:]

    # This script is single-process CPU-only by design. Platform launchers
    # (kmaker etc.) inject torchrun-style env vars into every container; if
    # left set, TrainingArguments.__post_init__ -> accelerate PartialState ->
    # init_process_group(env://) hangs in _create_c10d_store waiting for peer
    # ranks that will never exist. Strip them before LF parses args.
    removed = strip_distributed_env()
    if removed:
        print(f"[lf_tokenize] single-process CPU job: stripped launcher env vars: {', '.join(removed)}")

    # Our own flags (not LF's).
    argv, overwrite = pop_flag(argv, "--overwrite")

    # Strip --tt_* (engine-only; no engine is built here) but accept them so the
    # exact training command line can be reused.
    import torchturbo
    from torchturbo.engine.args import parse_torchturbo_args

    tt_args, argv = parse_torchturbo_args(argv)

    # tokenized_path is required — it is the output dir of this tool.
    argv, tokenized_path = pop_value(argv, "--tokenized_path")
    if not tokenized_path:
        raise SystemExit(
            "lf_tokenize.py requires --tokenized_path <output_dir> "
            "(where the tokenized dataset is saved)."
        )
    tokenized_path = os.path.abspath(os.path.expanduser(tokenized_path))

    # Activate the LF adapter (registers turbo model templates).
    torchturbo.activate("llamafactory_v0")

    # No engine here: build a standalone PluginManager and consume --tt_plugin into the now-active LF adapter.
    if tt_args.plugins:
        from torchturbo.adapters.llamafactory_v0.plugins import install_plugins
        from torchturbo.engine.plugins import PluginManager

        install_plugins(PluginManager(list(tt_args.plugins)))

    if os.path.exists(tokenized_path) and os.listdir(tokenized_path):
        if not overwrite:
            raise SystemExit(
                f"{tokenized_path} already exists and is not empty. "
                "Pass --overwrite to re-tokenize (the old directory is deleted first)."
            )
        shutil.rmtree(tokenized_path)

    from llamafactory.hparams import get_train_args, read_args

    raw_args = read_args(argv)

    # Friendly guard: the repo's scripts/runs/**/lf_*.yaml files are torchturbo
    # ENGINE configs (root key `engine_config`, consumed via --tt_config), NOT
    # LF training configs. Feeding one here yields LF's raw
    # "Please provide `model_name_or_path`" — catch that and redirect.
    try:
        model_args, data_args, training_args, finetuning_args, _gen_args = get_train_args(raw_args)
    except (ValueError, KeyError) as e:
        hint = ""
        if _looks_like_engine_config(raw_args) or "model_name_or_path" in str(e):
            hint = "\n" + _ENGINE_CONFIG_HINT
        raise SystemExit(f"[lf_tokenize] 参数解析失败: {e}{hint}") from e

    stage = finetuning_args.stage
    if stage not in {"sft", "pt", "dpo"}:
        raise SystemExit(f"stage {stage!r} is not taken over by torchturbo; tokenize natively instead.")

    # Feed the artifact path into data_args so get_dataset saves there.
    data_args.tokenized_path = tokenized_path

    # Multimodal PT parity: online, PretrainProcessorPatch is applied after the
    # engine is built; offline there is no engine, so apply it explicitly. Its
    # patch() never touches the engine argument.
    if stage == "pt":
        from torchturbo.adapters.llamafactory_v0.patches import PretrainProcessorPatch

        PretrainProcessorPatch().patch(engine=None)

    from llamafactory.data import get_dataset, get_template_and_fix_tokenizer
    from llamafactory.model import load_tokenizer

    tokenizer_module = load_tokenizer(model_args)
    template = get_template_and_fix_tokenizer(tokenizer_module["tokenizer"], data_args)

    print(f"[lf_tokenize] stage={stage} dataset={data_args.dataset} -> {tokenized_path}")
    dataset_module = get_dataset(
        template=template,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        stage=stage,
        tokenizer=tokenizer_module["tokenizer"],
        processor=tokenizer_module.get("processor"),
    )
    train_ds = dataset_module.get("train_dataset")
    n = len(train_ds) if train_ds is not None else 0
    print(f"[lf_tokenize] done: {n} train rows saved to {tokenized_path}")
    print(f"[lf_tokenize] train with: run_lf.py <same yaml/flags> --tokenized_path {tokenized_path}")


if __name__ == "__main__":
    main()
