#!/usr/bin/env bash

# 8 * 95GiB
export NCCL_DEBUG=INFO
export NCCL_COMM_TIMEOUT=3600
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=900

export NCCL_IB_DISABLE=1


NODE_NAME=`echo $ILOGTAIL_PODNAME | awk -F 'ptjob-' '{print $2}'`
NODE_NAME=${NODE_NAME:-master-0}

output_dir=/mnt4/ckpt/yintaoye.yty/mxc-qwen38-27B-261010-128kFP8-rl-dpo

mkdir -p $output_dir
log_file=$output_dir/log-$NODE_NAME.txt

export TORCH_NCCL_AVOID_RECORD_STREAMS="1"
export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'


bridge_file=/opt/conda/lib/python3.10/site-packages/mcore_bridge/bridge/gpt_bridge.py
bridge_old="tensor = None if tensor is None else tensor.to('cuda')"
bridge_new="tensor = None if tensor is None else tensor.to('cuda').contiguous()"

if [[ ! -f "$bridge_file" ]]; then
    echo "ERROR: file not found: $bridge_file" >&2
    exit 1
elif grep -Fq "$bridge_new" "$bridge_file"; then
    echo "Patch already applied: $(hostname)"
elif grep -Fq "$bridge_old" "$bridge_file"; then
    cp -a "$bridge_file" "${bridge_file}.bak.$(date +%Y%m%d-%H%M%S)"

    sed -i "s|$bridge_old|$bridge_new|" "$bridge_file"

    grep -Fq "$bridge_new" "$bridge_file" || {
        echo "ERROR: patch failed: $(hostname)" >&2
        exit 1
    }

    python -m py_compile "$bridge_file" || {
        echo "ERROR: syntax check failed: $(hostname)" >&2
        exit 1
    }

    echo "Patch applied: $(hostname)"
else
    echo "ERROR: unexpected _all_gather_tp implementation: $(hostname)" >&2
    exit 1
fi

mkdir -p /root/train_data
chmod 700 /root/train_data
rpm -Uhv "https://artifacts.antgroup-inc.cn/artifact/repositories/ant_7_x86_64_current/alipay-linkc-zeta/alipay-linkc-zeta-0.17.1-436968808.x86_64.rpm"
zeta co https://$FINLLMOPS_ZETA_USERNAME:$FINLLMOPS_ZETA_TOKEN@zeta.alipay.com/$FINLLMOPS_DATA_SELECTION_ZETA_GROUP/$FINLLMOPS_DATA_SELECTION_ZETA_REPO --commit $FINLLMOPS_DATA_SELECTION_COMMIT_ID /root/train_data
# train_data_files=$(find "/root/train_data/SFT/AntWealth-WealthMxc-AllIntent-All" -type f \( -name "*html_instruct.json" \) -exec realpath {} \;)
# cp -r /mnt4/data/yintaoye.yty/train_data/SFT/AntWealth-WealthMxc-AllIntent-All/20251018_yintaoye.yty_map_think_sft_moe_new_html_instruct.json /root/train_data/SFT/AntWealth-WealthMxc-AllIntent-All/.
# train_data_files=$(find "/root/train_data/SFT/AntWealth-WealthMxc-AllIntent-All" -type f \( -name "20251023_yintaoye.yty_*" \) -exec realpath {} \;)
# train_data_files_str=$(echo $train_data_files | tr '\n' ' ')
dpo_data_files=$(find "/root/train_data/DPO" -type f \( -name "20260407_yintaoye.yty_map_think_dpo_moe_html_instruct_merge.json" \) -exec realpath {} \;)
dpo_data_files_str=$(echo $dpo_data_files | tr '\n' ' ')


export NCCL_BUFFSIZE=1048576
export NCCL_MAX_NCHANNELS=4

export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export NVTE_UNFUSED_ATTN=0
export NVTE_DEBUG=1
export NVTE_DEBUG_LEVEL=2


unset NVTE_FLASH_ATTN
unset NVTE_FUSED_ATTN
unset NVTE_UNFUSED_ATTN

export SWIFT_DIR=/mnt4/code/yintaoye.yty/ms-swift-260914
export PYTHONPATH="${SWIFT_DIR}:${PYTHONPATH:-}"

cd "$SWIFT_DIR"

export model=/mnt4/ckpt/xiaxiao.ld/35b_rl/GSPO-mxc-38-0904-xiaxiao-nothink-mtp-cp-gdpo-epoch-rmv20/global_step_176/actor/huggingface

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=12 \
NPROC_PER_NODE=8 \
NNODES=${WORLD_SIZE:-1} \
NODE_RANK=${RANK:-0} \
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1} \
MASTER_PORT=20091 \
megatron rlhf \
    --rlhf_type dpo \
    --ref_model $model \
    --model $model \
    --save_safetensors true \
    --truncation_strategy 'delete' \
    --model_type qwen3_5 \
    --dataset /mnt4/code/xinhaochen.cxh/0训练/mf_qwen3.5_397b_a17b_instruct-Main-Selected/DPO/AntWealth-WealthMxc-AllIntent-All/business_user_flag_answer.json \
    --agent_template qwen3_5 \
    --template qwen3_8 \
    --beta 0.1 \
    --rpo_alpha 0.1 \
    --clip_grad 1.0 \
    --label_smoothing 0.05 \
    --global_batch_size 24 \
    --loss_type sigmoid \
    --add_non_thinking_prefix true \
    --linear_decoupled_in_proj true \
    --load_from_cache_file false \
    --loss_scale last_round+ignore_empty_think \
    --fp8_recipe blockwise \
    --fp8_format e4m3 \
    --fp8_param_gather true \
    --split_dataset_ratio 0.01 \
    --tuner_type full \
    --tensor_model_parallel_size 4 \
    --pipeline_model_parallel_size 4 \
    --context_parallel_size 4 \
    --micro_batch_size 1 \
    --use_distributed_optimizer true \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --num_train_epochs 1 \
    --packing true \
    --finetune true \
    --freeze_llm false \
    --freeze_vit true \
    --save_strategy epoch \
    --freeze_aligner true \
    --nccl_comm_warmup true \
    --use_precision_aware_optimizer true \
    --mtp_loss_scaling_factor 0.0 \
    --lr_warmup_fraction 0.05 \
    --lr 1e-6 \
    --min_lr 1e-7 \
    --output_dir $output_dir \
    --eval_steps 20000 \
    --save_steps 20000 \
    --max_length 128000 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --no_save_optim true \
    --no_save_rng true \
    --mtp_num_layers 1 \
    --save_missing_weights true \
    --sequence_parallel true \
    --logging_steps 1 \
    --sequence_packing_scheduler dp_balanced \
    --padding_free true \
    --ddp_timeout 1800000000 \
    --attention_backend flash  2>&1 | tee $log_file
 cd fs_turbo/fanyaxin.fyx/a3_qwen38_27b_dpo_20261010
