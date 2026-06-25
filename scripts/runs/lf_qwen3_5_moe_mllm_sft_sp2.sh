#!/usr/bin/env bash
# qwen3_moe SFT training with torchturbo
# Usage:
#   bash scripts/runs/lf_qwen3_5_moe_sft.sh

set -o pipefail

set -ex

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export ASCEND_GLOBAL_LOG_LEVEL=3
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_ASYNC_ERROR_HANDLING=0
export HCCL_IF_BASE_PORT=29972
export GLOO_SOCKET_IFNAME=eth0
export GLOO_LOG_LEVEL=DEBUG
export MULTI_STREAM_MEMORY_REUSE=2 # 多流

model_name_or_path=/sfs_turbo/zhisan/models/Qwen3___5-35B-A3B/
output_dir=/sfs_turbo/zhisan/0604/lf_logs_qwen35/run_qwen3_5_moe_debug_35b_sft_mllm_sp2ep8/ckpt

seq_length=$((1024 * 4))
dataset=llava_1k_zh
packing_mode=packing_sep
finetuning_type=full

export WANDB_DISABLED=True
export ALLOW_EXTRA_ARGS=True
export PYTHONPATH=/opt/MindSpeed:$PYTHONPATH
export TRANSFORMERS_VERBOSITY=error

export NNODES=${NNODES:-1}
export NPUS_PER_NODE=${NPUS_PER_NODE:-16}
export NODE_RANK=${RANK:-0}
export WORLD_SIZE=$(($NPUS_PER_NODE * $NNODES))
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-23556}

export ASCEND_PROCESS_LOG_PATH=$output_dir/log

mkdir -p $output_dir
if [ -d $output_dir ]; then
  echo 'Exist: ', $output_dir
else
  echo 'Not exist: ', $output_dir
fi

log_file=$output_dir/log-$NODE_RANK.txt
echo $output_dir
cp -r $0 $output_dir
cur_time=$(date "+%Y%m%d-%H%M%S")
echo $cur_time >> $log_file
echo $output_dir >> $log_file

DISTRIBUTED_ARGS="
    --nproc_per_node $NPUS_PER_NODE \
    --nnodes $NNODES \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank $NODE_RANK
"
RUN_LF="$(dirname "$0")/run_lf.py"
DATASET_DIR="$(dirname "$0")/../../dataset/"

export PYTHONPATH=/sfs_turbo/zhisan/0604/LlamaFactory/src:$PYTHONPATH
export PYTHONPATH=/sfs_turbo/zhisan/0604/torchturbo/src:$PYTHONPATH
export PYTHONPATH=/sfs_turbo/zhisan/0604/transformers_4_57_3_pure/src:$PYTHONPATH

torchrun $DISTRIBUTED_ARGS $RUN_LF \
    --tt_sp_size 2 \
    --tt_ep_size 8 \
    --num_train_epochs 1 \
    --report_to tensorboard \
    --logging_dir /home/admin/logs/tfevent \
    --gradient_checkpointing \
    --dataset_dir ${DATASET_DIR} \
    --max_steps 30 \
    --trust_remote_code \
    --disable_shuffling \
    --stage sft \
    --model_name_or_path $model_name_or_path \
    --do_train \
    --flash_attn fa2 \
    --dataset $dataset \
    --preprocessing_num_workers 16 \
    --cutoff_len ${seq_length} \
    --neat_packing \
    --template qwen3_5 \
    --finetuning_type ${finetuning_type} \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --eval_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --overwrite_cache \
    --save_strategy "steps" \
    --eval_steps 200000000 \
    --save_steps 200000000 \
    --warmup_ratio 0.0 \
    --learning_rate 1.25e-5 \
    --save_total_limit 1 \
    --plot_loss \
    --seed 42 \
    --bf16 \
    --tt_freeze_vision_tower \
    --ddp_timeout 1800000000 2>&1 | tee -a $log_file

    # --tt_freeze_vision_tower \
    # --tt_freeze_multi_modal_projector \
    # --tt_freeze_language_model \

# sp2 4k [Full training]
# {'loss': 1.3501, 'grad_norm': 32.84737014770508, 'learning_rate': 1.25e-05, 'epoch': 0.17}
# {'loss': 1.0642, 'grad_norm': 15.853341102600098, 'learning_rate': 1.2465761846051709e-05, 'epoch': 0.33}
# {'loss': 1.0784, 'grad_norm': 5.326085567474365, 'learning_rate': 1.2363422504586286e-05, 'epoch': 0.5}
# {'loss': 1.0366, 'grad_norm': 3.382215738296509, 'learning_rate': 1.219410322684471e-05, 'epoch': 0.67}
# {'loss': 1.0752, 'grad_norm': 3.4580605030059814, 'learning_rate': 1.1959659110266256e-05, 'epoch': 0.83}
# {'loss': 0.9819, 'grad_norm': 2.5178868770599365, 'learning_rate': 1.1662658773652743e-05, 'epoch': 1.0}
# {'loss': 0.8333, 'grad_norm': 3.340057849884033, 'learning_rate': 1.1306356214843423e-05, 'epoch': 1.17}
# {'loss': 0.7656, 'grad_norm': 3.0199737548828125, 'learning_rate': 1.0894655159233715e-05, 'epoch': 1.33}
# {'loss': 0.6758, 'grad_norm': 2.812812566757202, 'learning_rate': 1.0432066289742864e-05, 'epoch': 1.5}
# {'loss': 0.6481, 'grad_norm': 3.601930856704712, 'learning_rate': 9.923657826827957e-06, 'epoch': 1.67}
# {'loss': 0.7343, 'grad_norm': 9.601107597351074, 'learning_rate': 9.375000000000001e-06, 'epoch': 1.83}
# {'loss': 0.6166, 'grad_norm': 4.606403827667236, 'learning_rate': 8.792104019223753e-06, 'epoch': 2.0}
# {'loss': 0.5273, 'grad_norm': 1.9955930709838867, 'learning_rate': 8.181356214843423e-06, 'epoch': 2.17}
# {'loss': 0.4858, 'grad_norm': 3.908517837524414, 'learning_rate': 7.5494480676109954e-06, 'epoch': 2.33}
# {'loss': 0.3981, 'grad_norm': 3.8920586109161377, 'learning_rate': 6.903302895422835e-06, 'epoch': 2.5}
# {'loss': 0.3912, 'grad_norm': 3.524088144302368, 'learning_rate': 6.25e-06, 'epoch': 2.67}
# {'loss': 0.5508, 'grad_norm': 3.1083548069000244, 'learning_rate': 5.596697104577167e-06, 'epoch': 2.83}
# {'loss': 0.3361, 'grad_norm': 5.796478748321533, 'learning_rate': 4.950551932389005e-06, 'epoch': 3.0}
# {'loss': 0.2524, 'grad_norm': 4.0508341789245605, 'learning_rate': 4.3186437851565795e-06, 'epoch': 3.17}
# {'loss': 0.3188, 'grad_norm': 5.864254951477051, 'learning_rate': 3.7078959807762497e-06, 'epoch': 3.33}
# {'loss': 0.2659, 'grad_norm': 5.305734634399414, 'learning_rate': 3.1250000000000014e-06, 'epoch': 3.5}
# {'loss': 0.2524, 'grad_norm': 3.3514158725738525, 'learning_rate': 2.5763421731720436e-06, 'epoch': 3.67}
# {'loss': 0.4, 'grad_norm': 5.960437297821045, 'learning_rate': 2.0679337102571383e-06, 'epoch': 3.83}
# {'loss': 0.2288, 'grad_norm': 3.1630730628967285, 'learning_rate': 1.6053448407662854e-06, 'epoch': 4.0}
# {'loss': 0.149, 'grad_norm': 2.9621315002441406, 'learning_rate': 1.1936437851565791e-06, 'epoch': 4.17}
# {'loss': 0.2292, 'grad_norm': 3.2937395572662354, 'learning_rate': 8.373412263472582e-07, 'epoch': 4.33}
# {'loss': 0.1925, 'grad_norm': 3.4835057258605957, 'learning_rate': 5.403408897337439e-07, 'epoch': 4.5}
# {'loss': 0.1817, 'grad_norm': 2.766528367996216, 'learning_rate': 3.0589677315529047e-07, 'epoch': 4.67}
# {'loss': 0.3439, 'grad_norm': 2.717902660369873, 'learning_rate': 1.3657749541371446e-07, 'epoch': 4.83}
# {'loss': 0.1927, 'grad_norm': 2.731218099594116, 'learning_rate': 3.423815394829194e-08, 'epoch': 5.0}

# sp2 4k [Freeze vision encoder]
# {'loss': 1.3501, 'grad_norm': 26.905658721923828, 'learning_rate': 1.25e-05, 'epoch': 0.17}
# {'loss': 1.0649, 'grad_norm': 2.7463181018829346, 'learning_rate': 1.2465761846051709e-05, 'epoch': 0.33}
# {'loss': 1.0872, 'grad_norm': 3.291201591491699, 'learning_rate': 1.2363422504586286e-05, 'epoch': 0.5}
# {'loss': 1.043, 'grad_norm': 1.6768263578414917, 'learning_rate': 1.219410322684471e-05, 'epoch': 0.67}
# {'loss': 1.0723, 'grad_norm': 1.8708635568618774, 'learning_rate': 1.1959659110266256e-05, 'epoch': 0.83}
# {'loss': 0.9893, 'grad_norm': 1.3991374969482422, 'learning_rate': 1.1662658773652743e-05, 'epoch': 1.0}
# {'loss': 0.8594, 'grad_norm': 1.4073065519332886, 'learning_rate': 1.1306356214843423e-05, 'epoch': 1.17}
# {'loss': 0.658, 'grad_norm': 2.1496877670288086, 'learning_rate': 1.0894655159233715e-05, 'epoch': 1.33}
# {'loss': 0.7175, 'grad_norm': 1.7620593309402466, 'learning_rate': 1.0432066289742864e-05, 'epoch': 1.5}
# {'loss': 0.6465, 'grad_norm': 1.5859441757202148, 'learning_rate': 9.923657826827957e-06, 'epoch': 1.67}
# {'loss': 0.7293, 'grad_norm': 2.9706978797912598, 'learning_rate': 9.375000000000001e-06, 'epoch': 1.83}
# {'loss': 0.6176, 'grad_norm': 3.068765163421631, 'learning_rate': 8.792104019223753e-06, 'epoch': 2.0}
# {'loss': 0.5267, 'grad_norm': 1.6911381483078003, 'learning_rate': 8.181356214843423e-06, 'epoch': 2.17}
# {'loss': 0.4155, 'grad_norm': 2.1181600093841553, 'learning_rate': 7.5494480676109954e-06, 'epoch': 2.33}
# {'loss': 0.4467, 'grad_norm': 2.481220245361328, 'learning_rate': 6.903302895422835e-06, 'epoch': 2.5}
# {'loss': 0.3658, 'grad_norm': 3.2286550998687744, 'learning_rate': 6.25e-06, 'epoch': 2.67}
# {'loss': 0.5029, 'grad_norm': 2.817906379699707, 'learning_rate': 5.596697104577167e-06, 'epoch': 2.83}
# {'loss': 0.358, 'grad_norm': 1.9538999795913696, 'learning_rate': 4.950551932389005e-06, 'epoch': 3.0}
# {'loss': 0.2732, 'grad_norm': 3.0012454986572266, 'learning_rate': 4.3186437851565795e-06, 'epoch': 3.17}
# {'loss': 0.2503, 'grad_norm': 4.623661041259766, 'learning_rate': 3.7078959807762497e-06, 'epoch': 3.33}
# {'loss': 0.3009, 'grad_norm': 4.430187702178955, 'learning_rate': 3.1250000000000014e-06, 'epoch': 3.5}
# {'loss': 0.2545, 'grad_norm': 3.178412437438965, 'learning_rate': 2.5763421731720436e-06, 'epoch': 3.67}
# {'loss': 0.3635, 'grad_norm': 1.467491626739502, 'learning_rate': 2.0679337102571383e-06, 'epoch': 3.83}
# {'loss': 0.2193, 'grad_norm': 2.029318332672119, 'learning_rate': 1.6053448407662854e-06, 'epoch': 4.0}
# {'loss': 0.1738, 'grad_norm': 1.886759638786316, 'learning_rate': 1.1936437851565791e-06, 'epoch': 4.17}
# {'loss': 0.1797, 'grad_norm': 1.274606704711914, 'learning_rate': 8.373412263472582e-07, 'epoch': 4.33}
# {'loss': 0.2326, 'grad_norm': 1.5925649404525757, 'learning_rate': 5.403408897337439e-07, 'epoch': 4.5}
# {'loss': 0.1978, 'grad_norm': 1.592167854309082, 'learning_rate': 3.0589677315529047e-07, 'epoch': 4.67}
# {'loss': 0.289, 'grad_norm': 2.2618703842163086, 'learning_rate': 1.3657749541371446e-07, 'epoch': 4.83}
# {'loss': 0.1827, 'grad_norm': 2.1502349376678467, 'learning_rate': 3.423815394829194e-08, 'epoch': 5.0}
