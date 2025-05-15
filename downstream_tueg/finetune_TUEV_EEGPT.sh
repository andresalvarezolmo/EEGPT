#!/usr/bin/env bash
set -x  # print the commands

# export MASTER_PORT=${MASTER_PORT:-12320}  # You should set the same master_port in all the nodes
export MASTER_PORT=$((12000 + $RANDOM % 20000))

# official train/test splits. valid numbers: 1, 2, 3
SPLIT=${SPLIT:-1}

N_NODES=${N_NODES:-1}  # Number of nodes
GPUS_PER_NODE=${GPUS_PER_NODE:-2}  # Number of GPUs in each node
SRUN_ARGS=${SRUN_ARGS:-""}  # Other slurm task args
PY_ARGS=${@:2}  # Other training args

# fine tuned 

nohup env CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
        python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=0 --master_addr="localhost" \
        run_class_finetuning_EEGPT_change_tuev.py \
        --output_dir ./checkpoints_TUEV/finetune_tuev_eegpt/ \
        --log_dir ./log/finetune_tuev_eegpt \
        --model EEGPT \
        --finetune /home/aa2650/playground/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt \
        --weight_decay 0.05 \
        --batch_size 300 \
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 5 \
        --epochs 10 \
        --layer_decay 0.65 \
        --drop_path 0.2 \
        --dist_eval \
        --save_ckpt_freq 0 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset TUEV\
        --seed 1234 \
  > tuev-fine-tune.log 2>&1 &


# Zero shot

# nohup env CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} \
#         --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=0 --master_addr="localhost" \
#         run_class_finetuning_EEGPT_change_tuev.py \
#         --output_dir ./checkpoints_TUEV/finetune_tuev_eegpt/ \
#         --log_dir ./log/finetune_tuev_eegpt \
#         --model EEGPT \
#         --eval \
#         --finetune /home/aa2650/playground/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt \
#         --batch_size 400\
#         --drop_path 0.2 \
#         --dist_eval \
#         --save_ckpt_freq 5 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --dataset TUEV\
#         --seed 32354987
# > tuev-zero-shot.log 2>&1 &

# Original command

# nohup env CUDA_VISIBLE_DEVICES=0,2 OMP_NUM_THREADS=1 \
#         python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} \
#         --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=0 --master_addr="localhost" \
#         run_class_finetuning_EEGPT_change_tuev.py \
#         --output_dir ./checkpoints_TUEV/finetune_tuev_eegpt/ \
#         --log_dir ./log/finetune_tuev_eegpt \
#         --model EEGPT \
#         --finetune /home/aa2650/playground/EEGPT/checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt \
#         --weight_decay 0.05 \
#         --batch_size 400\
#         --lr 5e-4 \
#         --update_freq 1 \
#         --warmup_epochs 5 \
#         --epochs 20 \
#         --layer_decay 0.65 \
#         --drop_path 0.2 \
#         --dist_eval \
#         --save_ckpt_freq 5 \
#         --disable_rel_pos_bias \
#         --abs_pos_emb \
#         --dataset TUEV\
#         --enable_deepspeed \
#         --seed 0 \
#   > tuev-fine-tune.log 2>&1 &
