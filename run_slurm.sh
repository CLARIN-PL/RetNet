#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=6
#SBATCH --mem=60G
#SBATCH --job-name=retnet-pl
#SBATCH --output=slurm.out

eval "$(conda shell.bash hook)"
conda activate retnet

time torchrun  --nproc_per_node=3 train.py \
    --model_size 300m \
    --output_dir checkpoints \
    --do_train --do_eval \
    --prediction_loss_only 
    --remove_unused_columns False \
    --learning_rate 6e-4 \
    --weight_decay 0.01 \
    --max_steps 20000 \
    --logging_steps 100 \                                                                                                                                                                                         --max_steps 1000 \                                                                                                                                                                                            --logging_steps 12 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4
