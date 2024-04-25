#!/bin/bash
#SBATCH --job-name=train_owt
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=100G
#SBATCH --time=1:30:00
#SBATCH --output=train_owt_%j.out
#SBATCH --error=train_owt_%j.err
#SBATCH --gpus=1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate transformer_lm

python3 train.py \
	--name "owt" \
	--vocab_size 32000 \
	--context_length 512 \
	--d_model 1024 \
	--num_layers 8 \
	--num_heads 32 \
	--d_ff 4096 \
	--attn_pdrop 0.1 \
	--resid_pdrop 0.1 \
	--lr_max 0.001 \
	--lr_min 0.00001 \
	--t_warmup 500 \
	--t_cos 10000 \
	--train_batch_size 32 \
	--val_batch_size 16 \
	--num_steps 6000 \
	--num_val_batches 10 \
	--train_dataset "/data/owt-train.bin" \
	--val_dataset "/data/owt-valid.bin"
