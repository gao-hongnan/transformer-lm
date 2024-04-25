#!/bin/bash
#SBATCH --job-name=train_tiny
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=100G
#SBATCH --time=5:05:00
#SBATCH --output=train_tiny_%j.out
#SBATCH --error=train_tiny_%j.err
#SBATCH --gpus=1

source $(conda info --base)/etc/profile.d/conda.sh
conda activate transformer_lm

python3 train.py \
	--name "tiny" \
	--vocab_size 10000 \
	--context_length 256 \
	--d_model 512 \
	--num_layers 4 \
	--num_heads 16 \
	--d_ff 2048 \
	--attn_pdrop 0.1 \
	--resid_pdrop 0.1 \
	--lr_max 0.001 \
	--lr_min 0.00001 \
	--t_warmup 500 \
	--t_cos 10000 \
	--train_batch_size 64 \
	--val_batch_size 64 \
	--num_steps 5000 \
	--num_val_batches 1 \
	--train_dataset "/data/TinyStoriesV2-GPT4-train.bin" \
	--val_dataset "/data/TinyStoriesV2-GPT4-valid.bin"
