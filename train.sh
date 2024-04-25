!python3 train.py \
	--name "tiny" \
	--vocab_size 10000 \
	--context_length 256 \
	--d_model 512 \
	--num_layers 4 \
	--num_heads 16 \
	--d_ff 2048 \
	--attn_pdrop 0.1 \
	--resid_pdrop 0.1 \
	--token_position_pdrop 0.1 \
	--weight_tie \
	--activation_name "gelu" \
	--lr_max 3e-4 \
	--lr_min 3e-5 \
	--t_warmup 500 \
	--t_cos 10000 \
	--checkpoint_dir "./checkpoints" \
	--clip_norm 1.0 \
	--train_batch_size 128 \
	--val_batch_size 128 \
	--num_steps 10000 \
	--num_val_batches 1 \
    --val_every 20 \
	--use_scheduler \
	--train_dataset "/content/drive/MyDrive/Colab Notebooks/tinystories_train_tokens.npy" \
	--valid_dataset "/content/drive/MyDrive/Colab Notebooks/tinystories_valid_tokens.npy"