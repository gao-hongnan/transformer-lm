# generated from a612dbd60f49f41b2752d68b5c53b750eaf1f532
python evaluate.py \
    --prompt "Once upon a time," \
    --max_length 256 \
    --temperature 0.8 \
    --top_p 0.85 \
    --vocab_size 10000 \
    --context_len 256 \
    --d_model 512 \
    --num_layers 4 \
    --num_heads 16 \
    --d_ff 2048 \
    --attn_pdrop 0.1 \
    --resid_pdrop 0.1 \
    --checkpoint_path "./data/checkpoints/a612dbd60f49f41b2752d68b5c53b750eaf1f532_tiny_best_0.0003_128.pth" \
    --vocab_filepath "./data/TinyStoriesV2-GPT4-train_vocab.pkl" \
    --merges_filepath "./data/TinyStoriesV2-GPT4-train_merges.pkl"