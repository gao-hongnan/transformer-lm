from perf.bpe.util import profile_train_bpe


def profile_bpe_training():
    path = "/data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]

    profile_train_bpe(path, vocab_size, special_tokens, "tiny")


if __name__ == "__main__":
    profile_bpe_training()
