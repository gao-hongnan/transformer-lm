from perf.bpe.util import profile_train_bpe


def profile_bpe_training():
    path = "/data/owt_train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]

    profile_train_bpe(path, vocab_size, special_tokens, "owt")


if __name__ == "__main__":
    profile_bpe_training()
