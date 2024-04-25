from perf.bpe.util import profile_train_bpe


def profile_bpe_training():
    path = "tests/fixtures/corpus.en"
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]

    profile_train_bpe(path, vocab_size, special_tokens, "corpus")


if __name__ == "__main__":
    profile_bpe_training()
