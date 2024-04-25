import cProfile
import io
import pstats
import time

from memory_profiler import profile

from models.tokenizer.tokenizer import Tokenizer


@profile
def profile_train_bpe(input_path, vocab_size, special_tokens, name):
    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()
    # vocab, merges = tokenizer.from_file(input_path)
    tokenizer = Tokenizer.train_from_file(input_path, vocab_size, special_tokens)
    end_time = time.time()

    pr.disable()

    tokenizer.save("data/tokenizer", name)

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("time")
    ps.print_stats()
    # print(s.getvalue())

    training_time = end_time - start_time
    print(f"Training time: {training_time:.2f} seconds")
