from __future__ import annotations

import logging
import pickle
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, List, Literal, Tuple, Type

import numpy as np
import regex as re
from rich.pretty import pprint

from rustsrc import RustTokenizer, train_bpe

logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s): %(message)s")
logger = logging.getLogger(__name__)

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
DATASET_TRAIN_PATHS = {
    "owt": "data/owt_train.txt",
    "tinystories": "data/TinyStoriesV2-GPT4-train.txt",
}
DATASET_VALID_PATHS = {
    "owt": "data/owt_valid.txt",
    "tinystories": "data/TinyStoriesV2-GPT4-valid.txt",
}
TEST_DATASET_PATH = "data/taylorswift.txt"


@contextmanager
def timer(block_name: str) -> Generator[None, None, None]:
    start = time.time()
    yield
    logger.info(f"{block_name} took {time.time() - start} seconds")


@dataclass
class Tokenizer:
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]
    rust_tokenizer: RustTokenizer | None

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        if special_tokens is None:
            special_tokens = []
        self.rust_tokenizer = RustTokenizer(vocab, merges, special_tokens)

    @classmethod
    def from_corpus(
        cls: Type[Tokenizer],
        text: bytes | str,
        vocab_size: int,
        special_tokens: List[str] | None = None,
    ) -> Tokenizer:
        if isinstance(text, str):
            text = text.encode("utf-8", errors="strict")

        if special_tokens is None:
            special_tokens = []

        with timer("Training the bpe model"):
            vocab, merges = train_bpe(text, vocab_size, special_tokens)
        return cls(vocab, merges, special_tokens)

    @classmethod
    def load_tokenizer_from(
        cls: Type[Tokenizer],
        vocab_filepath: Path | str,
        merges_filepath: Path | str,
        special_tokens: List[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    @classmethod
    def from_files(
        cls: Type[Tokenizer],
        in_path: Path | str,
        vocab_size: int,
        special_tokens: List[str] | None = None,
    ) -> Tokenizer:
        with open(in_path, "rb") as f:
            text = f.read()
        return cls.from_corpus(text, vocab_size, special_tokens)

    def encode(self, text: str | bytes, as_list: bool = True) -> List[int] | np.ndarray:
        if isinstance(text, str):
            text = text.encode("utf-8", errors="replace")
        tokens = self.rust_tokenizer.encode(text)
        if as_list:
            return tokens.tolist()
        return tokens

    def encode_iterable(self, iterable: Iterable[str], as_list: bool = True) -> Iterator[int] | Iterator[np.ndarray]:
        for text in iterable:
            yield from self.encode(text, as_list=as_list)

    def decode(self, tokens: List[int] | np.ndarray) -> str:
        return self.rust_tokenizer.decode(tokens)


def train_on_dataset(dataset_name: Literal["owt", "tinystories"]):
    assert dataset_name in ["owt", "tinystories"]
    in_path = Path(DATASET_TRAIN_PATHS[dataset_name])
    tokenizer = Tokenizer.from_files(
        in_path=in_path,
        vocab_size=10_000 if dataset_name == "tinystories" else 32_000,
        special_tokens=["<|endoftext|>"],
    )
    out_path = Path("tokenizers/")
    with open(out_path / f"{in_path.stem}_vocab.pkl", "wb") as f:
        pickle.dump(tokenizer.vocab, f)
    with open(out_path / f"{in_path.stem}_merges.pkl", "wb") as f:
        pickle.dump(tokenizer.merges, f)
    return tokenizer


def load_tokenizer_for_dataset(dataset: Literal["owt", "tinystories"]) -> Tokenizer:
    dataset_path = Path(DATASET_TRAIN_PATHS[dataset])
    tokenizer_path = Path("tokenizers/")
    with open(tokenizer_path / f"{dataset_path.stem}_vocab.pkl", "rb") as f:
        vocab = pickle.load(f)
    with open(tokenizer_path / f"{dataset_path.stem}_merges.pkl", "rb") as f:
        merges = pickle.load(f)

    tokenizer = Tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    return tokenizer


def sample_and_compress(
    dataset: Literal["owt", "tinystories"],
    with_tokenizer_from: Literal["owt", "tinystories"] | None = None,
):
    if with_tokenizer_from is None:
        with_tokenizer_from = dataset
    tokenizer = load_tokenizer_for_dataset(with_tokenizer_from)
    with open(DATASET_TRAIN_PATHS[dataset], "r") as f:
        text = f.read(1_000_000)
    # Get the fist 10 documents
    first_10_lines = next(iter(re.finditer(r"([\s\S]*?<\|endoftext\|>){10}", text))).group(0)
    before_len = len(first_10_lines.encode("utf-8", errors="replace"))
    tokens = tokenizer.encode(first_10_lines, as_list=False)
    after_len = tokens.nbytes
    compression_ratio = before_len / after_len
    print(f"Compression ratio for {dataset}: {compression_ratio}")


def tokenizer_throughput(dataset: Literal["owt", "tinystories"]):
    tokenizer = load_tokenizer_for_dataset(dataset)
    n_bytes = 2**30
    with open(DATASET_TRAIN_PATHS[dataset], "rb") as f:
        text = f.read(n_bytes)
    print("Starting encode")
    start = time.time()
    _tokens = tokenizer.encode(text, as_list=False)
    end = time.time()
    print(f"Encoded {n_bytes} bytes in {end - start} seconds")
    print(f"Throughput: {(n_bytes / 1e9) / (end - start)} GB/s")


def tokenize_full_dataset(dataset: Literal["owt", "tinystories"]):
    tokenizer = load_tokenizer_for_dataset(dataset)
    with open(DATASET_TRAIN_PATHS[dataset], "rb") as f:
        text = f.read()
    print("Tokenizing...")
    tokens = tokenizer.encode(text, as_list=False)
    print("Done tokenizing!")
    with open(f"data/{dataset}_train_tokens.npy", "wb") as f:
        np.save(f, tokens)

    with open(DATASET_VALID_PATHS[dataset], "rb") as f:
        text = f.read()
    tokens = tokenizer.encode(text, as_list=False)
    with open(f"data/{dataset}_valid_tokens.npy", "wb") as f:
        np.save(f, tokens)


def test_training():
    with open("data/bpe_trivial.txt", "r") as f:
        text = f.read()
    tokenizer = Tokenizer.from_corpus(text, 256 + 12)
    print(tokenizer.vocab)
    print(tokenizer.merges)


if __name__ == "__main__":
    # test_training()
    # train_on_dataset("tinystories")
    # train_on_dataset("owt")

    # sample_and_compress('tinystories', 'tinystories')
    # sample_and_compress('owt', 'owt')

    # sample_and_compress('tinystories', 'owt')
    # sample_and_compress('owt', 'tinystories')

    # tokenizer_throughput('tinystories')
    # tokenizer_throughput('owt')

    tokenize_full_dataset("tinystories")
    # tokenize_full_dataset('owt')
