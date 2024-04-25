import collections
import logging
import time
from typing import Dict, List, Set, Tuple

import regex as re
from tqdm import tqdm

from models.tokenizer.vocab import Vocab

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s (%(levelname)s): %(message)s"
)
logger = logging.getLogger(__name__)


def extract_subword_frequencies(
    input_path: str, special_tokens: Set[str], pattern: re.Pattern
) -> Dict[str, int]:
    frequencies = {}

    for match in pattern.finditer(
        open(input_path, "r", encoding="utf-8").read(), concurrent=True
    ):
        match_str = match.group()
        if match_str not in special_tokens:
            frequencies[match_str] = frequencies.get(match_str, 0) + 1

    return frequencies


def encode_subwords(subwords: List[str]) -> List[List[bytes]]:
    return [[bytes([byte]) for byte in subword.encode("utf-8")] for subword in subwords]


def calculate_byte_pair_frequencies(
    encoded_subwords: List[List[bytes]], subword_frequencies: Dict[str, int]
) -> Tuple[Dict[Tuple[bytes, bytes], int], Dict[Tuple[bytes, bytes], Dict[int, int]]]:
    byte_pair_frequencies = collections.defaultdict(int)
    token_indices = {}
    for index, encoded_subword in enumerate(encoded_subwords):
        subword_str = bytes(b"".join(encoded_subword)).decode("utf-8")
        frequency = subword_frequencies[subword_str]
        for i in range(len(encoded_subword) - 1):
            byte_pair = (encoded_subword[i], encoded_subword[i + 1])
            byte_pair_frequencies[byte_pair] += frequency
            if byte_pair not in token_indices:
                token_indices[byte_pair] = {}
            token_indices[byte_pair][index] = frequency
    return byte_pair_frequencies, token_indices


def update_frequencies_after_merge(
    encoded_subwords: List[List[bytes]],
    subword_index: int,
    byte_index: int,
    new_byte: bytes,
    frequencies: Dict[Tuple[bytes, bytes], int],
    count: int,
) -> None:
    if byte_index > 0:
        left_pair = (
            encoded_subwords[subword_index][byte_index - 1],
            encoded_subwords[subword_index][byte_index],
        )
        frequencies[left_pair] -= count
        frequencies[
            (encoded_subwords[subword_index][byte_index - 1], new_byte)
        ] += count

    if byte_index < len(encoded_subwords[subword_index]) - 2:
        right_pair = (
            encoded_subwords[subword_index][byte_index + 1],
            encoded_subwords[subword_index][byte_index + 2],
        )
        frequencies[right_pair] -= count
        frequencies[
            (new_byte, encoded_subwords[subword_index][byte_index + 2])
        ] += count


def update_token_indices(
    encoded_subwords: List[List[bytes]],
    subword_index: int,
    byte_index: int,
    token_indices: Dict[Tuple[bytes, bytes], Dict[int, int]],
) -> None:
    def pair_exists(subword, pair):
        return pair in zip(subword, subword[1:])

    if byte_index > 0:
        pair = (
            encoded_subwords[subword_index][byte_index - 1],
            encoded_subwords[subword_index][byte_index],
        )
        if not pair_exists(encoded_subwords[subword_index], pair):
            del token_indices[pair][subword_index]

    if byte_index < len(encoded_subwords[subword_index]) - 2:
        pair = (
            encoded_subwords[subword_index][byte_index + 1],
            encoded_subwords[subword_index][byte_index + 2],
        )
        if not pair_exists(encoded_subwords[subword_index], pair):
            del token_indices[pair][subword_index]


def create_new_token_indices(
    subwords: List[List[bytes]],
    tkn_idx: int,
    merged_idx: int,
    token_indices: Dict[Tuple[bytes, bytes], Dict[int, int]],
    count: int,
) -> None:
    if merged_idx > 0:
        left_pair = (
            subwords[tkn_idx][merged_idx - 1],
            subwords[tkn_idx][merged_idx],
        )
        if left_pair not in token_indices:
            token_indices[left_pair] = {}
        token_indices[left_pair][tkn_idx] = count
    if merged_idx < len(subwords[tkn_idx]) - 1:
        right_pair = (
            subwords[tkn_idx][merged_idx],
            subwords[tkn_idx][merged_idx + 1],
        )
        if right_pair not in token_indices:
            token_indices[right_pair] = {}
        token_indices[right_pair][tkn_idx] = count


def merge_subwords(
    encoded_subwords: List[List[bytes]],
    subword_index: int,
    byte_index: int,
    new_byte: bytes,
) -> None:
    encoded_subwords[subword_index][byte_index] = new_byte
    encoded_subwords[subword_index].pop(byte_index + 1)


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str] = []):
    pattern = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
        re.UNICODE,
    )

    start_time = time.time()
    logger.info("Creating vocab")
    vocab = Vocab(special_tokens=special_tokens)
    logger.info("Took %s seconds to create vocab", round(time.time() - start_time, 2))

    start_time = time.time()
    logger.info("Extracting subword frequencies")
    subword_frequencies = extract_subword_frequencies(
        input_path, set(special_tokens), pattern
    )
    logger.info(
        "Took %s seconds to extract subword frequencies",
        round(time.time() - start_time, 2),
    )

    start_time = time.time()
    logger.info("Encoding subwords")
    encoded_subwords = encode_subwords(list(subword_frequencies.keys()))
    logger.info(
        "Took %s seconds to encode subwords", round(time.time() - start_time, 2)
    )

    start_time = time.time()
    logger.info("Calculating byte pair frequencies")
    byte_pair_frequencies, token_indices = calculate_byte_pair_frequencies(
        encoded_subwords, subword_frequencies
    )
    logger.info(
        "Took %s seconds to calculate byte pair frequencies",
        round(time.time() - start_time, 2),
    )

    start_time = time.time()
    logger.info("Merging subwords")
    merges = []
    for _ in tqdm(range(vocab_size - len(vocab))):
        if len(byte_pair_frequencies) == 0:
            break

        best_pair = max(
            byte_pair_frequencies, key=lambda x: (byte_pair_frequencies[x], x)
        )
        new_byte = best_pair[0] + best_pair[1]
        vocab.add_token(new_byte)
        subword_indices = list(token_indices[best_pair].keys())

        for subword_index in subword_indices:
            byte_index = 0
            while byte_index < len(encoded_subwords[subword_index]) - 1:
                if (
                    encoded_subwords[subword_index][byte_index] == best_pair[0]
                    and encoded_subwords[subword_index][byte_index + 1] == best_pair[1]
                ):
                    count = token_indices[best_pair][subword_index]

                    update_frequencies_after_merge(
                        encoded_subwords,
                        subword_index,
                        byte_index,
                        new_byte,
                        byte_pair_frequencies,
                        count,
                    )
                    update_token_indices(
                        encoded_subwords, subword_index, byte_index, token_indices
                    )
                    merge_subwords(
                        encoded_subwords, subword_index, byte_index, new_byte
                    )
                    create_new_token_indices(
                        encoded_subwords,
                        subword_index,
                        byte_index,
                        token_indices,
                        count,
                    )
                byte_index += 1

        byte_pair_frequencies.pop(best_pair)
        token_indices.pop(best_pair)
        merges.append(best_pair)
    logger.info("Took %s seconds to merge subwords", round(time.time() - start_time, 2))

    return vocab.get_idx_to_token(), merges
