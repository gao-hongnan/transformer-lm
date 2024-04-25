class Vocab:
    def __init__(self, special_tokens: list[str] = []) -> None:
        self.idx_to_token: dict[int, bytes] = {}

        for token in special_tokens:
            self.add_token(token.encode("utf-8"))

        # init 256 possible byte values
        for i in range(256):
            self.add_token(bytes([i]))

        self.unk_idx: int = 0

    @classmethod
    def from_dict(cls, vocab: dict[int, bytes], special_tokens: list[str] = []) -> "Vocab":
        instance = cls(special_tokens)
        instance.idx_to_token = vocab
        return instance

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, idx: int) -> bytes:
        return self.idx_to_token.get(idx, self.idx_to_token[self.unk_idx])

    def add_token(self, token: bytes) -> None:
        if token in self.idx_to_token.values():
            return

        idx = len(self.idx_to_token)
        self.idx_to_token[idx] = token
        return

    def get_inv(self) -> dict[bytes, int]:
        return {v: k for k, v in self.idx_to_token.items()}

    def get_idx_to_token(self) -> dict[int, bytes]:
        return self.idx_to_token

    def set_unk_idx(self, unk_idx: int) -> None:
        self.unk_idx = unk_idx
