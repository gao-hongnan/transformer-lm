from typing import Literal

from pydantic import BaseModel


class GPTConfig(BaseModel):
    approximate: Literal["tanh"] | None = None
    activation_name: Literal["gelu"] = "gelu"
    d_model: int
    d_ff: int
    num_heads: int
    context_length: int
    attn_pdrop: float = 0.0
    resid_pdrop: float = 0.0
    bias: bool = False
    vocab_size: int
    num_blocks: int
    token_position_pdrop: float = 0.0
    weight_tie: bool = False
