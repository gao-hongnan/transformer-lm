import argparse
from typing import Literal

from pydantic import BaseModel

__all__ = ["GPTConfig", "parse_args"]


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


def parse_args() -> argparse.Namespace:
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Train a Transformer model with custom hyperparameters and utilities.")
    parser.add_argument("--name", type=str, default="tiny")
    parser.add_argument("--train_dataset", type=str, default=None)
    parser.add_argument("--valid_dataset", type=str, default=None)

    parser.add_argument("--vocab_size", type=int, default=10_000)  # NOTE: Start of Model Metrics
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--resid_pdrop", type=float, default=0.1)
    parser.add_argument("--token_position_pdrop", type=float, default=0.1)
    parser.add_argument("--weight_tie", action="store_true")
    parser.add_argument("--linear_bias", action="store_true")
    parser.add_argument("--activation_name", type=str, default="gelu")
    parser.add_argument("--gelu_approximation", type=str, default=None)
    parser.add_argument("--post_norm", type=bool, default=False)
    parser.add_argument("--layer_norm", action="store_true")
    parser.add_argument("--no_layer_norm", action="store_false", dest="layer_norm")
    parser.add_argument("--parallel", action="store_true")

    parser.add_argument("--beta1", type=float, default=0.9)  # NOTE: Start Optimizer
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--eps", type=float, default=1e-8)

    parser.add_argument("--lr_max", type=float, default=1e-2)  # NOTE: Start of Scheduler
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--t_warmup", type=int, default=0)
    parser.add_argument("--t_cos", type=int, default=1280000 // 256)

    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")

    parser.add_argument("--num_steps", type=int, default=12800000 // 256)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--valid_batch_size", type=int, default=128)
    parser.add_argument("--num_val_batches", type=int, default=2)
    parser.add_argument("--clip_norm", type=float, default=1.0)

    parser.add_argument("--val_every", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        help="Path to the model checkpoint.",
    )
    parser.add_argument(
        "--vocab_filepath",
        type=str,
        help="Path to the vocabulary file.",
    )

    parser.add_argument(
        "--merges_filepath",
        type=str,
        help="Path to the merges file.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum length of the generated text.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for softmax scaling. Lower is more deterministic.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p value for nucleus sampling. Lower is more focused.",
    )
    parser.add_argument("--prompt", type=str, default="Once upon a time,")
    # Parse arguments
    args = parser.parse_args()
    return args
