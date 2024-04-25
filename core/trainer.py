import argparse
import time
from argparse import Namespace
from functools import partial
from os import PathLike
from pathlib import Path
from typing import IO, BinaryIO

import numpy as np
import torch
import torch.nn as nn
import wandb
from rich.pretty import pprint
from tqdm.auto import tqdm

from core.config import GPTConfig
from core.data import data_generator, get_batch
from core.layers import GPT
from core.nn_utils import CrossEntropyLoss, cross_entropy_loss, gradient_clipping
from core.optimizer import AdamW
from core.scheduler import _cosine_schedule_with_warmup_and_post_annealing_lr_lambda


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | PathLike | BinaryIO | IO[bytes],
) -> None:
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        out,
    )


def load_checkpoint(
    src: str | PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train transformer model")
    parser.add_argument("--checkpoint-interval", default=10_000, type=int)
    parser.add_argument("--log-interval", default=100, type=int)
    parser.add_argument("--checkpoint-path", default="checkpoints/", type=str)

    parser.add_argument(
        "--dataset", default="tinystories", type=str, choices=["owt", "tinystories"]
    )
    parser.add_argument("--data_dir", default="data/", type=str)
    parser.add_argument("--d_ff", default=2048, type=int)
    parser.add_argument(
        "--user_defined_total_steps", default=None, type=int
    )  # Just an estimate
    parser.add_argument("--d_model", default=512, type=int)
    parser.add_argument("--num_heads", default=16, type=int)
    parser.add_argument("--num_layer", default=4, type=int)
    parser.add_argument("--context_length", default=256, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--parallel_layers", default=False, type=bool)
    parser.add_argument("--post_norm", default=False, type=bool)
    parser.add_argument("--name", default=None)
    parser.add_argument("--rotary", default=True, type=bool)
    parser.add_argument("--activation", default="gelu", type=str, choices=["gelu"])
    parser.add_argument("--weight_tie", default=True, type=bool)
    parser.add_argument("--decay", default=0.1, type=float)
    parser.add_argument("--flash", default=False, type=bool)
    parser.add_argument("--num_epochs", default=1, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--compile", default=False, type=bool)
    parser.add_argument("--bias", default=False, type=bool)
    parser.add_argument("--vocab_size", default=10_000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--warmup_steps", default=500, type=int)

    return parser.parse_args()


def train_model(args: Namespace) -> None:
    vocab_size = args.vocab_size
    if args.dataset == "owt":
        assert vocab_size == 32_000
    elif args.dataset == "tinystories":
        assert vocab_size == 10_000
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    gpt_config = GPTConfig(
        approximate=None,
        activation_name=args.activation,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        context_length=args.context_length,
        attn_pdrop=args.dropout,
        resid_pdrop=args.dropout,
        bias=False,
        vocab_size=vocab_size,
        num_blocks=args.num_layer,
        token_position_pdrop=args.dropout,
        weight_tie=args.weight_tie,
    )
    pprint(gpt_config)

    model = GPT(config=gpt_config)
    pprint(model)
    model.to("cuda")
    print(
        f"model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
    )

    if args.compile:
        model = torch.compile(model, fullgraph=True)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # TODO: investigate
    # loss_func = torch.compile(cross_entropy_loss, fullgraph=True)
    loss_func = CrossEntropyLoss()

    train_dataset = np.memmap(
        f"{args.data_dir}/{args.dataset}_train_tokens.npy", dtype=np.uint16, mode="r"
    )
    valid_dataset = np.memmap(
        f"{args.data_dir}/{args.dataset}_valid_tokens.npy", dtype=np.uint16, mode="r"
    )
    print(f"Train dataset: {train_dataset.shape}")
    print(f"Firt 10 elements: {train_dataset[:10]}")

    checkpoints_dir = Path(args.checkpoint_path)
    checkpoints_dir.mkdir(exist_ok=True)

    # Format time to be used in checkpoint filenames
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")

    train_dataset_size = len(train_dataset)
    print(f"Train dataset size AKA total tokens: {train_dataset_size}")
    total_samples = train_dataset_size // args.context_length
    print(f"Total samples: {total_samples}")
    total_batches = total_samples // args.batch_size
    print(f"Total batches: {total_batches}")

    total_steps_per_epoch = total_batches
    print(f"Total steps per epoch: {total_steps_per_epoch}")

    if args.user_defined_total_steps is not None:
        total_steps = args.user_defined_total_steps
    else:
        total_steps = total_steps_per_epoch * args.num_epochs

    print(f"Total steps: {total_steps}")

    scheduler = partial(
        _cosine_schedule_with_warmup_and_post_annealing_lr_lambda,
        max_learning_rate=args.lr,
        min_learning_rate=args.lr * args.decay,
        warmup_iters=args.warmup_steps,  # int(total_steps * 0.1)
        cosine_cycle_iters=total_steps,
    )

    with torch.autocast(
        device_type="cuda", dtype=torch.bfloat16
    ):  # FIXME: change to bfloat
        model.train()
        section_training_loss = (0.0, 0)

        pbar = tqdm(
            data_generator(
                train_dataset,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device_type="cuda",
            )
        )
        for training_step, (train_x, train_y) in enumerate(pbar):
            if (
                args.user_defined_total_steps is not None
                and training_step >= args.user_defined_total_steps
            ):
                break

            lr_value = scheduler(training_step)
            optimizer.zero_grad()
            optimizer.param_groups[0]["lr"] = lr_value
            y_pred = model(train_x)
            # training_loss = loss_func(y_pred, train_y)

            training_loss = cross_entropy_loss(y_pred, train_y)

            training_loss.backward()
            gradient_clipping(model.parameters(), max_norm=1.0)
            optimizer.step()
            section_training_loss = (
                section_training_loss[0] + training_loss.item(),
                section_training_loss[1] + 1,
            )
            del training_loss
            if training_step % args.checkpoint_interval == 0 and training_step != 0:
                save_checkpoint(
                    model,
                    optimizer,
                    training_step,
                    checkpoints_dir
                    / f"{wandb.run.name}_{time_str}_{training_step // 1000}k",
                )
            if training_step % args.log_interval == 0:
                model.eval()
                # Compute validation loss
                valid_x, valid_y = get_batch(
                    dataset=valid_dataset,
                    batch_size=args.batch_size,
                    context_length=args.context_length,
                    device_type="cuda",
                )
                with torch.no_grad():
                    # valid_loss = loss_func(model(valid_x), valid_y)
                    valid_loss = cross_entropy_loss(model(valid_x), valid_y)
                training_loss_avg = section_training_loss[0] / section_training_loss[1]
                section_training_loss = (0.0, 0)
                if training_step == 0:
                    wandb.init(
                        project="cs336-assignment-1",
                        entity="ee2023ee2023ee",
                        config=vars(args),
                        name=args.name,
                    )
                wandb.log({"loss/train": training_loss_avg}, step=training_step)
                wandb.log(
                    {"perplexity/train": torch.exp(torch.tensor(training_loss_avg))},
                    step=training_step,
                )
                wandb.log({"loss/valid": valid_loss.item()}, step=training_step)
                wandb.log(
                    {"perplexity/valid": torch.exp(valid_loss).item()},
                    step=training_step,
                )
                wandb.log({"lr": optimizer.param_groups[0]["lr"]}, step=training_step)
                pbar.set_postfix(
                    {
                        "train_loss": training_loss_avg,
                        "valid_loss": valid_loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "train_ppl": torch.exp(torch.tensor(training_loss_avg)).item(),
                        "valid_ppl": torch.exp(valid_loss).item(),
                    }
                )
                model.train()


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn

    train_model(args)
