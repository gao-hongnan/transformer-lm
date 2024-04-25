import argparse
import logging
import os
from functools import partial

import numpy as np
import torch
import wandb
from omnivault.modules.loss import CrossEntropyLoss
from omnivault.modules.nn_utils import gradient_clipping
from omnivault.optimizers.adamw import AdamW
from omnivault.schedulers.cosine_annealing_warmup import _cosine_schedule_with_warmup_and_post_annealing_lr_lambda
from rich.pretty import pprint
from tqdm.auto import tqdm

from core.config import GPTConfig, parse_args
from core.data import get_batch
from core.layers import GPT
from core.utils import load_checkpoint, save_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s (%(levelname)s): %(message)s")

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        criterion: CrossEntropyLoss,
        optimizer,
        scheduler,
        train_dataloader,
        valid_dataloader,
        clip_norm,
        device,
        checkpoint_dir,
        train_batch_size,
        val_batch_size,
        context_length,
        num_steps,
        num_val_batches,
        name,
        resume,
        lr,
        val_every,
        t_warmup,
        vocab_size,
    ) -> None:
        self.model = model

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.clip_norm = clip_norm
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.context_length = context_length

        self.num_steps = num_steps
        self.num_val_batches = num_val_batches

        self.name = name
        self.resume = resume
        self.lr = lr
        self.val_every = val_every
        self.t_warmup = t_warmup
        self.best_val_loss = float("inf")
        self.vocab_size = vocab_size

    def validate(self):
        self.model.eval()
        total_valid_loss = 0
        total_perpl = 0
        with torch.no_grad():
            for _ in tqdm(range(self.num_val_batches), desc="Validation"):
                inputs, targets = get_batch(
                    dataset=self.valid_dataloader,
                    batch_size=self.val_batch_size,
                    context_length=self.context_length,
                    device_type=self.device.type,
                )
                # FIXME: min to bypass shape error at random
                inputs = torch.minimum(inputs, torch.tensor(self.vocab_size - 1))
                targets = torch.minimum(targets, torch.tensor(self.vocab_size - 1))
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs)
                valid_loss = self.criterion(logits, targets)
                total_perpl += torch.exp(valid_loss)
                total_valid_loss += valid_loss.item()

            average_valid_loss = total_valid_loss / self.num_val_batches
            average_perpl = total_perpl / self.num_val_batches
            wandb.log(
                {
                    "average_valid_loss": average_valid_loss,
                    "average_perpl": average_perpl,
                }
            )
            return average_valid_loss, average_perpl

    def train(self):
        current_step = 0
        if self.resume:
            current_step = load_checkpoint(
                f"{self.checkpoint_dir}/{self.name}_best_{self.lr}_{self.train_batch_size}.pth",
                self.model,
                self.optimizer,
            )

        total_train_loss = 0
        for current_step in tqdm(
            range(current_step, self.num_steps),
            desc=f"Training Step {self.num_steps+1}",
        ):
            inputs, targets = get_batch(
                dataset=self.train_dataloader,
                batch_size=self.train_batch_size,
                context_length=self.context_length,
                device_type=self.device.type,
            )
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            loss.backward()

            gradient_clipping(self.model.parameters(), max_norm=self.clip_norm, epsilon=1e-6)
            self.optimizer.step()

            if self.scheduler is not None:
                # self.scheduler.step(current_step)
                lr_or_lrs = self.scheduler(current_step)
                if isinstance(lr_or_lrs, (float, int)):
                    self.optimizer.param_groups[0]["lr"] = lr_or_lrs
                elif isinstance(lr_or_lrs, (list, tuple)):
                    for i, lr in enumerate(lr_or_lrs):
                        self.optimizer.param_groups[i]["lr"] = lr

            total_train_loss += loss.item()

            if current_step % 100 == 0:
                average_train_loss = total_train_loss / (current_step + 1)

                wandb.log({"average_train_loss": average_train_loss})

            if current_step % self.val_every == 0:
                val_loss, val_perpl = self.validate()

                logger.info(
                    f"Training Loss: {total_train_loss / (current_step + 1):.4f} | Validation Loss: {val_loss:.4f}, Perplexity: {val_perpl:.4f}, lr: {self.optimizer.param_groups[0]['lr']}"
                )
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    latest_checkpoint_path = os.path.join(
                        self.checkpoint_dir,
                        f"{self.name}_best_{self.lr}_{self.train_batch_size}.pth",
                    )
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.num_steps,
                        latest_checkpoint_path,
                    )

        average_train_loss = total_train_loss / self.num_steps
        logger.info(f"Training Loss: {average_train_loss:.4f}")
        wandb.log({"average_train_loss": average_train_loss})


def main(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    run_name = f"lr{args.lr_max}-bs{args.train_batch_size}"
    wandb.init(
        project=f"cs336-assignment-1-{args.name}",
        entity="ee2023ee2023ee",
        config=vars(args),
        name=run_name,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_data, valid_data = np.memmap(args.train_dataset, dtype=np.uint16, mode="r"), np.memmap(
        args.valid_dataset, dtype=np.uint16, mode="r"
    )

    gpt_config = GPTConfig(
        approximate=args.gelu_approximation,
        activation_name=args.activation_name,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        context_length=args.context_length,
        attn_pdrop=args.attn_pdrop,
        resid_pdrop=args.resid_pdrop,
        bias=args.linear_bias,
        vocab_size=args.vocab_size,
        num_blocks=args.num_layers,
        token_position_pdrop=args.token_position_pdrop,
        weight_tie=args.weight_tie,
    )
    pprint(gpt_config)

    model = GPT(config=gpt_config)
    model.to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
        eps=args.eps,
    )
    criterion = CrossEntropyLoss()

    scheduler = partial(
        _cosine_schedule_with_warmup_and_post_annealing_lr_lambda,
        max_learning_rate=args.lr_max,
        min_learning_rate=args.lr_min,
        warmup_iters=args.t_warmup,
        cosine_cycle_iters=args.t_cos,
    )
    if args.t_cos != args.num_steps:
        logger.warning(
            f"Number of steps ({args.num_steps}) and cosine cycle iterations ({args.t_cos}) are not equal. Will have more than 1 cycle."
        )

    # Checkpoint directory
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Trainer initialization and training
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_dataloader=train_data,
        valid_dataloader=valid_data,
        clip_norm=args.clip_norm,
        device=device,
        checkpoint_dir=checkpoint_dir,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        context_length=args.context_length,
        num_steps=args.num_steps,
        num_val_batches=args.num_val_batches,
        name=args.name,
        resume=args.resume,
        lr=args.lr_max,
        lr_min=args.lr_min,
        val_every=args.val_every,
        t_warmup=args.t_warmup,
        vocab_size=args.vocab_size,
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    pprint(args)
    main(args=args)
