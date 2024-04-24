import argparse
import torch
from tqdm import tqdm
import os
import wandb
import numpy as np
import logging
from functools import partial
from torch.optim.lr_scheduler import LambdaLR

from models.transformer.transformer import TransformerLM
from models.transformer.util import (
    AdamW,
    cross_entropy_loss,
    cosine_learning_rate_schedule,
    clip_gradients,
    perplexity,
)
from models.util import save_checkpoint, load_checkpoint, load_batch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s (%(levelname)s): %(message)s"
)

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(
        self,
        model,
        train_dataloader,
        valid_dataloader,
        optimizer,
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
        lr_min,
        val_every,
        use_scheduler,
        t_warmup,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.optimizer = optimizer
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
        self.lr_min = lr_min
        self.val_every = val_every
        self.use_scheduler = use_scheduler
        self.t_warmup = t_warmup
        self.scheduler = None
        self.best_val_loss = float("inf")

        if self.use_scheduler:
            # self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            #     self.optimizer,
            #     lr_lambda=lambda step: cosine_learning_rate_schedule(
            #         step,
            #         self.lr,
            #         self.lr_min,
            #         self.t_warmup,
            #         self.num_steps,
            #     ),
            # )
            print("Using scheduler")
            print(f"lr: {self.lr}, lr_min: {self.lr_min}, t_warmup: {self.t_warmup}, num_steps: {self.num_steps}")
            lr_lambda = partial(
                cosine_learning_rate_schedule,
                max_learning_rate=self.lr,
                min_learning_rate=self.lr_min,
                warmup_iters=self.t_warmup,
                cosine_cycle_iters=self.num_steps,
            )

            self.scheduler = LambdaLR(self.optimizer, lr_lambda)

    def validate(self):
        self.model.eval()
        total_valid_loss = 0
        total_perpl = 0
        with torch.no_grad():
            for _ in tqdm(range(self.num_val_batches), desc="Validation"):
                inputs, targets = load_batch(
                    self.valid_dataloader,
                    self.val_batch_size,
                    self.context_length,
                    self.device,
                )
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.model(inputs)
                valid_loss = cross_entropy_loss(logits, targets)
                total_perpl += perplexity(logits, targets)
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
            inputs, targets = load_batch(
                self.train_dataloader,
                self.train_batch_size,
                self.context_length,
                self.device,
            )
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(inputs)
            loss = cross_entropy_loss(logits, targets)
            loss.backward()

            clip_gradients(self.model.parameters(), self.clip_norm)
            self.optimizer.step()

            if self.use_scheduler and self.scheduler is not None:
                self.scheduler.step()

            total_train_loss += loss.item()

            if current_step % 100 == 0:
                average_train_loss = total_train_loss / (current_step + 1)
                wandb.log({"average_train_loss": average_train_loss})

            if current_step % self.val_every == 0:
                val_loss, val_perpl = self.validate()
                logger.info(
                    f"Validation Loss: {val_loss:.4f}, Perplexity: {val_perpl:.4f}, lr: {self.optimizer.param_groups[0]['lr']}"
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


def main():
    torch.manual_seed(42)

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Train a Transformer model with custom hyperparameters and utilities."
    )
    parser.add_argument("--name", type=str, default="tiny")
    parser.add_argument("--train_dataset", type=str, required=True)
    parser.add_argument("--val_dataset", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--ctx_len", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--attn_pdrop", type=float, default=0.1)
    parser.add_argument("--residual_pdrop", type=float, default=0.1)
    parser.add_argument("--tie", action="store_true")
    parser.add_argument("--lr_max", type=float, default=1e-2)
    parser.add_argument("--lr_min", type=float, default=1e-4)
    parser.add_argument("--t_warmup", type=int, default=0)
    parser.add_argument("--t_cos", type=int, default=1280000 // 256)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument(
        "--resume",
        type=bool,
        default=False,
        help="Resume training from the latest checkpoint.",
    )
    parser.add_argument("--num_steps", type=int, default=12800000 // 256)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--val_batch_size", type=int, default=128)
    parser.add_argument("--num_val_batches", type=int, default=2)
    parser.add_argument("--post_norm", type=bool, default=False)
    parser.add_argument("--layer_norm", action="store_true")
    parser.add_argument("--no_layer_norm", action="store_false", dest="layer_norm")
    parser.add_argument("--val_every", type=int, default=400)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--use_scheduler", action="store_true")

    # Parse arguments
    args = parser.parse_args()

    # Initialize WandB
    run_name = f"lr{args.lr_max}-bs{args.train_batch_size}"
    wandb.init(
        project="cs336-assignment-1-review",
        entity="ee2023ee2023ee",
        config=vars(args),
        name=args.name,
    )

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data loading
    train_data, valid_data = np.memmap(
        args.train_dataset, dtype=np.uint16, mode="r"
    ), np.memmap(args.val_dataset, dtype=np.uint16, mode="r")

    # Model initialization
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.ctx_len,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        attn_pdrop=args.attn_pdrop,
        residual_pdrop=args.residual_pdrop,
        post_norm=args.post_norm,
        layer_norm=args.layer_norm,
    ).to(device)

    # Optimizer setup
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr_max,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # Checkpoint directory
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Trainer initialization and training
    trainer = Trainer(
        model=model,
        train_dataloader=train_data,
        valid_dataloader=valid_data,
        optimizer=optimizer,
        clip_norm=1.0,
        device=device,
        checkpoint_dir=checkpoint_dir,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        context_length=args.ctx_len,
        num_steps=args.num_steps,
        num_val_batches=args.num_val_batches,
        name=args.name,
        resume=args.resume,
        lr=args.lr_max,
        lr_min=args.lr_min,
        val_every=args.val_every,
        use_scheduler=args.use_scheduler,
        t_warmup=args.t_warmup,
    )

    trainer.train()


if __name__ == "__main__":
    main()
