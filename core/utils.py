import os
from typing import IO, BinaryIO

import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    loaded_checkpoint = torch.load(src, map_location=torch.device("cpu"))
    model.load_state_dict(loaded_checkpoint["model_state_dict"])
    optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    return loaded_checkpoint["iteration"]
