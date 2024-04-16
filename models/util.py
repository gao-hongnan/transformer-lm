import torch

import numpy as np
import numpy.typing as npt

from typing import IO, BinaryIO, Iterable, Optional, Type
import os


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "optimizer_state_dict": optimizer.state_dict(),
        "model_state_dict": model.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    loaded_checkpoint = torch.load(src)
    optimizer.load_state_dict(loaded_checkpoint["optimizer_state_dict"])
    model.load_state_dict(loaded_checkpoint["model_state_dict"])

    return loaded_checkpoint["iteration"]


def load_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a batch of data from the dataset."""

    if torch.cuda.is_available() and "cuda" in device:
        device = device
    else:
        device = "cpu"

    inputs = np.zeros((batch_size, context_length))
    target_labels = np.zeros((batch_size, context_length))

    l = len(dataset) - context_length
    start_idx = np.random.randint(0, l, batch_size)
    for row, idx in enumerate(start_idx):

        inputs[row] = dataset[idx : idx + context_length]
        target_labels[row] = dataset[idx + 1 : idx + context_length + 1]

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)

    return inputs, target_labels