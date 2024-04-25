from __future__ import annotations

from typing import Literal, Tuple

import numpy as np
import numpy.typing as npt
import torch

__all__ = ["get_batch"]


def get_batch(
    *,
    dataset: npt.NDArray[np.uint16],
    batch_size: int,
    context_length: int,
    generator: torch.Generator | None = None,
    device_type: Literal["cpu", "cuda"] = "cpu",  # exclude mps
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generates a batch of input-output pairs from the given dataset.

    Parameters
    ----------
    dataset : numpy.ndarray
        The dataset to generate batches from. It should be a 1D NumPy array of type uint16.
    batch_size : int
        The number of samples in each batch.
    context_length : int
        The length of the context window for each sample.
    generator : torch.Generator, optional
        The PyTorch generator to use for generating random indices. If not provided,
        no seed is set.
    device_type : str, optional
        The device type to use for the generated tensors. Can be either "cpu" or "cuda".
        Defaults to "cpu".

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        A tuple containing two tensors:
        - x: The input tensor of shape (batch_size, context_length).
        - y: The output tensor of shape (batch_size, context_length), where each element
             is shifted by 1 compared to the corresponding element in x.

    Notes
    -----
    - The function assumes that the dataset is a memory-mapped array to avoid memory leaks.
    - If the device type is "cuda", the function uses pinned memory for faster data transfer
      to the GPU.

    Examples
    --------
    >>> dataset = np.arange(100, dtype=np.uint16)
    >>> batch_size = 4
    >>> context_length = 10
    >>> x, y = get_batch(dataset=dataset, batch_size=batch_size, context_length=context_length)
    >>> x.shape
    torch.Size([4, 10])
    >>> y.shape
    torch.Size([4, 10])
    """
    # Source: Karpathy, We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122

    # if not isinstance(dataset, np.memmap):
    #     raise ValueError("The dataset should be a memory-mapped array. Example: data = np.memmap('data.npy', dtype=np.uint16, mode='r')")

    device = torch.device("cuda") if device_type == "cuda" else torch.device("cpu")
    low, high = 0, len(dataset) - context_length
    size = (batch_size,)
    indices = torch.randint(low=low, high=high, size=size, generator=generator)

    x = torch.stack(
        [
            torch.from_numpy((dataset[index : index + context_length]).astype(np.int64))
            for index in indices
        ]
    )
    y = torch.stack(
        [
            torch.from_numpy(
                (dataset[index + 1 : index + 1 + context_length]).astype(np.int64)
            )
            for index in indices
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(
            device, non_blocking=True
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y
