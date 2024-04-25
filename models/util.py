from typing import Optional

import numpy as np
import numpy.typing as npt
import torch


def load_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load a batch of data from the dataset."""
    inputs = np.zeros((batch_size, context_length))
    target_labels = np.zeros((batch_size, context_length))

    l = len(dataset) - context_length
    start_idx = torch.randint(l, (batch_size,), generator=generator)
    for row, idx in enumerate(start_idx):
        inputs[row] = dataset[idx : idx + context_length]
        target_labels[row] = dataset[idx + 1 : idx + context_length + 1]

    inputs = torch.tensor(inputs, dtype=torch.long, device=device)
    target_labels = torch.tensor(target_labels, dtype=torch.long, device=device)

    return inputs, target_labels
