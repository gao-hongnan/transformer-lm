from enum import Enum
from typing import Iterable

import torch
from torch import nn


class Reduction(Enum):
    NONE = "none"
    MEAN = "mean"
    SUM = "sum"


class Softmax:
    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim

    def __call__(self, z: torch.Tensor) -> torch.Tensor:
        max_z = torch.max(z, dim=self.dim, keepdim=True).values
        numerator = torch.exp(z - max_z)
        denominator = torch.sum(numerator, dim=self.dim, keepdim=True)
        g = numerator / denominator
        return g


def cross_entropy_loss(
    logits: torch.FloatTensor, targets: torch.LongTensor
) -> torch.FloatTensor:
    """Given a tensor of logits and a tensor of targets, compute the cross-entropy loss.

    Args:
        logits: torch.FloatTensor
            Tensor of logits from the model.
            Shape is (batch_size, seq_len, vocab_size).
        targets: torch.LongTensor
            Tensor of targets.
            Shape is (batch_size, seq_len).

    Returns:
        loss: torch.FloatTensor
            Scalar tensor representing the cross-entropy loss.
    """

    if len(logits.shape) == 3:
        logits = logits.view(-1, logits.size(-1))
        targets = targets.view(-1)

    assert logits.size(0) == targets.size(0)

    s_logits = logits - torch.max(logits, dim=1, keepdim=True)[0]
    sum_logits = torch.sum(torch.exp(s_logits), dim=1)
    sum_log_exp = torch.log(sum_logits)

    logits_true_class = torch.gather(
        s_logits, dim=1, index=targets.unsqueeze(1)
    ).squeeze(1)
    logits_true_class = logits_true_class.squeeze()

    loss_per_example = sum_log_exp - logits_true_class
    return torch.mean(loss_per_example)


class CrossEntropyLoss:
    def __init__(self, reduction: Reduction = Reduction.MEAN):
        """
        Initialize the CrossEntropyLoss.

        Parameters
        ----------
        reduction : Reduction, optional
            Specifies the reduction to apply to the output.
            Defaults to Reduction.MEAN.
        """
        self.reduction = reduction

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the cross entropy loss for a given set of predicted logits and targets.

        Parameters
        ----------
        logits : torch.Tensor
            Predicted logits tensor of shape (batch_size, ..., vocab_size)
            where ... represents an arbitrary number of dimensions. Note
            `vocab_size` is the last dimension but is used in context of transformer models.
        targets : torch.Tensor
            Target tensor of shape (batch_size, ...).

        Returns
        -------
        torch.Tensor
            The computed cross entropy loss, reduced according to the specified reduction strategy.
        """
        if logits.shape[:-1] != targets.shape:
            raise ValueError(
                f"Logits and targets must have compatible shapes. "
                f"Received logits shape: {logits.shape} and targets shape: {targets.shape}"
            )

        logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
        logits_stable = logits - logits_max

        log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1))
        target_logits = logits_stable.gather(
            dim=-1, index=targets.unsqueeze(-1)
        ).squeeze(-1)
        neg_log_likelihood = -target_logits + log_sum_exp

        if self.reduction == Reduction.NONE:
            return neg_log_likelihood
        elif self.reduction == Reduction.MEAN:
            return torch.mean(neg_log_likelihood)
        elif self.reduction == Reduction.SUM:
            return torch.sum(neg_log_likelihood)
        else:
            raise ValueError(f"Invalid reduction: {self.reduction}")


def gradient_clipping(
    parameters: Iterable[nn.Parameter], max_norm: float, epsilon: float = 1e-6
) -> None:
    """
    Clips the gradients of the provided parameters to have a maximum L2 norm of `max_norm`.

    Parameters
    ----------
    parameters : Iterable[nn.Parameter]
        An iterable of PyTorch parameters.
    max_norm : float
        The maximum L2 norm for gradient clipping.
    epsilon : float, optional
        A small value added for numerical stability. Defaults to 1e-6.
    """
    parameters_with_grad = [p for p in parameters if p.grad is not None]

    # l2 norm of the gradients
    grad_norms = torch.stack([torch.norm(p.grad, 2) for p in parameters_with_grad])
    total_norm = torch.norm(grad_norms, 2)

    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + epsilon)
        for p in parameters_with_grad:
            assert p.grad is not None  # appease mypy
            p.grad.mul_(clip_coef)