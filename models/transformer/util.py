import math
from functools import partial

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import Optimizer


def _cosine_schedule_with_warmup_and_post_annealing_lr_lambda(
    iter: int,
    *,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """
    Calculate the learning rate using cosine annealing schedule with warmup.

    Parameters
    ----------
    iter : int
        The current training iteration.
    max_learning_rate : float
        The maximum learning rate (used at the end of the warmup).
    min_learning_rate : float
        The minimum (final) learning rate after cosine annealing.
    warmup_iters : int
        The number of iterations for the warmup phase.
    cosine_cycle_iters : int
        The total number of iterations for the cosine annealing cycle (including warmup).

    Returns
    -------
    float
        The calculated learning rate for the current training iteration.
    """
    if iter < warmup_iters:  # warmup phase
        return (iter / max(1, warmup_iters)) * max_learning_rate
    elif iter <= cosine_cycle_iters:  # cosine annealing phase
        progress = (iter - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(math.pi * progress)
        )
    else:  # post-annealing phase
        return min_learning_rate


# NOTE: see cs336
def get_cosine_annealing_with_warmup_and_post_annealing(
    optimizer: Optimizer,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
    last_epoch: int = -1,
    verbose: bool = False,
) -> LambdaLR:
    """
    Create a learning rate scheduler with warmup followed by cosine annealing.

    Parameters
    ----------
    optimizer : `torch.optim.Optimizer`
        The optimizer for which to schedule the learning rate.
    max_learning_rate : float
        The initial and maximum learning rate during the warmup.
    min_learning_rate : float
        The minimum learning rate after cosine annealing.
    warmup_iters : int
        Number of warmup iterations.
    cosine_cycle_iters : int
        Total number of iterations for the cosine annealing cycle (including warmup).
    last_epoch : int
        The index of the last epoch when resuming training.
    verbose : bool
        Print the learning rate at every update.

    Returns
    -------
    `torch.optim.lr_scheduler.LambdaLR`
        The scheduler with the appropriate schedule.
    """
    lr_lambda = partial(
        _cosine_schedule_with_warmup_and_post_annealing_lr_lambda,
        max_learning_rate=max_learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_iters=warmup_iters,
        cosine_cycle_iters=cosine_cycle_iters,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch, verbose=verbose)


# def clip_gradients(parameters, max_norm):
#     """
#     Clip the gradients of the parameters to the specified maximum l2-norm.

#     Args:
#         parameters: Iterable[torch.nn.Parameter]
#             An iterable of parameters whose gradients need to be clipped.
#         max_norm: float
#             The maximum norm for the gradients.

#     Returns:
#         None; the function modifies gradients in-place.
#     """
#     # Calculate the total norm of all parameters
#     total_norm = torch.sqrt(
#         sum(torch.sum(p.grad.data**2) for p in parameters if p.grad is not None)
#         + 1e-6
#     )

#     # Scale down gradients if the total norm exceeds the max_norm
#     if total_norm > max_norm:
#         scale = max_norm / total_norm
#         for p in parameters:
#             if p.grad is not None:
#                 p.grad.data.mul_(scale)
