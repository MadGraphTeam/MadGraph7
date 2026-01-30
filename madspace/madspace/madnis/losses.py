"""Implementation of multi-channel loss functions"""

from collections.abc import Callable
from functools import wraps

import torch


def dtype_epsilon(tensor: torch.Tensor) -> float:
    return torch.finfo(tensor.dtype).eps


SingleChannelLoss = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
MultiChannelLoss = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None], torch.Tensor
]


def multi_channel_loss(loss: SingleChannelLoss) -> MultiChannelLoss:
    """
    Turns a single-channel loss function into a multi-channel loss function by evaluating it for
    each channel separately and then adding them weighted by TODO weighted by what?

    Args:
        loss: single-channel loss function, that expects the integrand value, test probability and
            sampling probability as arguments
    Returns:
        multi-channel loss function, that expects the integrand value, test probability and,
        optionally, sampling probability and channel indices as arguments.
    """

    # TODO: this unfortunately does not yield the correct signature (with the extra channels argument),
    # so it does not show up in the documentation
    @wraps(loss)
    def wrapped_multi(
        f_true: torch.Tensor,
        q_test: torch.Tensor,
        q_sample: torch.Tensor | None = None,
        channels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if q_sample is None:
            q_sample = q_test
        if channels is None:
            return loss(f_true, q_test, q_sample)

        loss_tot = 0
        for channel in channels.unique():
            mask = channels == channel
            fi, qti, qsi = f_true[mask], q_test[mask], q_sample[mask]
            ni = mask.count_nonzero()
            # loss_tot += ni / q_sample.shape[0] * loss(fi, qti, qsi) if ni > 0 else 0.0
            loss_tot += loss(fi, qti, qsi) if ni > 0 else 0.0
        return loss_tot

    return wrapped_multi


def stratified_variance(
    f_true: torch.Tensor,
    q_test: torch.Tensor,
    q_sample: torch.Tensor | None = None,
    channels: torch.Tensor | None = None,
):
    """
    Computes the stratified variance as introduced in [2311.01548] for two given sets of
    probabilities, ``f_true`` and ``q_test``. It uses importance sampling with a sampling
    probability specified by ``q_sample``.

    Args:
        f_true: normalized integrand values
        q_test: estimated function/probability
        q_sample: sampling probability
        channels: channel indices or None in the single-channel case
    Returns:
        computed stratified variance
    """
    if q_sample is None:
        q_sample = q_test
    if channels is None:
        abs_integral = torch.mean(f_true.detach().abs() / q_sample)
        return _variance(f_true, q_test, q_sample) / abs_integral.square()

    stddev_sum = 0
    abs_integral = 0
    for i in channels.unique():
        mask = channels == i
        fi, qti, qsi = f_true[mask], q_test[mask], q_sample[mask]
        stddev_sum += torch.sqrt(_variance(fi, qti, qsi) + dtype_epsilon(f_true))
        abs_integral += torch.mean(fi.detach().abs() / qsi)
    return (stddev_sum / abs_integral) ** 2

    # variances = []
    # abs_integrals = []
    # for i in channels.unique():
    #    mask = channels == i
    #    fi, qti, qsi = f_true[mask], q_test[mask], q_sample[mask]
    #    variances.append(_variance(fi, qti, qsi) + dtype_epsilon(f_true))
    #    abs_integrals.append(torch.mean(fi.abs() / qsi))
    # abs_integral_tot = sum(abs_integrals)
    # return sum(
    #    abs_integral / abs_integral_tot * variance
    #    for abs_integral, variance in zip(abs_integrals, variances)
    # ) / abs_integral_tot.detach().square()


@multi_channel_loss
def variance(
    f_true: torch.Tensor, q_test: torch.Tensor, q_sample: torch.Tensor
) -> torch.Tensor:
    abs_integral = torch.mean(f_true.detach().abs() / q_sample) + dtype_epsilon(f_true)
    return _variance(f_true, q_test, q_sample) / abs_integral.square()


def _variance(
    f_true: torch.Tensor,
    q_test: torch.Tensor,
    q_sample: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the variance for two given sets of probabilities, ``f_true`` and ``q_test``. It uses
    importance sampling with a sampling probability specified by ``q_sample``.

    Args:
        f_true: normalized integrand values
        q_test: estimated function/probability
        q_sample: sampling probability
    Returns:
        computed variance
    """
    ratio = q_test / q_sample
    mean = torch.mean(f_true / q_sample)
    sq = (f_true / q_test - mean) ** 2
    return (
        torch.mean(sq * ratio)
        if len(f_true) > 0
        else torch.tensor(0.0, device=f_true.device, dtype=f_true.dtype)
    )


@multi_channel_loss
def kl_divergence(
    f_true: torch.Tensor, q_test: torch.Tensor, q_sample: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Kullback-Leibler divergence for two given sets of probabilities, ``f_true`` and
    ``q_test``. It uses importance sampling, i.e. the estimator is divided by an additional factor
    of ``q_sample``.

    Args:
        f_true: normalized integrand values
        q_test: estimated function/probability
        q_sample: sampling probability
        channels: channel indices or None in the single-channel case
    Returns:
        computed KL divergence
    """
    f_true = f_true.detach().abs()
    f_true /= torch.mean(f_true / q_sample)
    log_q = torch.log(q_test)
    log_f = torch.log(f_true + dtype_epsilon(f_true))
    return torch.mean(f_true / q_sample * (log_f - log_q))


@multi_channel_loss
def rkl_divergence(
    f_true: torch.Tensor, q_test: torch.Tensor, q_sample: torch.Tensor
) -> torch.Tensor:
    """
    Computes the reverse Kullback-Leibler divergence for two given sets of probabilities, ``f_true``
    and ``q_test``. It uses importance sampling, i.e. the estimator is divided by an additional
    factor of ``q_sample``.

    Args:
        f_true: normalized integrand values
        q_test: estimated function/probability
        q_sample: sampling probability
        channels: channel indices or None in the single-channel case
    Returns:
        computed KL divergence
    """
    f_true = f_true.detach().abs()
    f_true /= torch.mean(f_true / q_sample)
    ratio = q_test / q_sample
    log_q = torch.log(q_test)
    log_f = torch.log(f_true + dtype_epsilon(f_true))
    return torch.mean(ratio * (log_q - log_f))
