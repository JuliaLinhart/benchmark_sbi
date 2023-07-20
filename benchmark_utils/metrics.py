r"""Objective and metrics."""

import numpy as np
import ot
import pyro
import sbibm.metrics as metrics
import torch

from torch import Tensor
from tqdm import tqdm
from typing import Callable, List, Tuple


pyro.distributions.enable_validation(False)


def negative_log_lik(
    log_prob: Callable[[Tensor, Tensor], Tensor],
    theta: Tensor,
    x: Tensor,
) -> float:
    r"""Expected negative log-likelihood :math:`\log p(\theta | x)` of a given set of
    parameters :math:`\theta` conditionned on the observation :math:`x`. Expectation is
    taken over the joint distribution :math:`p(\theta, x)`:

    .. math:: \mathbb{E}_{\theta,x} [ -\log p(\theta | x) ]

    Args:
    -----
        log_prob: A function that computes :math:`\log p(\theta | x)`.
        theta: A batch of parameter-sets :math:`\theta`.
        x : A batch of corresponding observations :math:`x \sim p(x | \theta)`.

    Returns:
    --------
        Expected negative log-likelihood over the joint samples :math:`(\theta, x)`.
    """  # noqa:E501

    return -log_prob(theta, x).mean().item()


def emd(
    theta_ref: List[Tensor],
    theta_est: List[Tensor],
) -> Tuple[float, float]:
    r"""Earth mover's distance (EMD) between reference posteriors :math:`p(\theta | x_i)`
    and estimated posteriors :math:`q(\theta | x_i)` conditioned on observations
    :math:`x_i`.

    Args:
        theta_ref: A list of reference posterior samples.
        theta_est: A list of estimated posterior samples.

    Returns:
        The mean and standard deviation of the EMD.
    """  # noqa:E501

    emd_scores = [
        ot.emd2(P.new_tensor(()), Q.new_tensor(()), torch.cdist(P, Q)).item()
        for P, Q in zip(theta_ref, theta_est)
    ]

    return np.mean(emd_scores), np.std(emd_scores)


def c2st(
    theta_ref: List[Tensor],
    theta_est: List[Tensor],
    n_folds: int = 5,
    z_score: bool = True,
) -> Tuple[float, float]:
    r"""Classifier 2-samples test (C2ST) between reference posteriors :math:`p(\theta | x_i)`
    and estimated posteriors :math:`q(\theta | x_i)` conditioned on observations
    :math:`x_i`.

    Implementation taken from Lueckmann et al. (2021).

    Args:
        theta_ref: A list of reference posterior samples.
        theta_est: A list of estimated posterior samples.
        n_folds: The number of cross-validation folds.
        z_score: Whether to normalize the data before training the classifier.

    Returns:
        The mean and standard deviation of the C2ST scores

    References:
        | Benchmarking Simulation-Based Inference (Lueckmann et al., 2021)
        | https://arxiv.org/abs/2101.04653
    """  # noqa:E501

    print()

    c2st_scores = [
        metrics.c2st(X=P, Y=Q, z_score=z_score, n_folds=n_folds).item()
        for P, Q in tqdm(
            zip(theta_ref, theta_est), desc="C2ST"
        )  # TODO: hide progress bar between runs (or n_iter)
    ]

    return np.mean(c2st_scores), np.std(c2st_scores)


def mmd(
    theta_ref: List[Tensor],
    theta_est: List[Tensor],
    z_score: bool = False,
) -> Tuple[float, float]:
    r"""Maximum mean discrepancy (MMD) between reference posteriors :math:`p(\theta | x_i)`
    and estimated posteriors :math:`q(\theta | x_i)` conditioned on observations
    :math:`x_i`.

    Implementation taken from Lueckmann et al. (2021).

    Args:
        theta_ref: A list of reference posterior samples.
        theta_est: A list of estimated posterior samples.
        z_score: Whether to normalize the data before computing the MMD.

    Returns:
        The mean and standard deviation of the MMD.

    References:
        | Benchmarking Simulation-Based Inference (Lueckmann et al., 2021)
        | https://arxiv.org/abs/2101.04653
    """  # noqa:E501

    mmd_scores = [
        metrics.mmd(X=P, Y=Q, z_score=z_score).item()
        for P, Q in zip(theta_ref, theta_est)
    ]

    return np.mean(mmd_scores), np.std(mmd_scores)
