r"""Objective and metrics.

References
----------
    [1] Benchmarking Simulation-Based Inference (Lueckmann et al. (2021))
        https://arxiv.org/abs/2101.04653
"""

import numpy as np
import ot
import pyro
import sbibm
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
    r"""Compute the expected negative log-likelihood (NLL) of an estimator q.

    .. math:: \mathbb{E}_{p(\theta,x)} [ -\log q(\theta | x) ]

    This quantity is approximated via Monte Carlo by taking the average over
    i.i.d. samples from the true joint distribution :math:`p(\theta, x)`.

    Parameters
    ----------
    log_prob : Callable[[Tensor, Tensor], Tensor]
        A function that computes :math:`\log q(\theta | x)`.
    theta : Tensor
        A batch of parameter sets :math:`\theta`.
    x : Tensor
        A batch of corresponding observations :math:`x \sim p(x | \theta)`.

    Returns
    -------
    float
        Expected negative log-likelihood over the joint samples :math:`(\theta, x)`.
    """  # noqa:E501
    return -log_prob(theta, x).mean().item()


def emd(
    theta_ref: List[Tensor],
    theta_est: List[Tensor],
) -> Tuple[float, float]:
    r"""Compute Earth mover's distance (EMD) between reference (p) and estimator (q).

    Mean and std over different conditionning observations :math:`x_ref` for which
    samples :mod:`theta_ref` from :math:`p(\theta | x_ref)` and :mod:`theta_est`
    from :math:`q(\theta | x_ref)` are available.

    Parameters
    ----------
    theta_ref : List[Tensor]
        A list of reference posterior samples.
    theta_est : List[Tensor]
        A list of estimated posterior samples.

    Returns
    -------
    Tuple[float, float]
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
) -> Tuple[float, float]:
    r"""Compute Classifier 2-Samples Test (C2ST) between reference (p) and estimator (q).

    Mean and std over different conditionning observations :math:`x_ref` for which
    samples :mod:`theta_ref` from :math:`p(\theta | x_ref)` and :mod:`theta_est`
    from :math:`q(\theta | x_ref)` are available.

    C2ST-scores are computed using a MLP-classifier with 5-fold cross-validation.
    Implementation taken from Lueckmann et al. (2021) [1].

    Parameters
    ----------
    theta_ref : List[Tensor]
        A list of reference posterior samples.
    theta_est : List[Tensor]
        A list of estimated posterior samples.
    n_folds : int, optional
        The number of cross-validation folds, by default 5.

    Returns
    -------
    Tuple[float, float]
         The mean and standard deviation of the C2ST scores
    """  # noqa:E501
    print()

    c2st_scores = [
        sbibm.metrics.c2st(
            X=P,
            Y=Q,
            z_score=False,  # no z_score: data already normalized (Objective.set_data)
            n_folds=n_folds,
        ).item()
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
    r"""Compute Maximum Mean Discrepancy (MMD) between reference (p) and estimator (q).

    Mean and std over different conditionning observations :math:`x_ref` for which
    samples :mod:`theta_ref` from :math:`p(\theta | x_ref)` and :mod:`theta_est`
    from :math:`q(\theta | x_ref)` are available.

    Implementation taken from Lueckmann et al. (2021) [1].

    Parameters
    ----------
        theta_ref: List[Tensor]
            A list of reference posterior samples.
        theta_est: List[Tensor]
            A list of estimated posterior samples.
        z_score: bool, optional
            Whether to normalize the data before computing the MMD.

    Returns
    -------
    Tuple[float, float]
        The mean and standard deviation of the MMD.
    """  # noqa:E501
    mmd_scores = [
        sbibm.metrics.mmd(X=P, Y=Q, z_score=z_score).item()
        for P, Q in zip(theta_ref, theta_est)
    ]

    return np.mean(mmd_scores), np.std(mmd_scores)
