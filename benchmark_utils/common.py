r""""""

import numpy as np
import sbibm
import sbibm.metrics as metrics
import ot
import torch

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
from torch import Tensor
from tqdm import tqdm
from typing import Callable, Dict, List, Tuple


@contextmanager
def fork():
    """Context manager which resets the torch random seed to its previous
    state when exiting.
    """

    try:
        state = torch.random.get_rng_state()
        yield
    finally:
        torch.set_rng_state(state)


@contextmanager
def dump():
    """Context manager which dumps the standard outputs (stdout) and
    standard errors (stderr).
    """

    with StringIO() as f:
        with redirect_stdout(f), redirect_stderr(f):
            try:
                yield
            finally:
                pass


def negative_log_lik(
    log_prob: Callable[[Tensor, Tensor], Tensor],
    theta: Tensor,
    x: Tensor,
) -> float:
    r"""Computes the the negative posterior log-density :math:`\log p(\theta | x)`
    of a given set of parameters :math:`\theta` conditionned on the observation
    :math:`x`.

    Args:
        log_prob: A function that computes :math:`p(\theta | x)`.
        theta: A set of parameters :math:`\theta`.
        x : An observation :math:`x`.

    Returns:
        The evaluated negative log posterior density of :math:`\theta` conditionned by an observation :math:`x`.
    """
    return -log_prob(theta, x).mean().item()


def emd(
    theta_ref: List[Tensor],
    theta_est: List[Tensor],
) -> Tuple[float, float]:
    """Computes the mean and standard deviation of the earth mover's distance (EMD)
    over reference posterior and estimated posterior samples.

    Args:
        theta_ref: A list of reference posterior samples.
        theta_est: A list of estimated posterior samples.

    Returns:
        Mean and standard deviation of the C2ST scores.
    """

    emd_scores = [
        ot.emd2(P.new_tensor(()), Q.new_tensor(()), torch.cdist(P, Q)).item()
        for P, Q in zip(theta_ref, theta_est)
    ]

    return np.mean(emd_scores), np.std(emd_scores)


def c2st(
    theta_ref: List[Tensor],
    theta_est: List[Tensor],
) -> Tuple[float, float]:
    """Computes the mean and standard deviation of the classifier 2-samples test (C2ST)
    scores over reference posterior and estimated posterior samples.

    Args:
        theta_ref: The reference posterior samples.
        theta_est: The estimated posterior samples.

    Returns:
        Mean and standard deviation of the C2ST scores.
    """

    print()

    c2st_scores = [
        metrics.c2st(X=P, Y=Q, z_score=True, n_folds=5).item()
        for P, Q in tqdm(zip(theta_ref, theta_est), desc="C2ST")
    ]

    return np.mean(c2st_scores), np.std(c2st_scores)


def data_generator_sbibm(
    name: str,
    train_size: int,
    test_size: int,
    ref_size: int = 0,
    n_per_ref: int = 1024,
) -> Dict:
    r"""Generates training, test and reference pairs of :math:`\theta`, :math:`x` and a prior over :math:`\theta` using sbibm

    Args:
        name: A task name, either "slcp" or "two_moons".
        train_size: Number of samples to be considered in the training set.
        test_size: Number of samples to be considered in the test set.
        ref_size: Number of observation of reference :math:`x`.
        n_per_ref: Number of :math:`\theta`sampled from the reference posterior distribution condionned on :math:`x`.

    Returns:
        Training, test and reference pairs of :math:`\theta`, :math:`x` and the prior over :math:`\theta`.
    """

    task = sbibm.get_task(name)
    prior = task.get_prior()
    simulator = task.get_simulator()

    theta_train = prior(num_samples=train_size)
    x_train = simulator(theta_train)

    theta_test = prior(num_samples=test_size)
    x_test = simulator(theta_test)

    if ref_size > 0:
        theta_ref = prior(num_samples=ref_size)
        x_ref = simulator(theta_ref)

        theta_ref = []

        for i in tqdm(range(ref_size), desc="Posterior sampling"):
            with dump():
                theta_ref.append(
                    task._sample_reference_posterior(
                        num_samples=n_per_ref,
                        observation=x_ref[i][None],
                        num_observation=None,
                    )
                )
    else:
        theta_ref = None
        x_ref = None

    return dict(
        prior=task.prior_dist,
        theta_train=theta_train,
        x_train=x_train,
        theta_test=theta_test,
        x_test=x_test,
        theta_ref=theta_ref,
        x_ref=x_ref,
    )
