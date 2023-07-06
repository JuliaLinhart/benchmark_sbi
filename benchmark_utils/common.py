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
    r"""Expected negative posterior log-likelihood :math:`\log p(\theta | x)` of a
    given set of parameters :math:`\theta` conditionned on the observation :math:`x`.
    Expectation is taken over the joint distribution :math:`p(\theta, x)`:

            :math:`\mathbb{E}_{\theta,x}[ - \log p(\theta | x) ]`

    Args:
        log_prob: A function that computes :math:`\log p(\theta | x)`.
        theta: A batch of parameter-sets :math:`\theta`.
        x : A batch of corresponding observations :math:`x \sim p(x | \theta)`.

    Returns:
        Expected negative log-likelihood over the joint samples :math`(\theta, x)`.
    """  # noqa:E501

    return -log_prob(theta, x).mean().item()


def emd(
    theta_ref: List[Tensor],
    theta_est: List[Tensor],
) -> Tuple[float, float]:
    """Earth mover's distance (EMD) between the reference posterior :math`p(\theta | x_0)`
    and estimated posterior :math`q(\theta | x_0)` conditioned on the same observation :math:`x_0`.

    Computes the mean and standard deviation of the EMD scores over a list
    of posterior samples corresponding to different observations :math:`x_0`.

    Args:
        theta_ref: A list of reference posterior samples.
        theta_est: A list of estimated posterior samples.

    Returns:
        Mean and standard deviation of the EMD scores.
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
    """Classifier 2-Samples Test (C2ST) between the reference posterior :math`p(\theta | x_0)`
    and estimated posterior :math`q(\theta | x_0)` conditioned on the same observation :math:`x_0`.

    Computes the mean and standard deviation of the C2ST scores (mean classification accuracy
    over a n-fold cross-validation) over a list of posterior samples corresponding
    to different observations :math:`x_0`.

    Args:
        theta_ref: A list of reference posterior samples.
        theta_est: A list of estimated posterior samples.
        n_folds: The number of cross-validation folds.
            Defaults to `5` as in [1].
        z_score: Whether to normalize the data before training the classifier.
            Defaults to `True` as recommended in [1].

    Returns:
        Mean and standard deviation of the C2ST scores.

    References:
        [1] `Benchmarking Simulation-Based Inference <https://arxiv.org/abs/2101.04653>`
    """  # noqa:E501

    print()

    c2st_scores = [
        metrics.c2st(X=P, Y=Q, z_score=z_score, n_folds=n_folds).item()
        for P, Q in tqdm(
            zip(theta_ref, theta_est), desc="C2ST"
        )  # TODO: hide progress bar between runs (or n_iter)
    ]

    return np.mean(c2st_scores), np.std(c2st_scores)


def data_generator_sbibm(
    name: str,
    train_size: int,
    test_size: int,
    ref_size: int = 0,
    n_per_ref: int = 1024,
) -> Dict:
    r"""Generates training, test and reference sets for a task of the :mod:`sbibm`
    package.

    Args:
        name: A task name (e.g. `'two_moons'`).
        train_size: The number of samples in the training set.
        test_size: The number of samples in the test set.
        ref_size: The number of reference posteriors.
        n_per_ref: The number of samples per reference posteriors.

    Returns:
        A dictionary with training, test and reference sets as well as the parameters
        prior.
    """  # noqa:E501

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
