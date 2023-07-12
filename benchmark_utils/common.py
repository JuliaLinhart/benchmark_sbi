r"""Common helpers."""

import pyro
import sbibm
import torch

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
from tqdm import tqdm
from typing import Dict


pyro.distributions.enable_validation(False)


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
