r""""""

import numpy as np
import sbibm
import sbibm.metrics as metrics
import torch

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
from torch import Tensor
from tqdm import tqdm
from typing import Callable, Dict, List, Tuple


@contextmanager
def fork():
    try:
        state = torch.random.get_rng_state()
        yield
    finally:
        torch.set_rng_state(state)


@contextmanager
def dump():
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
    return -log_prob(theta, x).mean().item()


def c2st(
    theta_ref: List[Tensor],
    theta_est: List[Tensor],
) -> Tuple[float, float]:
    r""""""

    print()

    c2st_scores = [
        metrics.c2st(X=P, Y=Q, z_score=True, n_folds=5)
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
    r""""""

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
