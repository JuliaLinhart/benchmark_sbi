from contextlib import contextmanager, redirect_stdout, redirect_stderr
from io import StringIO
from typing import *

import torch
import sbibm


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


def data_generator_sbibm(n: int, task_name: str, reference: bool = False) -> Dict:
    task = sbibm.get_task(task_name)
    prior = task.get_prior()
    simulator = task.get_simulator()

    theta = prior(num_samples=n)
    x = simulator(theta)

    if reference:
        sample_reference = lambda x, n: task._sample_reference_posterior(
            num_samples=n, observation=x, num_observation=1
        )
    else:
        sample_reference = None

    return dict(
        theta=theta,
        x=x,
        prior=task.prior_dist,
        sample_reference=sample_reference,
    )
