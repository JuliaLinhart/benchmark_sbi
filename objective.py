from benchopt import BaseObjective, safe_import_context
from typing import *

with safe_import_context() as import_ctx:
    import numpy as np
    import torch
    from torch import Tensor
    from torch.distributions import Distribution
    import sbibm.metrics as metrics
    from benchmark_utils import dump


def negative_log_likelihood(
    log_prob: Callable[[Tensor, Tensor], Tensor],
    theta: Tensor,
    x: Tensor,
) -> float:
    return -log_prob(theta, x).mean().item()


def c2st(
    sample: Callable[[Tensor, int], Tensor],
    sample_reference: Callable[[Tensor, int], Tensor],
    x: Tensor,
    num_observations: int,
    n_samples: int = 1000,  # default from sbibm is 20000
) -> float:
    c2st_scores = []
    for i in range(num_observations):
        print(f"observation {i + 1}/{num_observations}")
        with dump():
            P = sample_reference(x[i][None, :], n_samples)
        Q = sample(x[i], n_samples)
        c2st_scores.append(metrics.c2st(X=P, Y=Q, z_score=True, n_folds=5))
    return np.mean(c2st_scores), np.std(c2st_scores)


class Objective(BaseObjective):
    name = "Negative log-likelihood"
    parameters = {
        "split": [0.8],
        "num_observations": [10],
    }
    min_benchopt_version = "1.3"

    def set_data(
        self,
        theta: Tensor,
        x: Tensor,
        prior: Distribution,
        sample_reference: Callable[[Tensor, int], Tensor],
    ):
        size = len(theta)
        train_size = int(self.split * size)
        test_size = size - train_size

        self.theta_train, self.theta_test = torch.split(theta, (train_size, test_size))
        self.x_train, self.x_test = torch.split(x, (train_size, test_size))
        self.prior = prior
        self.sample_reference = sample_reference

    def compute(
        self,
        result: Tuple[
            Callable[[Tensor, Tensor], Tensor],
            Callable[[Tensor, int], Tensor],
        ],
    ):
        log_prob, sample = result

        nll_test = negative_log_likelihood(log_prob, self.theta_test, self.x_test)
        nll_train = negative_log_likelihood(log_prob, self.theta_train, self.x_train)
        if self.sample_reference is None:
            c2st_mean, c2st_std = None, None
        else:
            c2st_mean, c2st_std = c2st(
                sample, self.sample_reference, self.x_test, self.num_observations
            )

        return dict(
            value=nll_test, nll_train=nll_train, c2st_mean=c2st_mean, c2st_std=c2st_std
        )

    # def get_one_solution(self):
    #     pass

    def get_objective(self):
        return dict(theta=self.theta_train, x=self.x_train, prior=self.prior)
