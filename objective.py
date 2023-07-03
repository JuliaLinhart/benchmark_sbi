from benchopt import BaseObjective, safe_import_context
from typing import *

with safe_import_context() as import_ctx:
    import numpy as np
    import torch
    from torch import Tensor
    from torch.distributions import Distribution
    import sbibm.metrics as metrics


def negative_log_likelihood(
    log_prob: Callable[[Tensor, Tensor], Tensor],
    theta: Tensor,
    x: Tensor,
) -> float:
    return -log_prob(theta, x).mean().item()


def c2st(
    sample: Callable[[Tensor, int], Tensor],
    sample_reference: Callable[[Tensor, int, int], Tensor],
    x: Tensor,
    n_samples: int = 1000,
) -> float:
    c2st_scores = []
    for x_id in range(3):
        P = sample_reference(x[x_id][None, :], n_samples, x_id)
        Q = sample(x[x_id][None, :], n_samples)

        c2st_scores.append(metrics.c2st(X=P, Y=Q, z_score=True))
        print(f"C2ST for x_id={x_id}: {c2st_scores[-1].item()}")
    return np.mean(c2st_scores), np.std(c2st_scores)


class Objective(BaseObjective):
    name = "Negative log-likelihood"
    parameters = {
        "split": [0.8],
    }
    min_benchopt_version = "1.3"

    def set_data(
        self,
        theta: Tensor,
        x: Tensor,
        prior: Distribution,
        sample_reference: Callable[[Tensor, int, int], Tensor],
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

        nll = negative_log_likelihood(log_prob, self.theta_test, self.x_test)

        c2st_mean, c2st_std = c2st(sample, self.sample_reference, self.x_test)

        return dict(value=nll, c2st_mean=c2st_mean, c2st_std=c2st_std)

    # def get_one_solution(self):
    #     pass

    def get_objective(self):
        return dict(theta=self.theta_train, x=self.x_train, prior=self.prior)
