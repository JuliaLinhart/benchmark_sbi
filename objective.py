from benchopt import BaseObjective, safe_import_context
from typing import *

with safe_import_context() as import_ctx:
    import torch
    from torch import Tensor
    from torch.distributions import Distribution


def negative_log_likelihood(
    log_prob: Callable[[Tensor, Tensor], Tensor],
    theta: Tensor,
    x: Tensor,
) -> float:
    return -log_prob(theta, x).mean().item()


class Objective(BaseObjective):
    name = "Negative log-likelihood"
    parameters = {
        'split': [0.8],
    }
    min_benchopt_version = "1.3"

    def set_data(self, theta: Tensor, x: Tensor, prior: Distribution):
        theta = torch.tensor(theta, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)

        size = len(theta)
        train_size = int(self.split * size)
        test_size = size - train_size

        self.theta_train, self.theta_test = torch.split(theta, (train_size, test_size))
        self.x_train, self.x_test = torch.split(x, (train_size, test_size))
        self.prior = prior

    def compute(
        self,
        result: Tuple[
            Callable[[Tensor, Tensor], Tensor],
            Callable[[Tensor, int], Tensor],
        ],
    ):
        log_prob, sample = result

        nll = negative_log_likelihood(log_prob, self.theta_test, self.x_test)

        return dict(value=nll)

    # def get_one_solution(self):
    #     pass

    def get_objective(self):
        return dict(theta=self.theta_train, x=self.x_train, prior=self.prior)
