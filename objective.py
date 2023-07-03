from benchopt import BaseObjective, safe_import_context
from typing import *

with safe_import_context() as import_ctx:
    import numpy as np
    import torch

    from numpy.typing import ArrayLike
    from torch import Tensor


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

    def set_data(self, theta: ArrayLike, x: ArrayLike):
        theta = torch.tensor(theta, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.float32)

        size = len(theta)
        train_size = int(self.split * size)
        test_size = size - train_size

        self.theta_train, self.theta_test = torch.split(theta, (train_size, test_size))
        self.x_train, self.x_test = torch.split(x, (train_size, test_size))

    def compute(
        self,
        log_prob: Callable[[Tensor, Tensor], Tensor],
    ):
        nll = negative_log_likelihood(log_prob, self.theta_test, self.x_test)

        return dict(value=nll)

    def get_one_solution(self): # TODO: ask if we can output dict (or several outputs)
        return lambda theta, x: 0.0

    def get_objective(self):
        return dict(theta=self.theta_train, x=self.x_train)
