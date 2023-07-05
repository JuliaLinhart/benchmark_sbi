from benchopt import BaseObjective, safe_import_context
from benchmark_utils.typing import Distribution, Tensor
from typing import Callable, List, Tuple

with safe_import_context() as import_ctx:
    import torch

    from benchmark_utils.common import negative_log_lik, c2st, emd


class Objective(BaseObjective):
    """Benchmarks amortized simulation-based inference algorithms."""

    name = "sbi"
    parameters = {}
    min_benchopt_version = "1.3"

    install_cmd = "conda"
    requirements = [
        "torch",
        "pip:sbibm",
        "pip:POT",
    ]

    def set_data(
        self,
        prior: Distribution,
        theta_train: Tensor,
        x_train: Tensor,
        theta_test: Tensor,
        x_test: Tensor,
        theta_ref: List[Tensor] = None,
        x_ref: Tensor = None,
    ):
        self.prior = prior
        self.theta_train = theta_train
        self.x_train = x_train
        self.theta_test = theta_test
        self.x_test = x_test
        self.theta_ref = theta_ref
        self.x_ref = x_ref

    def compute(
        self,
        result: Tuple[
            Callable[[Tensor, Tensor], Tensor],
            Callable[[Tensor, int], Tensor],
        ],
    ):
        log_prob, sample = result
        nll_test = negative_log_lik(log_prob, self.theta_test, self.x_test)
        nll_train = negative_log_lik(log_prob, self.theta_train, self.x_train)

        if self.theta_ref is None:
            c2st_mean, c2st_std = None, None
            emd_mean, emd_std = None, None
        else:
            theta_est = [
                sample(x, self.theta_ref[i].shape[0]) for i, x in enumerate(self.x_ref)
            ]

            c2st_mean, c2st_std = c2st(self.theta_ref, theta_est)
            emd_mean, emd_std = emd(self.theta_ref, theta_est)

        return dict(
            value=nll_test,
            nll_train=nll_train,
            c2st_mean=c2st_mean,
            c2st_std=c2st_std,
            emd_mean=emd_mean,
            emd_std=emd_std,
        )

    def get_one_solution(self):
        return (
            lambda theta, x: torch.zeros(theta.shape[0]),
            lambda x, n: torch.randn(n, self.theta_train.shape[-1]),
        )

    def get_objective(self):
        return dict(theta=self.theta_train, x=self.x_train, prior=self.prior)
