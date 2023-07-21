from benchopt import BaseObjective, safe_import_context
from benchmark_utils.typing import Distribution, Tensor
from typing import Callable, List, Tuple

with safe_import_context() as import_ctx:
    import time
    import torch

    from benchmark_utils.metrics import negative_log_lik, c2st, emd, mmd
    from torch.distributions import AffineTransform, TransformedDistribution


class Objective(BaseObjective):
    """Benchmarks amortized simulation-based inference (SBI) algorithms."""

    name = "sbi"
    parameters = {}
    min_benchopt_version = "1.3"

    requirements = [
        "pytorch:pytorch",
        "pip:POT",
        "pip:sbibm",
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
        # Standardize data and prior
        theta_mean, theta_std = theta_train.mean(dim=0), theta_train.std(dim=0)
        x_mean, x_std = x_train.mean(dim=0), x_train.std(dim=0)

        transform_theta = AffineTransform(-theta_mean / theta_std, 1 / theta_std)
        transform_x = AffineTransform(-x_mean / x_std, 1 / x_std)

        self.theta_train = transform_theta(theta_train)
        self.x_train = transform_x(x_train)
        self.theta_test = transform_theta(theta_test)
        self.x_test = transform_x(x_test)

        if theta_ref is None:
            self.x_ref = None
            self.theta_ref = None
        else:
            self.theta_ref = [transform_theta(theta) for theta in theta_ref]
            self.x_ref = transform_x(x_ref)

        self.prior = TransformedDistribution(prior, transform_theta)

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
            mmd_mean, mmd_std = None, None
            sampling_time = None
        else:
            start = time.perf_counter()
            n_ref = (theta.shape[0] for theta in self.theta_ref)
            theta_est = [sample(x, n) for x, n in zip(self.x_ref, n_ref)]
            end = time.perf_counter()

            c2st_mean, c2st_std = c2st(self.theta_ref, theta_est)
            emd_mean, emd_std = emd(self.theta_ref, theta_est)
            mmd_mean, mmd_std = mmd(self.theta_ref, theta_est)
            sampling_time = end - start

        return dict(
            value=nll_test,
            nll_train=nll_train,
            c2st_mean=c2st_mean,
            c2st_std=c2st_std,
            emd_mean=emd_mean,
            emd_std=emd_std,
            mmd_mean=mmd_mean,
            mmd_std=mmd_std,
            sampling_time=sampling_time,
        )

    def get_one_solution(self):
        return (
            lambda theta, x: torch.zeros(theta.shape[0]),
            lambda x, n: torch.randn(n, self.theta_train.shape[-1]),
        )

    def get_objective(self):
        return dict(theta=self.theta_train, x=self.x_train, prior=self.prior)
