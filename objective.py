from benchopt import BaseObjective, safe_import_context
from benchmark_utils.typing import Distribution, Tensor
from typing import Callable, List, Tuple

with safe_import_context() as import_ctx:
    import time
    import torch

    from benchmark_utils.metrics import negative_log_lik, c2st, emd, mmd


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
        # Set prior
        # The prior distribution is used by some algorithms to generate simulations or compute log probabilites as an objective.
        self.prior = prior

        # Normalize and set data
        mean_theta, std_theta = theta_train.mean(dim=0), theta_train.std(dim=0)
        mean_x, std_x = x_train.mean(dim=0), x_train.std(dim=0)

        # Data consists of samples from a joint distribution of the parameters thetas and simulated data x.
        self.theta_train = (theta_train - mean_theta) / std_theta
        self.x_train = (x_train - mean_x) / std_x
        self.theta_test = (theta_test - mean_theta) / std_theta
        self.x_test = (x_test - mean_x) / std_x

        # x_ref and theta_ref are pairs obtained from the reference posterior p(theta|x) when it is known, used by some metrics.
        if theta_ref is None:
            # The problem does not have a tractable posterior, we do not have access to samples from the reference posterior.
            self.x_ref = x_ref
            self.theta_ref = theta_ref
        else:
            # Normalize and set reference samples.
            self.x_ref = (x_ref - mean_x) / std_x
            self.theta_ref = [
                (theta - mean_theta) / std_theta for theta in theta_ref
            ]

    def compute(
        self,
        result: Tuple[
            Callable[[Tensor, Tensor], Tensor],
            Callable[[Tensor, int], Tensor],
        ],
    ):
        # Computes negative log likelihood (NLL) on test and train data
        # NLL can always be computed and is used as the "value" to stop the run
        log_prob, sample = result
        nll_test = negative_log_lik(log_prob, self.theta_test, self.x_test)
        nll_train = negative_log_lik(log_prob, self.theta_train, self.x_train)

        if self.theta_ref is None:
            # Samples from the reference posterior are not available, these metrics cannot be computed.
            c2st_mean, c2st_std = None, None
            emd_mean, emd_std = None, None
            mmd_mean, mmd_std = None, None
            sampling_time = None
        else:
            # Samples from the reference posterior are available.
            # Starting inference about parameters theta
            start = time.perf_counter()
            n_ref = (theta.shape[0] for theta in self.theta_ref)
            theta_est = [sample(x, n) for x, n in zip(self.x_ref, n_ref)]
            end = time.perf_counter()

            # Computing metrics.
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
        # Information to pass to the solvers are samples from the joint distribution of the parameters thetas and simulated data x and the prior over parameters.
        return dict(theta=self.theta_train, x=self.x_train, prior=self.prior)
