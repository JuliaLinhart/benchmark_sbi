r"""Objective module."""

from benchopt import BaseObjective, safe_import_context
from benchmark_utils.typing import Distribution, Tensor
from typing import Callable, List, Tuple

with safe_import_context() as import_ctx:
    import time
    import torch

    from benchmark_utils.metrics import negative_log_lik, c2st, emd, mmd
    from torch.distributions import AffineTransform, TransformedDistribution


class Objective(BaseObjective):
    r"""Benchmark amortized simulation-based inference (SBI) algorithms.

    Datasets:
        - train/test: parameter-observation pairs from the prior-simulator
            joint distribution :math:`p(\theta, x)=p(\theta)p(x|\theta)`.
        - reference (optional): observations math:`x_ref` and corresponding
            samples from the reference posterior :math:`p(\theta|x_ref)`
            (if available).

    Solvers: amortized SBI algorithms trained on the joint to approximate
        the posterior :math:`p(\theta|x)` for any observation :math:`x`.

    Metrics:
        - expected negative log likelihood (NLL) on test (stopping criterion)
            and train datasets.
        - C2ST, EMD, MMD on reference dataset (optional).
    """  # noqa: E501

    name = "sbi: maximum likelihood on test set"
    parameters = {}  # No parameters for this objective.
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
        r"""Set the data.

        Input parameters are the output of `Dataset.get_data`.

        Parameters
        ----------
        prior: Distribution
            prior over simulator parameters
            required by some solvers (NRE) to compute the `result` functions
        theta_train: Tensor
            parameters generated from the prior
            of shape (train_size, dim_theta)
        x_train: Tensor
            observations generated via the simulator for each parameter in `theta_train`
            of shape (train_size, dim_x)
        theta_test: Tensor
            parameters generated from the prior
            of shape (test_size, dim_theta)
        x_test: Tensor
            observations generated from the simulator for each parameter in `theta_test`
            of shape (test_size, dim_x)
        theta_ref: List[Tensor], optional
            reference posterior samples for every observation `x_ref`, by defaut None
            of shape [(n_per_ref, dim_theta)] * n_ref
        x_ref: Tensor, optional
            set of observations for which the reference posterior is known, by default None.
            of shape (n_ref, dim_x)
        """  # noqa: E501
        # Standardize data and prior
        mean_theta, std_theta = theta_train.mean(dim=0), theta_train.std(dim=0)
        mean_x, std_x = x_train.mean(dim=0), x_train.std(dim=0)
        
        # Define standardization transform
        t_theta = AffineTransform(-mean_theta / std_theta, 1 / std_theta)
        t_x = AffineTransform(-mean_x / std_x, 1 / std_x)
        
        # Standardize data
        self.theta_train = t_theta(theta_train)
        self.x_train = t_x(x_train)
        self.theta_test = t_theta(theta_test)
        self.x_test = t_x(x_test)

        if theta_ref is None:
            self.x_ref = None
            self.theta_ref = None
        else:
            self.theta_ref = [t_theta(theta) for theta in theta_ref]
            self.x_ref = t_x(x_ref)
        
        # Standardize prior
        self.prior = TransformedDistribution(prior, t_theta)


    def compute(
        self,
        result: Tuple[
            Callable[[Tensor, Tensor], Tensor],
            Callable[[Tensor, int], Tensor],
        ],
    ):
        """Compute the metrics.

        Input parameters are the output of `Solver.get_result`.

        Parameters
        ----------
        result : Tuple[ Callable[[Tensor, Tensor], Tensor], Callable[[Tensor, int], Tensor], ]
            - log_prob: computes the log probabilities of the approximate posterior
                for a given batch of observations and parameters.
                Returns a tensor of shape (batch_size,).
            - sample: samples from the approximate posterior for a given observation
                and sample size. Returns a tensor `theta_est` of shape

        Returns
        -------
        Dict
            dictionary of computed metrics
                - value: metric used as stopping criterion (NLL on test data)
                - any other metric computed ...

        """  # noqa:E501
        # Get result: `log_prob`` and `sample` functions from the solver.
        log_prob, sample = result

        # Compute NLL on train and test data.
        # Always computed. Do not require samples from the reference posterior.
        nll_test = negative_log_lik(log_prob, self.theta_test, self.x_test)
        nll_train = negative_log_lik(log_prob, self.theta_train, self.x_train)

        # Compute metrics on reference data if available.
        if self.theta_ref is None:
            c2st_mean, c2st_std = None, None
            emd_mean, emd_std = None, None
            mmd_mean, mmd_std = None, None
            sampling_time = None
        else:
            # Sampling from the approximate posterior.
            start = time.perf_counter()
            # same sample size as for the reference posterior
            n_ref = (theta.shape[0] for theta in self.theta_ref)
            # same conditioning observation as for the reference posterior
            theta_est = [sample(x, n) for x, n in zip(self.x_ref, n_ref)]
            end = time.perf_counter()

            # Compute metrics.
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
        r"""Return the same type of output as `Solver.get_result`.

        For testing purposes only.
        """  # noqa:E501
        return (
            lambda theta, x: torch.zeros(theta.shape[0]),
            lambda x, n: torch.randn(n, self.theta_train.shape[-1]),
        )

    def get_objective(self):
        r"""Information to pass to the solvers.

        This information is used by the solvers to compute the result
        (`log_prob` and `sample` functions of the trained SBI algorithm).

        Returns
        -------
        Dict
            contains training data and prior required by some solvers (NRE)
            to compute the result: `log_prob` and `sample` functions.
        """  # noqa:E501
        return dict(theta=self.theta_train, x=self.x_train, prior=self.prior)
