r"""Solver module for NRE, :mod:`lampe` implementation.

References
----------
    [1] Approximating Likelihood Ratios with Calibrated Discriminative Classifiers
        (Cranmer et al., 2015), https://arxiv.org/abs/1506.02169
    [2] Likelihood-free MCMC with Amortized Approximate Ratio Estimators
        (Hermans et al., 2019), https://arxiv.org/abs/1903.04057
"""  # noqa:E501

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchmark_utils.typing import Distribution, Tensor
from typing import Callable

with safe_import_context() as import_ctx:
    import lampe
    import torch


class Solver(BaseSolver):
    r"""Neural ratio estimation (NRE).

    The solver trains a classifier to discriminate between pairs sampled from the joint
    distribution :math:`p(\theta, x)` and the product of marginals :math:`p(\theta)
    p(x)`.

    Implementated with the :mod:`lampe` package.
    """  # noqa:E501

    name = "nre_lampe"
    # training is stopped when the objective on the callback
    # does not decrease for over `patience=3` iterations
    stopping_criterion = SufficientProgressCriterion(
        patience=3, strategy="callback"
    )
    # parameters that can be called with `self.<>`,
    # all possible combinations are used in the benchmark
    parameters = {
        "layers": [3, 5],
    }

    requirements = [
        "pip:lampe",
    ]

    @staticmethod
    def get_next(n_iter: int) -> int:
        r"""Evaluate the result every 10 epochs.

        Evaluating metrics (such as C2ST) at each epoch is time consuming
        and comes with noisy validation curves (1 iteration = 10 epochs).
        """
        return n_iter + 10

    def set_objective(self, theta: Tensor, x: Tensor, prior: Distribution):
        r"""Set the data and prior for the NRE."""
        self.theta, self.x, self.prior = theta, x, prior

    def run(self, cb: Callable):
        r"""Initialize and train the NRE."""
        # Initialize the NRE with given `parameters`
        self.nre = lampe.inference.NRE(
            theta.shape[-1],
            x.shape[-1],
            hidden_features=(64,) * self.layers,
        )

        # Initialize the loss and optimizer
        self.loss = lampe.inference.NRELoss(self.nre)
        self.optimizer = torch.optim.Adam(self.nre.parameters(), lr=1e-3)

        # Define the training dataset
        dataset = lampe.data.JointDataset(
            self.theta,
            self.x,
            batch_size=128,
            shuffle=True,
        )

        # Train the NRE
        while cb(self.get_result()):  # cb is a callback function
            for theta, x in dataset:
                self.optimizer.zero_grad()
                loss = self.loss(theta, x)
                loss.backward()
                self.optimizer.step()

    def get_result(self):
        r"""Define the estimator's log-prob function and sampler.

        Requires the prior to be set in `Objevtive.set_data`.
        Returns the input of the `Objective.compute` method.
        """
        return (
            lambda theta, x: self.nre(theta, x) + self.prior.log_prob(theta),
            lambda x, n: self.sample(x, n),
        )

    def sample(self, x: Tensor, n: int) -> Tensor:
        r"""Sampler for the NRE estimator.

        Parameters
        ----------
        x : Tensor
            conditionning observation.
        n : int
            number of samples to generate.

        Returns
        -------
        Tensor
            samples from the estimated posterior at given observation.
        """  # noqa:E501
        theta_0 = self.prior.sample((n,))

        # log q(theta | x): log probability of the estimated posterior
        def log_q(theta):
            return self.nre(theta, x) + self.prior.log_prob(theta)

        # sample using MCMC
        sampler = lampe.inference.MetropolisHastings(theta_0, log_f=log_q)
        samples = next(sampler(1024 + 1, burn=1024))  # TODO: add to parameters

        return samples
