r"""Solver module for NPE, :mod:`lampe` implementation.

References
----------
    [1] Fast :math:`\espilon`-free Inference of Simulation Models with
        Bayesian Conditional Density Estimation (Papamakarios et al., 2016),
        https://arxiv.org/abs/1605.06376
    [2] Automatic posterior transformation for likelihood-free inference
        (Greenberg et al., 2019), https://arxiv.org/abs/1905.07488
"""

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchmark_utils.typing import Distribution, Tensor
from typing import Callable

with safe_import_context() as import_ctx:
    import lampe
    import torch
    import zuko


class Solver(BaseSolver):
    r"""Neural posterior estimation (NPE) [1,2].

    The solver trains a parametric conditional distribution :math:`q_\phi(\theta | x)`
    to approximate the posterior distribution :math:`p(\theta | x)` of parameters given
    observations.

    Implementated with the :mod:`lampe` package.
    """  # noqa:E501

    name = "npe_lampe"
    # training is stopped when the objective on the callback
    # does not decrease for over `patience=3` iterations
    stopping_criterion = SufficientProgressCriterion(
        patience=3, strategy="callback"
    )
    # parameters that can be called with `self.<>`,
    # all possible combinations are used in the benchmark
    parameters = {
        "flow": ["maf", "nsf"],
        "transforms": [1, 3, 5],
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
        r"""Set the data for the NPE."""
        self.theta, self.x = theta, x

    def run(self, cb: Callable):
        r"""Initialize and train the NPE."""
        # Initialize the NPE with given `parameters`
        build = zuko.flows.MAF if self.flow == "maf" else zuko.flows.NSF

        self.npe = lampe.inference.NPE(
            self.theta.shape[-1],
            self.x.shape[-1],
            build=build,
            transforms=self.transforms,
            hidden_features=(64, 64),
        )

        # Initialize the loss and optimizer
        self.loss = lampe.inference.NPELoss(self.npe)
        self.optimizer = torch.optim.Adam(self.npe.parameters(), lr=1e-3)

        # Define the training dataset
        dataset = lampe.data.JointDataset(
            self.theta,
            self.x,
            batch_size=128,
            shuffle=True,
        )

        # Train the NPE
        while cb(self.get_result()):  # cb is a callback function
            for theta, x in dataset:
                self.optimizer.zero_grad()
                loss = self.loss(theta, x)
                loss.backward()
                self.optimizer.step()

    def get_result(self):
        r"""Define the estimator's log-prob function and sampler.

        Returns the input of the `Objective.compute` method.
        """
        return (
            lambda theta, x: self.npe.flow(x).log_prob(theta),
            lambda x, n: self.npe.flow(x).sample((n,)),
        )
