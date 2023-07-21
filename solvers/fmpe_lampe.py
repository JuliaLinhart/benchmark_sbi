r"""Solver module for FMPE, :mod:`lampe` implementation.

References
----------
    [1] Flow Matching for Generative Modeling (Lipman et al., 2023)
        https://arxiv.org/abs/2210.02747
    [2] Flow Matching for Scalable Simulation-Based Inference (Dax et al., 2023)
        https://arxiv.org/abs/2305.17161
"""

from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion
from benchmark_utils.typing import Distribution, Tensor

from typing import Callable

with safe_import_context() as import_ctx:
    import lampe
    import torch


class Solver(BaseSolver):
    r"""Flow matching posterior estimator (FMPE) [1,2].

    The solver trains a regression network to approximate a vector field inducing a
    time-continuous normalizing flow between the posterior distribution and a standard
    Gaussian distribution.

    Implemented with the :mod:`lampe` package.
    """  # noqa:E501

    name = "fmpe_lampe"
    # training is stopped when the objective on the callback
    # does not decrease for over 10 iterations.
    stopping_criterion = SufficientProgressCriterion(
        patience=10, strategy="callback"
    )
    # parameters that can be called with `self.<>`,
    # all possible combinations are used in the benchmark.
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
        r"""Initialize the solver with the given `parameters`."""
        self.theta, self.x = theta, x
        self.fmpe = lampe.inference.FMPE(
            theta.shape[-1],
            x.shape[-1],
            hidden_features=(64,) * self.layers,
            activation=torch.nn.ELU,
        )

        self.loss = lampe.inference.FMPELoss(self.fmpe)
        self.optimizer = torch.optim.Adam(self.fmpe.parameters(), lr=1e-3)

    def run(self, cb: Callable):
        r"""Train the FMPE for one iteration."""
        dataset = lampe.data.JointDataset(
            self.theta,
            self.x,
            batch_size=128,
            shuffle=True,
        )

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
            lambda theta, x: self.fmpe.flow(x).log_prob(theta),
            lambda x, n: self.fmpe.flow(x).sample((n,)),
        )
