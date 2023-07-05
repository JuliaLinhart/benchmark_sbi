from benchopt import BaseSolver, safe_import_context
from benchmark_utils.typing import Distribution, Tensor
from typing import Callable

with safe_import_context() as import_ctx:
    import lampe
    import torch


class Solver(BaseSolver):
    r"""Neural ratio estimation (NRE) solver implemented with the
    :mod:`lampe` package.

    The solver trains a classifier to discriminate between pairs sampled from the joint
    distribution :math:`p(\theta, x)` and the product of marginals :math:`p(\theta)
    p(x)`.

    References:
        | Approximating Likelihood Ratios with Calibrated Discriminative Classifiers (Cranmer et al., 2015)
        | https://arxiv.org/abs/1506.02169

        | Likelihood-free MCMC with Amortized Approximate Ratio Estimators (Hermans et al., 2019)
        | https://arxiv.org/abs/1903.04057
    """  # noqa:E501

    name = "nre_lampe"
    stopping_strategy = "callback"
    parameters = {
        "layers": [3, 5],
    }

    requirements = [
        "pip:lampe",
    ]

    def get_next(self, n_iter: int) -> int:
        return int(max(n_iter + 10, n_iter * 1.5))

    def set_objective(self, theta: Tensor, x: Tensor, prior: Distribution):
        self.theta, self.x, self.prior = theta, x, prior

        self.nre = lampe.inference.NRE(
            theta.shape[-1],
            x.shape[-1],
            hidden_features=(64,) * self.layers,
        )

        self.loss = lampe.inference.NRELoss(self.nre)
        self.optimizer = torch.optim.Adam(self.nre.parameters(), lr=1e-3)

    def run(self, cb: Callable):
        dataset = lampe.data.JointDataset(
            self.theta,
            self.x,
            batch_size=128,
            shuffle=True,
        )

        while cb(self.get_result()):
            for theta, x in dataset:
                self.optimizer.zero_grad()
                loss = self.loss(theta, x)
                loss.backward()
                self.optimizer.step()

    def get_result(self):
        return (
            lambda theta, x: self.nre(theta, x) + self.prior.log_prob(theta),
            lambda x, n: self.sample(x, n),
        )

    def sample(self, x: Tensor, n: int) -> Tensor:
        theta_0 = self.prior.sample((n,))

        def log_p(theta):
            return self.nre(theta, x) + self.prior.log_prob(theta)

        sampler = lampe.inference.MetropolisHastings(theta_0, log_f=log_p)
        samples = next(sampler(1024 + 1, burn=1024))  # TODO mettre en params

        return samples
