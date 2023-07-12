from benchopt import BaseSolver, safe_import_context
from benchmark_utils.typing import Distribution, Tensor
from typing import Callable

with safe_import_context() as import_ctx:
    import lampe
    import torch
    import zuko


class Solver(BaseSolver):
    r"""Neural posterior estimation (NPE) solver implemented with the
    :mod:`lampe` package.

    The solver trains a parametric conditional distribution :math:`q_\phi(\theta | x)`
    to approximate the posterior distribution :math:`p(\theta | x)` of parameters given
    observations.

    References:
        | Fast :math:`\espilon`-free Inference of Simulation Models with Bayesian Conditional Density Estimation (Papamakarios et al., 2016)
        | https://arxiv.org/abs/1605.06376

        | Automatic posterior transformation for likelihood-free inference (Greenberg et al., 2019)
        | https://arxiv.org/abs/1905.07488
    """  # noqa:E501

    name = "npe_lampe"
    stopping_strategy = "callback"
    parameters = {
        "flow": ["maf", "nsf"],
        "transforms": [1, 3, 5],
    }

    requirements = [
        "pip:lampe",
    ]

    @staticmethod
    def get_next(n_iter: int) -> int:
        return n_iter + 10

    def set_objective(self, theta: Tensor, x: Tensor, prior: Distribution):
        self.theta, self.x = theta, x

        build = zuko.flows.MAF if self.flow == "maf" else zuko.flows.NSF

        self.npe = lampe.inference.NPE(
            theta.shape[-1],
            x.shape[-1],
            build=build,
            transforms=self.transforms,
        )

        self.loss = lampe.inference.NPELoss(self.npe)
        self.optimizer = torch.optim.Adam(self.npe.parameters(), lr=1e-3)

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
            lambda theta, x: self.npe.flow(x).log_prob(theta),
            lambda x, n: self.npe.flow(x).sample((n,)),
        )
