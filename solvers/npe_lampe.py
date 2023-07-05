from benchopt import BaseSolver, safe_import_context
from benchmark_utils.typing import Distribution, Tensor
from typing import Callable

with safe_import_context() as import_ctx:
    import lampe
    import torch
    import zuko


class Solver(BaseSolver):
    r"""Neural Posterior Estimator.

    Solver implemented using the LAMPE package.

    Trains a parametric conditional distribution to approximate the posterior distribution of parameters.
    The optimization problem is to minimize the Kullback-Leibler divergence between the distribution of the observations and the distribution of the flow.

    References:
        | Fast :math:`\espilon`-free Inference of Simulation Models with Bayesian Conditional Density Estimation (Papamakarios & Murray, 2016)
        | https://arxiv.org/abs/1605.06376

        | Automatic posterior transformation for likelihood-free inference (Greenberg, Nonnenmacher & Macke,. 2019)
        | https://arxiv.org/abs/1905.07488
    """

    name = "npe_lampe"
    stopping_strategy = "callback"
    parameters = {
        "flow": ["MAF", "NSF"],
        "transforms": [1, 3, 5],
    }

    install_cmd = "conda"
    requirements = [
        "pip:lampe",
        "pip:zuko",
    ]

    def get_next(self, n_iter: int) -> int:
        return int(max(n_iter + 10, n_iter * 1.5))

    def set_objective(self, theta: Tensor, x: Tensor, prior: Distribution):
        self.theta, self.x = theta, x

        build = zuko.flows.MAF if self.flow == "MAF" else zuko.flows.NSF

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
