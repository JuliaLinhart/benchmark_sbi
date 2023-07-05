from benchopt import BaseSolver, safe_import_context

from typing import Callable

with safe_import_context() as import_ctx:
    import lampe
    import torch


class Solver(BaseSolver):
    r"""Flow Matching Posterior Estimator.

    Solver implemented using the LAMPE package.
    Trains a regression network to approximate a vector field inducing a time-continuous normalizing flow between
    the posterior distribution and a standard Gaussian distribution.

    References:
        | Flow Matching for Generative Modeling (Lipman et al., 2023)
        | https://arxiv.org/abs/2210.02747

        | Flow Matching for Scalable Simulation-Based Inference (Dax et al., 2023)
        | https://arxiv.org/abs/2305.17161
    """

    name = "FMPE"
    stopping_strategy = "callback"
    parameters = {
        "layers": [3, 5],
        "freqs": [3, 5],  # Embedding frequencies
    }

    def set_objective(self, theta: Tensor, x: Tensor, prior: Distribution):
        self.theta, self.x = theta, x
        self.fmpe = lampe.inference.FMPE(
            theta.shape[-1],
            x.shape[-1],
            freqs=self.freqs,
            hidden_features=(64,) * self.layers,
            activation=torch.nn.ELU,
        )

        self.loss = lampe.inference.FMPELoss(self.fmpe)
        self.optimizer = torch.optim.Adam(self.fmpe.parameters(), lr=1e-3)

    def get_next(self, n_iter: int) -> int:
        return int(max(n_iter + 10, n_iter * 1.5))

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
            lambda theta, x: self.fmpe.flow(x).log_prob(theta),
            lambda x, n: self.fmpe.flow(x).sample((n,)),
        )
