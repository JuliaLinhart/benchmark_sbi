from benchopt import BaseSolver, safe_import_context
from typing import *

with safe_import_context() as import_ctx:
    import lampe
    import numpy as np
    import torch
    import zuko

    from torch import Tensor


class Solver(BaseSolver):
    name = "NPE"
    stopping_strategy = "callback"
    parameters = {
        "flow": ["MAF", "NSF"],
        "transforms": [1, 3, 5],
    }

    def set_objective(self, theta: Tensor, x: Tensor):
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
        return lambda theta, x: self.npe.flow(x).log_prob(theta)
