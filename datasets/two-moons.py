from benchopt import BaseDataset, safe_import_context
from typing import *

with safe_import_context() as import_ctx:
    import torch
    import sbibm
    from benchmark_utils import fork


class Dataset(BaseDataset):
    name = "Two-Moons"
    parameters = {
        "size": [1024, 4096],
        "seed": [42],
    }

    def get_data(self) -> Dict:
        with fork():
            torch.manual_seed(self.seed)

            task = sbibm.get_task("two_moons")
            prior = task.get_prior()
            simulator = task.get_simulator()

            theta = prior(num_samples=self.size)
            x = simulator(theta)

            sample_reference = (
                lambda x, n, x_id: task._sample_reference_posterior(
                    num_samples=n, observation=x, num_observation=x_id
                )
            )

        return dict(
            theta=theta,
            x=x,
            prior=task.prior_dist,
            sample_reference=sample_reference,
        )

        