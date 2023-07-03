from benchopt import BaseDataset, safe_import_context
from typing import *

with safe_import_context() as import_ctx:
    import numpy as np


class Dataset(BaseDataset):
    name = "Two-Moons"
    parameters = {
        "size": [1024, 4096],
        "seed": [42],
    }

    def get_data(self) -> Dict:
        rng = np.random.RandomState(self.seed)

        theta = rng.uniform(low=-1, high=1, size=(self.size, 2))

        alpha = rng.uniform(low=-np.pi / 2, high=np.pi / 2, size=self.size)
        r = rng.normal(loc=0.1, scale=0.01, size=self.size)

        x = np.stack(
            (
                r * np.cos(alpha)
                + 0.25
                - np.abs(theta[:, 0] + theta[:, 1]) / np.sqrt(2),
                r * np.sin(alpha) + (theta[:, 1] - theta[:, 0]) / np.sqrt(2),
            ),
            axis=-1,
        )

        return dict(theta=theta, x=x)
