from benchopt import BaseDataset, safe_import_context
from typing import Dict

with safe_import_context() as import_ctx:
    import torch


class Dataset(BaseDataset):
    """Dummy dataset. Required for testing purposes."""

    name = "simulated"
    parameters = {
        "train_size": [1024],
        "test_size": [256],
        "seed": [42],
    }

    def get_data(self) -> Dict:
        """Returns the input of the `Objective.set_data` method."""

        return dict(
            prior=torch.distributions.MultivariateNormal(
                torch.zeros(2),
                torch.eye(2),
            ),
            theta_train=torch.randn(self.train_size, 2),
            x_train=torch.randn(self.train_size, 3),
            theta_test=torch.randn(self.test_size, 2),
            x_test=torch.randn(self.test_size, 3),
            theta_ref=None,
            x_ref=None,
        )
