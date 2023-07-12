from benchopt import BaseDataset, safe_import_context
from typing import Dict

with safe_import_context() as import_ctx:
    import torch


class Dataset(BaseDataset):
    """Dummy dataset."""

    name = "simulated"
    parameters = {
        "train_size": [1024],
        "test_size": [256],
        "ref_size": [16],
        "n_per_ref": [1024],
        "seed": [42],
    }

    install_cmd = "conda"
    requirements = [
        "pytorch",
    ]

    def get_data(self) -> Dict:
        return dict(
            prior=torch.distributions.MultivariateNormal(
                torch.zeros(2),
                torch.eye(2),
            ),
            theta_train=torch.randn(self.train_size, 2),
            x_train=torch.randn(self.train_size, 3),
            theta_test=torch.randn(self.test_size, 2),
            x_test=torch.randn(self.test_size, 3),
            theta_ref=[
                torch.randn(self.n_per_ref, 2) 
                for i in range(self.ref_size)
            ],
            x_ref=torch.randn(self.ref_size, 3),
        )
