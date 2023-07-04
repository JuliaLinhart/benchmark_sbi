from benchopt import BaseDataset, safe_import_context
from typing import Dict

with safe_import_context() as import_ctx:
    import torch
    from benchmark_utils.common import fork, data_generator_sbibm


class Dataset(BaseDataset):
    name = "slcp"
    parameters = {
        "train_size": [1024, 4096],
        "test_size": [256],
        "ref_size": [0],
        "n_per_ref": [1024],
        "seed": [42],
    }

    def get_data(self) -> Dict:
        with fork():
            torch.manual_seed(self.seed)

            return data_generator_sbibm(
                self.name,
                self.train_size,
                self.test_size,
                self.ref_size,
                self.n_per_ref,
            )
