from benchopt import BaseDataset, safe_import_context
from typing import Dict

with safe_import_context() as import_ctx:
    import torch
    from benchmark_utils import fork, data_generator_sbibm


class Dataset(BaseDataset):
    name = "two_moons"
    parameters = {
        "size": [1024, 4096],
        "seed": [42],
    }

    def get_data(self) -> Dict:
        with fork():
            torch.manual_seed(self.seed)

            return data_generator_sbibm(self.size, self.name, reference=True)
