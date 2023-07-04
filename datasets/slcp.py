from benchopt import BaseDataset, safe_import_context
from typing import Dict

with safe_import_context() as import_ctx:
    import torch
    from benchmark_utils.common import fork, data_generator_sbibm


class Dataset(BaseDataset):
    name = "slcp"
    parameters = {
        "size": [1024, 4096],
        "seed": [42],
    }

    def get_data(self) -> Dict:
        with fork():
            torch.manual_seed(self.seed)

            return data_generator_sbibm(self.size, self.name, reference=False)
