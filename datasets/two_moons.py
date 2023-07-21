r"""Dataset for the two-moons benchmark.

References
----------
    [1] Benchmarking Simulation-Based Inference (Lueckmann et al., 2021)
        https://arxiv.org/abs/2101.04653
"""

from benchopt import BaseDataset, safe_import_context
from typing import Dict

with safe_import_context() as import_ctx:
    import torch

    from benchmark_utils.common import fork, data_generator_sbibm


class Dataset(BaseDataset):
    r"""Two-moons dataset.

    Taken from the :mod:`sbibm` package [1].
    """

    name = "two_moons"
    # parameters that can be called with `self.<>`,
    # all possible combinations are used in the benchmark.
    parameters = {
        "train_size": [1024, 4096],
        "test_size": [256],
        "ref_size": [16],
        "n_per_ref": [1024],
        "seed": [42],
    }

    def get_data(self) -> Dict:
        r"""Generate data.

        Returns the input of the `Objective.set_data` method.
        """
        with fork():
            torch.manual_seed(self.seed)

            return data_generator_sbibm(
                self.name,
                self.train_size,
                self.test_size,
                self.ref_size,
                self.n_per_ref,
            )
