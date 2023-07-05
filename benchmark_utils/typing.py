r""""""

from typing import TypeVar

try:
    from torch import Tensor
    from torch.distributions import Distribution
except:
    Tensor = TypeVar("Tensor")
    Distribution = TypeVar("Distribution")
