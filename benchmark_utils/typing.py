r""""""

from typing import TypeVar

try:
    from torch import Tensor
    from torch.distributions import Distribution
except ImportError:
    Tensor = TypeVar("Tensor")
    Distribution = TypeVar("Distribution")
