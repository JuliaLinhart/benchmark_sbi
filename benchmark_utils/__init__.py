import torch
from contextlib import contextmanager

@contextmanager
def fork():
    try:
        state = torch.random.get_rng_state()
        yield
    finally:
        torch.set_rng_state(state)
