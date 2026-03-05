
import torch
import torch.jit as jit

from jax_utils import time_it_torch



@jit.script
def _kernel(x, y):
    return x + x * y - 2 * y + 3.0 * x**2
    # return torch.cos(x - y)





x = torch.ones([1024, 1, 1000])
y = torch.ones([1, 2000, 1000])


def test1(a, b):
    return torch.sum(_kernel(a, b), dim=[2])

time_it_torch(test1, x, y)


