import jax
import jax.numpy as jnp

from jax_map_reduce import map_reduce
from jax_utils import time_it, exportGraph



### Tests
def _kernel(x, y):
    return x + x*y - 2*y + 3.0*x**2
    # return jnp.cos(x - y)

@jax.jit
def test1(x, y):
    return map_reduce(_kernel, x, y, axis=[2, 3], unroll_count=1)

@jax.jit
def test2(x, y):
    return map_reduce(_kernel, x, y, axis=[3, 2], unroll_count=1)


x = jnp.zeros([1024, 1, 1000, 1])
y = jnp.zeros([1, 2000, 1, 32])


exportGraph(test1, x, y)
exportGraph(test2, x, y)

time_it(test1, x, y)
time_it(test2, x, y)


