import jax
import jax.numpy as jnp



# from jax_map_reduce import map_reduce
# from jax_utils import time_it, export_graph



### Tests
def _kernel(x, y):
    return x + x*y - 2*y + 3.0*x**2
    # return jnp.cos(x - y)

@jax.jit
def test1(x, y):
    return map_reduce(_kernel, x, y, axis=[0, 1], unroll_count=1)

@jax.jit
def test2(x, y):
    return map_reduce(_kernel, x, y, axis=[0, 2], unroll_count=1)


# x = jnp.zeros([1024, 1, 1000, 33])
# y = jnp.zeros([1024, 2000, 1, 1])

x = jnp.zeros([33, 1024, 1024, 1])
y = jnp.zeros([1, 1, 1024, 2000])


export_graph(test1, x, y)
export_graph(test2, x, y)

time_it(test1, x, y)
time_it(test2, x, y)


