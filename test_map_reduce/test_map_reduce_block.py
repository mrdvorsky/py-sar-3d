import jax
import jax.numpy as jnp


from jax_map_reduce import map_reduce
from jax_utils import time_it, export_graph


### Tests
def _kernel(x, y):
    return x + x * y - 2 * y + 3.0 * x**2
    # return jnp.cos(x - y)


@jax.jit
def test1(x, y):
    return map_reduce(_kernel, x, y, axis=[0, 1], unroll_count=16)


@jax.jit
def test2(x, y):
    return map_reduce(_kernel, x, y, axis=[0, 1], unroll_count=16)


@jax.jit
def test3(x, y):
    return jnp.sum(_kernel(x, y), axis=[0, 1])


@jax.jit
def test4(x, y):
    def inner(carry, inputs):
        a, b = inputs
        return (carry + jnp.sum(_kernel(a, b), axis=0), None)

    ret, _ = jax.lax.scan(inner, jnp.zeros([1024, 2000]), (x, y))
    return ret


# x = jnp.ones([1024, 1, 3072])
# y = jnp.ones([1, 2000, 3072])
x = jnp.ones([32, 32, 1024, 1])
y = jnp.ones([32, 32, 1, 2000])


export_graph(test1, x, y)
# export_graph(test2, x, y)
# export_graph(test3, x, y)
export_graph(test4, x, y)

time_it(test1, x, y)
# time_it(test2, x, y)
# time_it(test3, x, y)
time_it(test4, x, y)
