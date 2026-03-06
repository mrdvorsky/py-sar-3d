import jax
import jax.numpy as jnp


from jax_map_reduce import map_reduce
from jax_map_reduce_indexing import map_reduce_indexing
from jax_utils import time_it, export_graph


### Tests
def _kernel(x, y):
    return x + x * y - 2 * y + 3.0 * x**2
    return jnp.cos(x - y)


@jax.jit
def test1(x, y):
    return map_reduce(_kernel, x, y, axis=[1, 2], unroll_count=32)


# @jax.jit
# def test2(x, y):
#     a = jnp.squeeze(x, axis=[0])

#     def _test2_helper(b):
#         print(a, b)
#         return jnp.sum(_kernel(a, b))

#     return jax.lax.map(_test2_helper, y, batch_size=16)


key = jax.random.PRNGKey(10)
x = jax.random.normal(key, [1*1, 1024, 2000])
y = jax.random.normal(key, [32*32, 1, 1])


export_graph(test1, x, y)
# export_graph(test2, x, y)

time_it(test1, x, y)
# time_it(test2, x, y)

print(jnp.max(jnp.abs(test1(x, y))))
# print(jnp.max(jnp.abs(test2(x, y) - test1(x, y))))
