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
    return map_reduce(_kernel, x, y, axis=[2, 3], unroll_count=1)




key = jax.random.PRNGKey(10)
x = jax.random.normal(key, [1, 1, 32, 32])
y = jax.random.normal(key, [2000, 3000, 1, 1])


export_graph(test1, x, y)
# export_graph(test4, x, y)

time_it(test1, x, y)
# time_it(test4, x, y)

# print(jnp.max(jnp.abs(test3(x, y))))
# print(jnp.max(jnp.abs(test4(x, y) - test3(x, y))))
