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


@jax.jit
def test4(x, y):
    return map_reduce_indexing(_kernel, x, y, axis=[2, 3], unroll_count=1)


@jax.jit
def test3(x, y):
    return jnp.sum(_kernel(x, y), axis=[2, 3])


# x = jnp.ones([1024, 1, 3072])
# y = jnp.ones([1, 2000, 3072])
# x = jnp.ones([32, 32, 1024, 1])
# y = jnp.ones([32, 32, 1, 2000])

key = jax.random.PRNGKey(10)
x = jax.random.normal(key, [32, 32, 1, 1])
y = jax.random.normal(key, [1, 1, 1024, 100])


export_graph(test1, x, y)
export_graph(test4, x, y)

time_it(test1, x, y)
time_it(test4, x, y)

print(jnp.max(jnp.abs(test3(x, y))))
print(jnp.max(jnp.abs(test4(x, y) - test3(x, y))))
