import jax
import jax.numpy as jnp


from jax_map_reduce import map_reduce
from jax_map_reduce_indexing import map_reduce_indexing
from jax_utils import time_it, export_graph


### Tests
def _kernel(x, y):
    return x - y
    # return jnp.cos(x - y)


@jax.jit(static_argnames=("axis",))
def test1(x, y, axis):
    return map_reduce(_kernel, x, y, axis=axis, unroll_count=16)


@jax.jit(static_argnames=("axis",))
def test2(x, y, axis):
    return map_reduce_indexing(_kernel, x, y, axis=axis, unroll_count=16)



# x = jnp.ones([1024, 1, 3072])
# y = jnp.ones([1, 2000, 3072])
# x = jnp.ones([32, 32, 1024, 1])
# y = jnp.ones([32, 32, 1, 2000])

key = jax.random.PRNGKey(10)
x = jax.random.normal(key, [100, 1, 3, 1024, 1])
y = jax.random.normal(key, [100, 64, 1, 1, 1000])
reduce_axis = (0, 1, 2)

export_graph(test1, x, y, reduce_axis)
export_graph(test2, x, y, reduce_axis)

time_it(test1, x, y, reduce_axis, num_reps=1)
time_it(test2, x, y, reduce_axis, num_reps=1)

# print(jnp.max(jnp.abs(test1(x, y, reduce_axis))))
# print(jnp.max(jnp.abs(test2(x, y, reduce_axis) - test1(x, y, reduce_axis))))


