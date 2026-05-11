import jax
import jax.numpy as jnp


from jax_map_reduce import map_reduce
from jax_map_reduce_indexing import map_reduce_indexing
from jax_utils import time_it, export_graph


### Tests
def _kernel(x, y):
    return x * y
    # return jnp.cos(x - y)


@jax.jit(static_argnames=("axis",))
def test1(x, y, axis):
    return map_reduce(_kernel, x, y, axis=axis, unroll_count=16)


@jax.jit(static_argnames=("axis",))
def test2(x, y, axis):
    return map_reduce_indexing(_kernel, x, y, axis=axis, unroll_count=2)


@jax.jit
def test3(x: jax.Array, y: jax.Array):
    x_shape = x.shape[:3]
    y_shape = y.shape[:3]
    out_shape = jnp.broadcast_shapes(x_shape, y_shape)
    x = jnp.reshape(x, (-1, x.shape[3], x.shape[4]))
    y = jnp.reshape(y, (-1, y.shape[3], y.shape[4]))

    inds = jnp.indices(out_shape, sparse=True)
    x_inds = jnp.ravel_multi_index(inds, x_shape, mode="clip")
    y_inds = jnp.ravel_multi_index(inds, y_shape, mode="clip")
    
    x_inds = jnp.reshape(x_inds, (-1, 32, 32, 32))
    y_inds = jnp.reshape(y_inds, (-1, 32, 32, 32))
    
    return jnp.sum(_kernel(x[x_inds], y[y_inds]), axis=(0, 1, 2))
    



key = jax.random.PRNGKey(10)
x = jax.random.normal(key, [128, 1, 64, 1024, 1])
y = jax.random.normal(key, [128, 32, 1, 1, 1000])
reduce_axis = (0, 1, 2)

# export_graph(test1, x, y, reduce_axis)
# export_graph(test2, x, y, reduce_axis)
# export_graph(test3, x, y)

# test3(x, y)
time_it(test1, x, y, reduce_axis, num_reps=1)
time_it(test2, x, y, reduce_axis, num_reps=1)


# print(jnp.max(jnp.abs(test1(x, y, reduce_axis))))
# print(jnp.max(jnp.abs(test2(x, y, reduce_axis) - test1(x, y, reduce_axis))))


