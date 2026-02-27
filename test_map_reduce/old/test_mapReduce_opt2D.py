from functools import partial

import jax
import jax.numpy as jnp

from jax_utils import export_graph, time_it


@jax.jit
def _kernelFun(x, y) -> jax.Array:
    return jnp.hypot(x, y)
    # return x*?


@jax.jit
def testFun1(x, y):
    x = x[:, jnp.newaxis]
    y = y[jnp.newaxis, :]

    return jnp.sum(_kernelFun(x, y), axis=[0])


@jax.jit
@partial(jnp.vectorize, excluded=(1, 2))
def testFun2(x_scalar, y_array):
    def bodyFun(carry, y_scalar):
        return (
            carry + _kernelFun(x_scalar, y_scalar),
            None,
        )

    return jax.lax.scan(bodyFun, 0, y_array, unroll=16)[0]


# @jax.jit
# def testFun3(x, x0, y0):  # Create Intermediate Product Optimally
#     x = x[jnp.newaxis, :, :]

#     x0, y0 = jnp.broadcast_arrays(x0, y0)
#     x0 = x0.ravel()[:, jnp.newaxis, jnp.newaxis]
#     y0 = y0.ravel()[:, jnp.newaxis, jnp.newaxis]

#     print(x.shape, x0.shape, y0.shape)

#     return jnp.sum(_kernelFun(x, x0, y0), axis=[0])


# Run
key = jax.random.PRNGKey(21312)
x = jax.random.normal(key, [256*256])
y = jax.random.normal(key, [256*128])

export_graph("dot_files/test_2d_1.dot", testFun1, x, y)
export_graph("dot_files/test_2d_2.dot", testFun2, x, y)


time_it(testFun1, x, y)
time_it(testFun2, x, y)

