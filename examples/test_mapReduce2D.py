from functools import partial

import jax
import jax.numpy as jnp

from jax_utils import exportGraph, time_it


def _kernelFun(x, y, x0, y0) -> jax.Array:
    # return jnp.hypot(x - x0, y - y0)
    return jnp.cos(x + x0 - y - y0)


@jax.jit
def testFun1(x, y, x0, y0):  # Create Intermediate Product
    x = x[:, :, jnp.newaxis, jnp.newaxis]
    y = y[:, :, jnp.newaxis, jnp.newaxis]
    x0 = x0[jnp.newaxis, jnp.newaxis, :, :]
    y0 = y0[jnp.newaxis, jnp.newaxis, :, :]

    return jnp.sum(_kernelFun(x, y, x0, y0), axis=[2, 3])


@jax.jit
@partial(jnp.vectorize, excluded=(2, 3))
def testFun2(x_scalar, y_scalar, x0_array, y0_array):
    x0_array, y0_array = jnp.broadcast_arrays(x0_array, y0_array)

    def bodyFun(carry, x0y0_scalar):
        return (
            carry + _kernelFun(x_scalar, y_scalar, x0y0_scalar[0], x0y0_scalar[1]),
            None,
        )

    return jax.lax.scan(bodyFun, 0, (x0_array.ravel(), y0_array.ravel()))[0]


@jax.jit
@partial(jnp.vectorize, excluded=(2, 3))
def testFun3(x_scalar, y_scalar, x0_array, y0_array):
    x0_array, y0_array = jnp.broadcast_arrays(x0_array, y0_array)

    def bodyFun(carry, x0y0_scalar):
        return (
            carry + _kernelFun(x_scalar, y_scalar, x0y0_scalar[0], x0y0_scalar[1]),
            None,
        )

    return jax.lax.scan(bodyFun, 0, (x0_array.ravel(), y0_array.ravel()), unroll=4)[0]


# Run
key = jax.random.PRNGKey(21312)
x = jax.random.normal(key, [256, 1])
y = jax.random.normal(key, [1, 256])

x0 = jax.random.normal(key, [136, 1])
y0 = jax.random.normal(key, [1, 248])

exportGraph("dot_files/test_2d_1.dot", testFun1, x, y, x0, y0)
exportGraph("dot_files/test_2d_2.dot", testFun2, x, y, x0, y0)
exportGraph("dot_files/test_2d_3.dot", testFun3, x, y, x0, y0)

time_it(testFun1, x, y, x0, y0)
time_it(testFun2, x, y, x0, y0)
time_it(testFun3, x, y, x0, y0)

# print(testFun1(x, y, x0, y0))
# print(testFun2(x, y, x0, y0))
