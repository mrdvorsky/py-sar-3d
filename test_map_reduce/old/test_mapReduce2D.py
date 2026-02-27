from functools import partial

import jax
import jax.numpy as jnp

from jax_utils import export_graph, time_it

print(jax.devices())


def _kernelFun(x, y, x0, y0) -> jax.Array:
    return jnp.hypot(x - x0, y - y0)
    # return jnp.cos(x + x0 - y - y0)


# @jax.jit
# def testFun1(x, y, x0, y0):  # Create Intermediate Product
#     x = x[:, :, jnp.newaxis, jnp.newaxis]
#     y = y[:, :, jnp.newaxis, jnp.newaxis]
#     x0 = x0[jnp.newaxis, jnp.newaxis, :, :]
#     y0 = y0[jnp.newaxis, jnp.newaxis, :, :]

#     # return jnp.sum(_kernelFun(x, y, x0, y0), axis=[2, 3])
#     data = _kernelFun(x, y, x0, y0)
#     # Define the reduction manually
#     return jax.lax.reduce(
#         data,
#         0.0,
#         jax.lax.add,
#         (2, 3) # The axes to reduce
#     )


@jax.jit
def testFun1(x, y, x0, y0):  # Create Intermediate Product
    x = x[:, :, jnp.newaxis]
    y = y[:, :, jnp.newaxis]

    x0, y0 = jnp.broadcast_arrays(x0, y0)


    return jnp.sum(_kernelFun(x, y, x0, y0), axis=[2, 3, 4, 5, 6])


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

    return jax.lax.scan(bodyFun, 0, (x0_array.ravel(), y0_array.ravel()), unroll=16)[0]


@jax.jit
def _kernel_scalar_array_4(x_scalar, y_scalar, y0_scalar, x0_array):
    def bodyFun(carry, x0_scalar):
        return (carry + _kernelFun(x_scalar, y_scalar, y0_scalar, x0_scalar), None)

    return jax.lax.scan(bodyFun, 0, x0_array, unroll=8)[0]


@jax.jit
@partial(jnp.vectorize, excluded=(2, 3))
def testFun4(x_scalar, y_scalar, x0_array, y0_array):
    def outerScan(carry, y0_scalar):
        return (
            carry
            + _kernel_scalar_array_4(x_scalar, y_scalar, y0_scalar, x0_array.ravel()),
            None,
        )

    return jax.lax.scan(outerScan, 0, y0_array.ravel(), unroll=1)[0]


@jax.jit
@partial(jnp.vectorize, excluded=(2, 3))
def testFun5(x_scalar, y_scalar, x0_array, y0_array):  # Lazy broadcast
    x0y0_shape = jnp.broadcast_shapes(x0_array.shape, y0_array.shape)
    total_size = x0y0_shape[0] * x0y0_shape[1]
    inds = jax.lax.iota(jnp.int32, total_size)

    def bodyFun(carry, ind_scalar):
        xi, yi = jnp.unravel_index(ind_scalar, x0y0_shape)
        return (
            carry + _kernelFun(x_scalar, y_scalar, x0_array[xi, 1], y0_array[1, yi]),
            None,
        )

    return jax.lax.scan(bodyFun, 0, inds, unroll=16)[0]


@jax.jit
def testFun_Associative_Tiled(x, y, x0, y0):
    """
    A high-performance replacement for testFun3.
    Uses tiling to prevent Memory Wall and Associative reduction for speed.
    """
    # 1. Flatten the reduction coordinates for a clean scan
    x0_f = x0.ravel()
    y0_f = y0.ravel()

    # Define tile size (128-256 is usually the sweet spot for CPU/GPU)
    tile_size = 128
    num_tiles = x0_f.shape[0] // tile_size

    # Reshape into tiles (Assumes x0/y0 size is multiple of tile_size for simplicity)
    # If not, you can pad or use a smaller tile.
    x0_tiles = x0_f.reshape(num_tiles, tile_size)
    y0_tiles = y0_f.reshape(num_tiles, tile_size)

    def body_fun(carry, tile_data):
        tx0, ty0 = tile_data

        # Calculate kernel for the whole tile at once (Vectorized)
        # Result shape: (dim_x, tile_size)
        kernel_tile = jnp.cos(x + tx0 - y - ty0)

        # Use associative_scan to sum across the tile dimension
        # This is more parallel-friendly than a serial loop
        tile_sum = jax.lax.associative_scan(jnp.add, kernel_tile, axis=-1)[:, -1]

        return carry + tile_sum, None

    # Outer loop is a memory-safe Scan
    initial_carry = jnp.zeros((x.shape[0], y.shape[1]))
    total_sum, _ = jax.lax.scan(body_fun, initial_carry, (x0_tiles, y0_tiles))

    return total_sum


# Run
key = jax.random.PRNGKey(21312)
x = jax.random.normal(key, [256, 1])
y = jax.random.normal(key, [1, 256])

x0 = jax.random.normal(key, [32, 32, 32, 8, 1])
y0 = jax.random.normal(key, [32, 32, 32, 8, 1])

export_graph("dot_files/test_2d_1.dot", testFun1, x, y, x0, y0)
# exportGraph("dot_files/test_2d_2.dot", testFun2, x, y, x0, y0)
# exportGraph("dot_files/test_2d_3.dot", testFun3, x, y, x0, y0)
# exportGraph("dot_files/test_2d_4.dot", testFun4, x, y, x0, y0)
# exportGraph("dot_files/test_2d_5.dot", testFun5, x, y, x0, y0)


time_it(testFun1, x, y, x0, y0)
# time_it(testFun2, x, y, x0, y0)
time_it(testFun3, x, y, x0, y0)
# time_it(testFun4, x, y, x0, y0)
time_it(testFun_Associative_Tiled, x, y, x0, y0)

# print(testFun1(x, y, x0, y0))
# print(testFun2(x, y, x0, y0))
