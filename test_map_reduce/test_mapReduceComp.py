import jax
import jax.numpy as jnp

from jax_utils import exportGraph, time_it


key = jax.random.PRNGKey(21312)
v1 = jax.random.normal(key, [256 * 128])
v2 = jax.random.normal(key, [256 * 126])


def _kernelFun(x, y) -> jax.Array:
    return jnp.hypot(x, 2 * y)
    # return jnp.cos(x - 2 * y)


@jax.jit
def testFun1(x, y):  # Create Intermediate Product
    x = x[:, None]
    y = y[None, :]
    return jnp.sum(_kernelFun(x, y), axis=1)


@jax.jit
def testFun2(x, y):  # Create x-array at each iteration
    def bodyFun(carry, y_scalar):
        return (carry + _kernelFun(x, y_scalar), None)

    return jax.lax.scan(bodyFun, jnp.zeros_like(x), y)[0]


@jax.jit
def testFun3(x, y):  # Create x-array at each iteration with unrolling
    def bodyFun(carry, y_scalar):
        return (carry + _kernelFun(x, y_scalar), None)

    return jax.lax.scan(bodyFun, jnp.zeros_like(x), y, unroll=8)[0]


@jax.jit
def testFun4(x, y):  # Create each element and stack
    def bodyFun(carry, x_scalar):
        return (carry, jnp.sum(_kernelFun(x_scalar, y)))

    return jax.lax.scan(bodyFun, None, x, unroll=1)[1]


@jax.jit
def testFun5(x, y):
    def process_row(xi):
        # We use a scan here to ensure the inner loop
        # stays serial and memory-efficient per thread.
        def body(carry, y_val):
            return carry + _kernelFun(xi, y_val), None

        # Unroll here to allow the thread to use SIMD internally
        res, _ = jax.lax.scan(body, 0.0, y, unroll=16)
        return res

    # lax.map on CPU often lowers to an OpenMP-style parallel for-loop
    return jax.lax.map(process_row, x, batch_size=16)

@jax.jit
def testFun6(x, y):
    def process_row(xi):
        return jnp.sum(_kernelFun(xi, y))

    # lax.map on CPU often lowers to an OpenMP-style parallel for-loop
    return jax.lax.map(process_row, x, batch_size=16)


# exportGraph("dot_files/test1.dot", testFun1, v1, v2)
# exportGraph("dot_files/test2.dot", testFun2, v1, v2)
# exportGraph("dot_files/test3.dot", testFun3, v1, v2)
# exportGraph("dot_files/test4.dot", testFun4, v1, v2)
# exportGraph("dot_files/test5.dot", testFun5, v1, v2)
# exportGraph("dot_files/test6.dot", testFun6, v1, v2)

# time_it(testFun1, v1, v2)
# time_it(testFun2, v1, v2)
time_it(testFun3, v1, v2)
# time_it(testFun4, v1, v2)
# time_it(testFun5, v1, v2)
