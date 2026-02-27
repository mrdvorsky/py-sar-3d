import math

import jax
import jax.numpy as jnp

from jax_utils import time_it, export_graph


@jax.jit
def _kernelFun(x, y, z) -> jax.Array:
    return x**2 + y - 3.0 * z
    return jnp.hypot(x, y + 2 * z)
    return jnp.cos(x - y + z)


@jax.jit
def testFun1(x, y, z):
    return jnp.sum(_kernelFun(x, y, z), axis=0)


@jax.jit
def testFun2(x, y, z):
    z_squeezed = jnp.squeeze(z, axis=0)

    def bodyFun(carry, inputs):
        xs, ys = inputs
        return (carry + _kernelFun(xs, ys, z_squeezed), None)

    return jax.lax.scan(
        bodyFun,
        jnp.zeros_like(z_squeezed),
        (x, y),
        unroll=16,
    )[0]


@jax.jit
def testFun3(x, y, z):
    x = jnp.transpose(x, (2, 0, 1))
    y = jnp.transpose(y, (2, 0, 1))
    z = jnp.transpose(z, (2, 0, 1))
    y_squeezed = jnp.squeeze(y, axis=0)

    def bodyFun(carry, inputs):
        xs, zs = inputs
        return (carry + _kernelFun(xs, y_squeezed, zs), None)

    return jax.lax.scan(
        bodyFun,
        jnp.zeros([y.shape[1], y.shape[2]]),
        (x, z),
        unroll=1,
    )[0]


@jax.jit
def _helper4(x, yS, z):
    def bodyFun(carry, inputs):
        xS, zS = inputs
        return (carry + _kernelFun(xS, yS, zS), None)

    return jax.lax.scan(
        bodyFun,
        jnp.zeros([1]),
        (x, z),
        unroll=16,
    )[0][0]


@jax.jit
def testFun4(x, y, z):
    return jnp.vectorize(_helper4, signature="(m),(1),(m)->()")(x, y, z)


def getAllRavelIndices(*args: jax.Array):
    shape = jnp.broadcast_shapes(*[a.shape for a in args])
    subsAll = jnp.indices(shape, sparse=True)
    subsAll = jnp.broadcast_arrays(*subsAll)
    return tuple(
        jnp.ravel_multi_index(subsAll, a.shape, mode="wrap").ravel() for a in args
    )

def getAllUnRavelSubs(*args: jax.Array):
    shape = jnp.broadcast_shapes(*[a.shape for a in args])
    subsAll = jnp.indices(shape, sparse=True)
    subsAll = jnp.broadcast_arrays(*subsAll)
    return tuple(
        jnp.ravel_multi_index(subsAll, a.shape, mode="wrap").ravel() for a in args
    )


@jax.jit
def _helper5(x, y, z):
    ind

    def bodyFun(carry, inputs):
        ixS, iyS, izS = inputs
        return (carry + _kernelFun(x[ixS], y[iyS], z[izS]), None)

    return jax.lax.scan(
        bodyFun,
        0,
        (xi, yi, zi),
        unroll=16,
    )[0]


@jax.jit
def testFun5(x, y, z):
    return jnp.vectorize(_helper5, signature="(m),(1),(m)->()")(x, y, z)


### Testing ###
key = jax.random.PRNGKey(10)
x = jax.random.normal(key, [1024, 1,    32, 32])
y = jax.random.normal(key, [1024, 2048, 1,  1])
z = jax.random.normal(key, [1, 2048,    1,  32])


# exportGraph("dot_files/test1.dot", testFun1, x, y, z)
# exportGraph("dot_files/test2.dot", testFun2, x, y, z)
# exportGraph("dot_files/test3.dot", testFun3, x, y, z)
# exportGraph("dot_files/test4.dot", testFun4, x, y, z)
export_graph("dot_files/test5.dot", testFun5, x, y, z)

time_it(testFun1, x, y, z)
# time_it(testFun2, x, y, z)
# time_it(testFun3, x, y, z)
# time_it(testFun4, x, y, z)
# time_it(testFun5, x, y, z)


# shape = (2, 3, 4)

# print(*getAllRavelIndices(jnp.zeros([2, 1]), jnp.zeros([2, 3])), sep="\n\n")
