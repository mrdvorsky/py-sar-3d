import math

import jax
import jax.numpy as jnp

from jax_utils import time_it, exportGraph


@jax.jit
def _kernelFun(x, y, z) -> jax.Array:
    return x**2 + y - 3.0 * z
    # return jnp.hypot(x, y + 2 * z)
    # return jnp.cos(x - y + z)


@jax.jit
def testDirect(x, y, z):
    return jnp.sum(_kernelFun(x, y, z), axis=[2, 3])


@jax.jit
def testScan(x, y, z):
    @jax.jit
    def _helperScan(x, yS, z):
        def bodyFun(carry, inputs):
            xS, zS = inputs
            return (carry + _kernelFun(xS, yS, zS), None)

        return jax.lax.scan(
            bodyFun,
            jnp.zeros([1]),
            (x, z),
            unroll=16,
        )[0][0]

    bShape = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
    x = jnp.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
    y = jnp.reshape(y, (y.shape[0], y.shape[1], y.shape[2] * y.shape[3]))
    z = jnp.broadcast_to(z, (z.shape[0], z.shape[1], bShape[2], bShape[3]))
    z = jnp.reshape(z, (z.shape[0], z.shape[1], bShape[2] * bShape[3]))

    return jnp.vectorize(_helperScan, signature="(m),(1),(m)->()")(x, y, z)


@jax.jit
def _helperMulti2(xa, ys, za):
    ys = jnp.squeeze(ys, axis=0)

    def bodyFun(carry, inputs):
        xs, zs = inputs
        return (carry + _kernelFun(xs, ys, zs), None)

    return jax.lax.scan(
        bodyFun,
        0,
        (xa, za),
        unroll=16,
    )[0]


@jax.jit
def _helperMulti1(xa, ys, zs):
    ys = jnp.squeeze(ys, axis=0)
    zs = jnp.squeeze(zs, axis=0)

    def bodyFun(carry, inputs):
        xs = inputs
        return (carry + _helperMulti2(xs, ys, zs), None)

    return jax.lax.scan(
        bodyFun,
        0,
        xa,
        unroll=1,
    )[0]


@jax.jit
def testScanMulti(x, y, z):
    return jnp.vectorize(_helperMulti1, signature="(m,n),(1,1),(1,n)->()")(x, y, z)


@jax.jit
def _helperLinear(x: jax.Array, y: jax.Array, z: jax.Array):
    shape = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
    subsAll = jnp.indices(shape)

    def bodyFun(carry, subs):
        s1,s2 = subs
        return (carry + _kernelFun(x[s2, s1], y[0, 0], z[0, s1]), None)

    return jax.lax.scan(
        bodyFun,
        0,
        (subsAll[0].ravel(), subsAll[1].ravel()),
        unroll=16,
    )[0]


@jax.jit
def testScanMultiLinear(x, y, z):
    return jnp.vectorize(_helperLinear, signature="(m,n),(1,1),(1,n)->()")(x, y, z)


### Testing ###
key = jax.random.PRNGKey(10)
x = jax.random.normal(key, [1024, 1, 256 * 1, 32])
y = jax.random.normal(key, [1024, 1000, 1, 1])
z = jax.random.normal(key, [1, 1000, 1, 32])


# exportGraph("dot_files/testDirect.dot", testDirect, x, y, z)
exportGraph("dot_files/testScan.dot", testScan, x, y, z)
exportGraph("dot_files/testScanMulti.dot", testScanMulti, x, y, z)
exportGraph("dot_files/testScanMultiLinear.dot", testScanMultiLinear, x, y, z)

# time_it(testDirect, x, y, z)
time_it(testScan, x, y, z)
time_it(testScanMulti, x, y, z)
time_it(testScanMultiLinear, x, y, z)


# print(testDirect(x, y, z).shape)
# print(testScan(x, y, z).shape)
# print(testScanMulti(x, y, z).shape)

# print(jnp.max(jnp.abs(testScanMulti(x, y, z) - testDirect(x, y, z))))
