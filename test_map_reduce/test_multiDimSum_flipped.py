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
    return jnp.sum(_kernelFun(x, y, z), axis=[0, 1])


@jax.jit
def testScan(x, yS, z):
    bShape = jnp.broadcast_shapes(x.shape, yS.shape, z.shape)
    x = jnp.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
    yS = jnp.reshape(yS, (yS.shape[0] * yS.shape[1], yS.shape[2], yS.shape[3]))
    z = jnp.broadcast_to(z, (bShape[0], bShape[1], z.shape[2], z.shape[3]))
    z = jnp.reshape(z, (bShape[0] * bShape[1], z.shape[2], z.shape[3]))

    yS = jnp.squeeze(yS, axis=0)

    def bodyFun(carry, inputs):
        xS, zS = inputs
        return (carry + _kernelFun(xS, yS, zS), None)

    return jax.lax.scan(
        bodyFun,
        jnp.zeros([bShape[2], bShape[3]]),
        (x, z),
        unroll=16,
    )[0]


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
def testScanMulti1(x, y, z):
    x = jnp.transpose(x, (2, 3, 1, 0))
    y = jnp.transpose(y, (2, 3, 1, 0))
    z = jnp.transpose(z, (2, 3, 1, 0))
    return jnp.vectorize(_helperMulti1, signature="(m,n),(1,1),(1,n)->()")(x, y, z)


def _helper2(x, yS, z):
    shape = jnp.broadcast_shapes(x.shape, yS.shape, z.shape)
    yS = jnp.squeeze(yS, axis=0)

    def bodyFun(carry, inputs):
        xS, zS = inputs
        return (carry + _kernelFun(xS, yS, zS), None)

    return jax.lax.scan(
        bodyFun,
        jnp.zeros([shape[-2], shape[-1]]),
        (x, z),
        unroll=16,
    )[0]


def _helper1(x, yS, zS):
    shape = jnp.broadcast_shapes(x.shape, yS.shape, zS.shape)
    yS = jnp.squeeze(yS, axis=0)
    zS = jnp.squeeze(zS, axis=0)

    def bodyFun(carry, inputs):
        xS = inputs
        return (carry + _helper2(xS, yS, zS), None)

    return jax.lax.scan(
        bodyFun,
        jnp.zeros([shape[-2], shape[-1]]),
        x,
        unroll=1,
    )[0]


@jax.jit
def testScanMulti2(x, y, z):
    return _helper1(x, y, z)


### Testing ###
key = jax.random.PRNGKey(10)
x = jax.random.normal(key, [256, 31, 1, 2000])
y = jax.random.normal(key, [1, 1, 1000, 2000])
z = jax.random.normal(key, [1, 31, 1000, 1])


# exportGraph("dot_files/testDirect.dot", testDirect, x, y, z)
exportGraph("dot_files/testScan.dot", testScan, x, y, z)
# exportGraph("dot_files/testScanMulti1.dot", testScanMulti1, x, y, z)
exportGraph("dot_files/testScanMulti2.dot", testScanMulti2, x, y, z)
# exportGraph("dot_files/testScanMultiLinear.dot", testScanMultiLinear, x, y, z)

# time_it(testDirect, x, y, z)
time_it(testScan, x, y, z)
# time_it(testScanMulti1, x, y, z)
time_it(testScanMulti2, x, y, z)
# time_it(testScanMultiLinear, x, y, z)
