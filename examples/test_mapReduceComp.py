
import jax
import jax.numpy as jnp

from jax_exportGraph import exportGraph

v1 = jnp.ones([10000, 1])
v2 = jnp.ones([1, 20000])


def _kernelFun(x, y) -> jax.Array:
    return jnp.cos(x - y)



@jax.jit
def testFun(x, y):
    return jnp.sum(_kernelFun(x, y), axis=1)

exportGraph("test.dot", testFun, v1, v2)
