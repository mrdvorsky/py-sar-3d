import math

import jax
import jax.numpy as jnp

from jax_utils import time_it, exportGraph


@jax.jit
def _kernelFun(x, y, z) -> jax.Array:
    return x**2 + y - 3.0 * z
    # return jnp.hypot(x, y + 2 * z)
    # return jnp.cos(x - y + z)








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
