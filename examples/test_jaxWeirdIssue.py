import jax
import jax.numpy as jnp

from jax_utils import exportGraph, time_it


def _kernelFun(x, y, z) -> jax.Array:
    # return jnp.hypot(x, y + 2 * z)
    # return jnp.cos(x - y + z)
    return x + (2*y**2) + z


@jax.jit
def testFun1(x, y, z):
    return jnp.sum(_kernelFun(x, y, z), axis=[0])


@jax.jit
def testFun2(x, y, z):
    return jnp.sum(_kernelFun(x, y, z), axis=[1])

@jax.jit
def testFun3(x, y, z):
    def bodyFun(carry, xy):
        return (
            carry + _kernelFun(xy[0], xy[1], jnp.squeeze(z, axis=0)),
            None,
        )

    return jax.lax.scan(bodyFun, jnp.zeros([128, 1024*16, 512*2]), (x, y), unroll=16)[0]


# Run
key = jax.random.PRNGKey(21312)
x = jax.random.normal(key, [32, 128, 1024*16, 1])
y = jax.random.normal(key, [32, 128, 1, 512*2])
z = jax.random.normal(key, [1, 1, 1024*16, 512*2])

exportGraph("dot_files/test_mixed_1.dot", testFun1, x, y, z)
exportGraph("dot_files/test_mixed_2.dot", testFun2, x, y, z)
exportGraph("dot_files/test_mixed_3.dot", testFun3, x, y, z)


time_it(testFun1, x, y, z)
# time_it(testFun2, x, y, z)
time_it(testFun3, x, y, z)

print(testFun1.lower(x, y, z).compile().memory_analysis())


# print(jax.make_jaxpr(testFun1)(x, y, z))
# print(jax.make_jaxpr(testFun2)(x, y, z))
