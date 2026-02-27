import jax
import jax.numpy as jnp

from jax_utils import export_graph, time_it


def _kernelFun(x, y, z) -> jax.Array:
    return jnp.hypot(x, y + 2*z)
    # return jnp.cos(x - y) + z


@jax.jit
def testFun1(x, y, z):
    return jnp.sum(_kernelFun(x, y, z), axis=[0, 1])


@jax.jit
def testFun2(x, y, z):
    shape = jnp.broadcast_shapes(x.shape, y.shape, z.shape)
    shape = (shape[1], shape[2])

    def bodyFun(carry, xy_reduced):
        return (
            carry + _kernelFun(xy_reduced[0], xy_reduced[1], jnp.squeeze(z)),
            None,
        )

    return jax.lax.scan(bodyFun, jnp.zeros(shape), (x, y), unroll=16)[0]


# Run
key = jax.random.PRNGKey(21312)
x = jax.random.normal(key, [32, 128, 1024, 1])
y = jax.random.normal(key, [32, 128, 1, 512])
z = jax.random.normal(key, [32, 1, 1024, 512])

export_graph("dot_files/test_mixed_1.dot", testFun1, x, y, z)
# exportGraph("dot_files/test_mixed_2.dot", testFun2, x, y, z)
# exportGraph("dot_files/test_mixed_3.dot", testFun3, x, y, z)


time_it(testFun1, x, y, z)
# time_it(testFun2, x, y, z)
# time_it(testFun3, x, y)
