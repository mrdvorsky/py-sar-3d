
import jax
import jax.numpy as jnp


from jax_utils import exportGraph, time_it



def _kernelFun(x, y, z) -> jax.Array:
    # return jnp.hypot(x, y + z)
    return jnp.cos(x + y - z)


@jax.jit
def testFun1(x, y, z):
    # x = jnp.permute_dims(x, (2, 3, 0, 1))
    # y = jnp.permute_dims(y, (2, 3, 0, 1))
    # z = jnp.permute_dims(z, (2, 3, 0, 1))
    sumVal = _kernelFun(x, y, z)
    return jnp.sum(sumVal, axis=[0])

@jax.jit
def testFun2(x, y, z):
    x = jnp.reshape(x, (32, 32, 1, 1024))
    y = jnp.reshape(y, (32, 32, 1024, 1))
    z = jnp.reshape(z, (32, 32, 1, 1024))
    sumVal = _kernelFun(x, y, z)
    return jnp.sum(sumVal, axis=[0, 1])





# Run
key = jax.random.PRNGKey(21312)
x = jax.random.normal(key, [1024, 1, 1024])
y = jax.random.normal(key, [1024, 1024, 1])
z = jax.random.normal(key, [1024, 1, 1024])
# x = jax.random.normal(key, [1024, 1, 1, 32])
# y = jax.random.normal(key, [1, 1024, 32, 1])
# z = jax.random.normal(key, [1, 1, 32, 32])

exportGraph("dot_files/test_2d_1.dot", testFun1, x, y, z)
exportGraph("dot_files/test_2d_2.dot", testFun2, x, y, z)


time_it(testFun1, x, y, z)
time_it(testFun2, x, y, z)


print(testFun1(x, y, z) - testFun2(x, y, z))
