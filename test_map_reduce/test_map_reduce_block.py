import jax
import jax.numpy as jnp


from jax_map_reduce import map_reduce
from jax_utils import time_it, export_graph


### Tests
def _kernel(x, y):
    return x - y
    return jnp.cos(x - y)


@jax.jit
def test1(x, y):
    return map_reduce(_kernel, x, y, axis=[2, 3], unroll_count=32)


@jax.jit
def test2(x, y):
    return jnp.sum(_kernel(x, y), axis=[2, 3])


@jax.jit
def test3(x, y):
    b = jnp.squeeze(y, axis=[0, 1])

    def _body_fun(a):
        a = jnp.ravel(a)[:, None]

        def _inner_fun(carry, inputs):
            return (carry + jnp.sum(_kernel(a, inputs), axis=[1]), None)

        return jax.lax.scan(_inner_fun, jnp.zeros([1]), b)[0]

    return jax.lax.map(_body_fun, x, batch_size=0)

# @jax.jit
# def test3(x, y):
#     x = jnp.reshape(x, (1000, 32, 1, 1, 1))
#     y = jnp.reshape(y, (1, 1, 32, 32, 32))
#     return jnp.sum(_kernel(x, y), axis=[2, 3, 4])


key = jax.random.PRNGKey(10)
x = jax.random.normal(key, [1000*128, 1, 1, 1])
y = jax.random.normal(key, [1, 1, 10240, 32])


export_graph(test1, x, y)
export_graph(test2, x, y)
export_graph(test3, x, y)

time_it(test1, x, y)
# time_it(test2, x, y)
time_it(test3, x, y)

print(jnp.max(jnp.abs(test1(x, y))))
print(jnp.max(jnp.abs(test3(x, y) - test1(x, y))))
