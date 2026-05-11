
import jax
import jax.numpy as jnp


from jax_map_reduce import map_reduce
from jax_utils import time_it, export_graph


### Tests
def _kernel(x, y):
    return x * y
    return jnp.cos(x - y)


@jax.jit
def test1(x, y):
    return map_reduce(_kernel, x, y, axis=[0, 1, 2], unroll_count=32)

@jax.jit
def test2(x, y):
    return map_reduce(_kernel, x, y, axis=[0], unroll_count=32)

@jax.jit
def test4(x, y):
    # return jnp.einsum("abc,dbe->ae", x, y, optimize=True)
    return jnp.einsum("abc,ade->be", x, y, optimize=True)


@jax.jit
def test5(x, y):
    # def inner_fun(inputs):
    #     a = jnp.squeeze(x, axis=[0])
    #     b, = inputs
    #     return jnp.sum(_kernel(a, b))
    
    # return jax.lax.map(inner_fun, (y, ), batch_size=32)
    return jnp.sum(_kernel(x, y), axis=[1, 2, 3])



key = jax.random.PRNGKey(10)
# x = jax.random.normal(key, [4000, 2048, 1])
# y = jax.random.normal(key, [1, 2048, 6200])

x = jax.random.normal(key, [2048, 4000, 1])
y = jax.random.normal(key, [2048, 1, 6200])

x5 = jax.random.normal(key, [1, 16, 16, 8])
y5 = jax.random.normal(key, [32*32*32*32*32, 1, 1, 1])



export_graph(test1, x, y)
export_graph(test2, x, y)
export_graph(test4, x, y)
export_graph(test5, x5, y5)

# time_it(test1, x, y)
time_it(test2, x, y)
time_it(test4, x, y)
time_it(test5, x5, y5)

# print(jnp.max(jnp.abs(test1(x, y))))
# print(jnp.max(jnp.abs(test2(x, y) - test1(x, y))))
