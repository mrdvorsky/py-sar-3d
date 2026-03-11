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
    return map_reduce(_kernel, x, y, axis=[1], unroll_count=32)

@jax.jit
def test3(x, y):
    return jnp.sum(_kernel(x, y), axis=None)

@jax.jit
def test4(x, y):
    return jnp.einsum("abc,dbe->ae", x, y, optimize=True)



@jax.jit
def test5(x, y):
    def scan_sum(val):
        def body_fun(carry, inputs):
            return (carry + inputs, None)
        
        return jax.lax.scan(body_fun, jnp.zeros((16, 32)), val)

    return scan_sum(_kernel(x, y))



key = jax.random.PRNGKey(10)
x = jax.random.normal(key, [100*120*200, 16, 1])
y = jax.random.normal(key, [1, 16, 32])


export_graph(test1, x, y)
export_graph(test2, x, y)
export_graph(test4, x, y)
export_graph(test5, x, y)

time_it(test1, x, y)
time_it(test2, x, y)
time_it(test3, x, y)
time_it(test4, x, y)

# print(jnp.max(jnp.abs(test1(x, y))))
# print(jnp.max(jnp.abs(test2(x, y) - test1(x, y))))
