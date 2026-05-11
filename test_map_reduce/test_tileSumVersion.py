
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
    x = jnp.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]*x.shape[3]))
    y = jnp.reshape(y, (y.shape[0], y.shape[1]*y.shape[2]*y.shape[3]))
    return map_reduce(_kernel, x, y, axis=[1], unroll_count=32)

@jax.jit
def test2(x, y):
    return jnp.sum(_kernel(x, y), axis=[1, 2, 3])



key = jax.random.PRNGKey(10)
x = jax.random.normal(key, [1, 32, 8, 1])
y = jax.random.normal(key, [32*32*32*32*32*32, 1, 1, 1])



export_graph(test1, x, y)
export_graph(test2, x, y)

time_it(test1, x, y)
time_it(test2, x, y)




