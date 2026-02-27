from functools import partial
from math import prod

import jax
import jax.numpy as jnp

from jax_utils import export_graph


@partial(jax.jit, static_argnums=(0, ))
def calcululate1(shape: tuple[int, ...]):
    # inds = jax.lax.iota(jnp.float32, prod(shape))
    inds = jnp.arange(0, prod(shape))
    inds = jnp.reshape(inds, shape)
    return jnp.sum(jnp.hypot(9.9, inds))

# print(jax.make_jaxpr(calcululate1)(jnp.empty([1000000])))


export_graph("dot_files/single1.dot", calcululate1, (16, 16, 16, 16, 16, 16, 16, 16, 16))

print(calcululate1((16, 16, 16, 16, 16, 16, 16, 16, 16)))


