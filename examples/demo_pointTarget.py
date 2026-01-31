import jax
import jax.numpy as jnp

from sar import _monostatic_helper


x = jnp.zeros([1])
y = jnp.zeros([1])
k0 = jnp.zeros([1])

tx = jnp.ones([1, 10000])
ty = jnp.ones([20000, 1])
tz = jnp.ones([20000, 1])
tSigma = jnp.ones([1, 1])


print(jax.make_jaxpr(_monostatic_helper)(x, y, k0, tx, ty, tz, tSigma))



print(_monostatic_helper(x, y, k0, tx, ty, tz, tSigma))






