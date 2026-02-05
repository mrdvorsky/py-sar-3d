
import jax
import jax.numpy as jnp
import matplotlib.pylab as plt
from sar import simulateSarMonostatic



key = jax.random.PRNGKey(758493)


x = jnp.linspace(-10, 10, 100)[:, None, None]
y = jnp.linspace(-20, 20, 200)[None, :, None]
k0 = jnp.linspace(10, 11, 1)[None, None, :]

tx = jax.random.normal(key, [5, 1, 1000])
ty = jax.random.normal(key, [1, 3])
tz = jnp.array([-5])
tSigma = jax.random.normal(key, [1, 3])


Img = simulateSarMonostatic(x, y, k0, tx, ty, tz, tSigma)
print(Img.shape)


plt.imshow(jnp.real(Img[:, :, 1]), interpolation="none")
plt.show()




