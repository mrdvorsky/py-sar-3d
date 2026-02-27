import jax
import jax.numpy as jnp
from functools import partial


def _kernel_oneWay_scalar(x, y, k0, tx, ty, tz, tSigma):
    R = jnp.hypot(jnp.hypot(x - tx, y - ty), tz)
    return tSigma * jnp.exp(1j * k0 * R) / R


@partial(jnp.vectorize, excluded=(3, 4, 5, 6))
def _monostatic_helper(x, y, k0, tx, ty, tz, tSigma):
    return map_sum(_kernel_oneWay_scalar, x, y, k0, tx, ty, tz, tSigma)


def simulateSarMonostatic(x, y, k0, tx, ty, tz, tSigma):
    return _monostatic_helper(x, y, k0, tx, ty, tz, tSigma)
