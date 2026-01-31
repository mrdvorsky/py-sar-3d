import jax
import jax.numpy as jnp
from functools import partial


def broadcast_scan(f, init, *xs):
    """
    Scan over the cartesian product of arrays with broadcastable shapes.
    Uses nested scans for efficiency.
    """
    if len(xs) == 0:
        return init, None
    if len(xs) == 1:
        return jax.lax.scan(lambda carry, x: f(carry, x), init, jnp.array(xs[0]).ravel())
    x_first = jnp.array(xs[0]).ravel()

    def outer_fn(carry, elem):
        def inner_fn(carry_inner, *rest_elems):
            return f(carry_inner, elem, *rest_elems)

        return broadcast_scan(inner_fn, carry, *xs[1:])

    return jax.lax.scan(outer_fn, init, x_first)


def map_sum(map_fun, *args):
    def _map_sum_helper(carry, *args):
        return (carry + map_fun(*args), None)

    return broadcast_scan(_map_sum_helper, 0, *args)[0]


def _kernel_oneWay_scalar(x, y, k0, tx, ty, tz, tSigma):
    R = jnp.hypot(jnp.hypot(x - tx, y - ty), tz)
    return tSigma * jnp.exp(1j * k0 * R) / R


@partial(jnp.vectorize, excluded=(3, 4, 5, 6))
def _monostatic_helper(x, y, k0, tx, ty, tz, tSigma):
    print(x.shape)
    print(y.shape)
    print(k0.shape)
    print(tx.shape)
    print(ty.shape)
    print(tz.shape)
    print(tSigma.shape)
    return map_sum(_kernel_oneWay_scalar, x, y, k0, tx, ty, tz, tSigma)


def simulateSarMonostatic(x, y, k0, tx, ty, tz, tSigma):
    print(x.shape)
    print(y.shape)
    print(k0.shape)
    print(tx.shape)
    print(ty.shape)
    print(tz.shape)
    print(tSigma.shape)
    return _monostatic_helper(x, y, k0, tx, ty, tz, tSigma)
