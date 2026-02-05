from typing import Callable, Sequence
import jax
import jax.numpy as jnp

# TODO: Try linear indexing vs. multi-scan


def _map_reduce_recursion():
    pass


@jax.jit(static_argnames=("map_fun", "axis"))
def _map_reduce_core(
    map_fun: Callable[..., jax.Array],
    *args: jax.Array,
    axis: tuple[int, ...],
) -> jax.Array:
    shape = jnp.broadcast_shapes(*[a.shape for a in args])
    args = tuple(jnp.moveaxis(a, axis, range(len(axis))) for a in args)

    print(*args, sep="\n")
    return


def _map_reduce_checker(
    map_fun: Callable[..., jax.Array],
    *args: jax.Array,
    axis: int | Sequence[int] | None = None,
) -> jax.Array:
    return jnp.sum(map_fun(*args), axis=axis)


def map_reduce(
    map_fun: Callable[..., jax.Array],
    *args: jax.Array,
    axis: int | Sequence[int] | None = None,
) -> jax.Array:
    jax.eval_shape(_map_reduce_checker, map_fun, *args, axis=axis)

    if axis is None:
        axis = range(max(*[len(a.shape) for a in args]))
    elif isinstance(axis, int):
        axis = (axis,)

    return _map_reduce_core(map_fun, *args, axis=tuple(axis))


def _kernelFun(a, b):
    return a + b


map_reduce(_kernelFun, jnp.zeros([2, 3, 4, 5]), jnp.zeros([1, 3, 1, 5]), axis=[2, 3])
