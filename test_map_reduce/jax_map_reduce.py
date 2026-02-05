from typing import Callable, Sequence
import jax
import jax.numpy as jnp


# TODO: Try linear indexing vs. multi-scan


def _map_reduce_recursion(
    map_fun: Callable[..., jax.Array],
    out_shape: jax.ShapeDtypeStruct,
    arrays: tuple[jax.Array, ...],
    unroll_count: int,
) -> jax.Array:
    if out_shape.ndim == max(a.ndim for a in arrays):
        return map_fun(*arrays)

    arrays_bcast_squeezed = list(
        (ind, jnp.squeeze(a, axis=0))
        for ind, a in enumerate(arrays)
        if (a.shape[0] == 1)
    )

    arrays_full = tuple(a for a in arrays if (a.shape[0] != 1))
    arrays_full_inds = tuple(ind for ind, a in enumerate(arrays) if (a.shape[0] != 1))

    def loop_body(carry, loop_arrays):
        all_args = sorted(
            list(zip(arrays_full_inds, loop_arrays)) + arrays_bcast_squeezed
        )
        return (
            carry
            + _map_reduce_recursion(
                map_fun,
                out_shape,
                tuple(a for _, a in all_args),
                unroll_count,
            ),
            None,
        )

    unrollFactor = 1
    if (out_shape.ndim + 1) == max(a.ndim for a in arrays):
        unrollFactor = unroll_count

    return jax.lax.scan(
        loop_body,
        jnp.zeros(shape=out_shape.shape, dtype=out_shape.dtype),
        arrays_full,
        unroll=unrollFactor,
    )[0]


def _map_reduce_optimize_dims(
    arrays: tuple[jax.Array, ...],
    axis: tuple[int, ...],
) -> tuple[jax.Array, ...]:
    return tuple(jnp.moveaxis(a, axis, range(len(axis))) for a in arrays)


@jax.jit(static_argnames=("map_fun", "axis", "unroll_count"))
def _map_reduce_core(
    map_fun: Callable[..., jax.Array],
    out_shape: jax.ShapeDtypeStruct,
    arrays: tuple[jax.Array, ...],
    axis: tuple[int, ...],
    unroll_count: int,
) -> jax.Array:
    arrays_optimized = _map_reduce_optimize_dims(arrays, axis=axis)
    return _map_reduce_recursion(map_fun, out_shape, arrays_optimized, unroll_count)


def map_reduce(
    map_fun: Callable[..., jax.Array],
    *arrays: jax.Array,
    axis: int | Sequence[int] | None = None,
    unroll_count: int = 16,
) -> jax.Array:
    out_shape: jax.ShapeDtypeStruct = jax.eval_shape(
        lambda a_all: jnp.sum(map_fun(*a_all), axis=axis),
        arrays,
    )

    if axis is None:
        axis = range(max(*(a.ndim for a in arrays)))
    elif isinstance(axis, int):
        axis = (axis,)

    return _map_reduce_core(map_fun, out_shape, arrays, tuple(axis), unroll_count)
