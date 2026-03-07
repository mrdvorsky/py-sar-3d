from itertools import pairwise
from typing import Callable, Sequence

import jax
import jax.numpy as jnp


# TODO: Data type of indices


def _map_reduce_optimize_dims(
    arrays: tuple[jax.Array, ...],
    axis: tuple[int, ...],
) -> tuple[tuple[jax.Array, ...], tuple[int, ...]]:
    return (
        tuple(jnp.moveaxis(a, axis, range(len(axis))) for a in arrays),
        tuple(range(len(axis))),
    )


@jax.jit(static_argnames=("map_fun", "axis", "unroll_count"))
def _map_reduce_core(
    map_fun: Callable[..., jax.Array],
    out_shape_type: jax.ShapeDtypeStruct,
    arrays: tuple[jax.Array, ...],
    axis: tuple[int, ...],
    unroll_count: int,
) -> jax.Array:
    arrays, axis = _map_reduce_optimize_dims(arrays, axis)

    reduce_concrete_shapes = tuple(arr.shape[: len(axis)] for arr in arrays)
    reduce_virtual_shape = jnp.broadcast_shapes(*reduce_concrete_shapes)
    reduce_virtual_subs = jnp.indices(reduce_virtual_shape, sparse=True)

    concrete_linear_inds: list[jax.Array] = []
    for shape in reduce_concrete_shapes:
        concrete_linear_inds.append(
            jnp.ravel(jnp.ravel_multi_index(reduce_virtual_subs, shape, mode="clip"))
        )

    arrays = tuple(jnp.reshape(arr, (-1,) + arr.shape[len(axis) :]) for arr in arrays)

    def _body_fun(carry, inputs):
        fun_args = tuple(
            arr.at[lin_inds].get(mode="promise_in_bounds", wrap_negative_indices=False)
            for lin_inds, arr in zip(inputs, arrays)
        )
        return (carry + map_fun(*fun_args), None)

    return jax.lax.scan(
        _body_fun,
        jnp.zeros(out_shape_type.shape, dtype=out_shape_type.dtype),
        concrete_linear_inds,
        unroll=unroll_count,
    )[0]


def map_reduce_indexing(
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
