import math

from typing import Callable, Sequence
import jax
import jax.numpy as jnp


# TODO: Data type of indices


# @jax.jit(static_argnames=("concrete_shapes",))
def _sub_to_linear(
    virtual_subs: tuple[jax.Array, ...],
    concrete_shapes: tuple[tuple[int, ...], ...],
) -> tuple[jax.Array, ...]:
    ret: list[jax.Array] = []

    for shape in concrete_shapes:
        strides: list[int] = []
        current_stride = 1
        for dim_size in reversed(shape):
            strides.append(current_stride if dim_size > 1 else 0)
            current_stride *= dim_size
        ret.append(
            sum(sub * stride for sub, stride in zip(virtual_subs, strides[::-1]))  # type: ignore
        )

    return tuple(ret)


# @jax.jit(static_argnames=("linear_stride", "virtual_shape"))
def _increment_subs(
    virtual_subs: tuple[jax.Array, ...],
    linear_stride: int,
    virtual_shape: tuple[int, ...],
) -> tuple[jax.Array, ...]:
    current_stride = linear_stride
    strides: list[int] = []
    for dim_size in reversed(virtual_shape):
        strides.append(current_stride % dim_size)
        current_stride //= dim_size
    strides = strides[::-1]

    new_virtual_subs = list(sub + stride for sub, stride in zip(virtual_subs, strides))
    for n in reversed(range(len(strides))):
        needs_wrap = new_virtual_subs[n] >= virtual_shape[n]
        new_virtual_subs[n] = jax.lax.select(
            needs_wrap,
            new_virtual_subs[n] - virtual_shape[n],
            new_virtual_subs[n],
        )
        new_virtual_subs[n - 1] = jax.lax.select(
            needs_wrap,
            new_virtual_subs[n - 1] + 1,
            new_virtual_subs[n - 1],
        )

    return tuple(new_virtual_subs)


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

    print(arrays, axis)

    reduce_concrete_shapes = tuple(arr.shape[: len(axis)] for arr in arrays)
    reduce_virtual_shape = jnp.broadcast_shapes(*reduce_concrete_shapes)
    reduce_count = math.prod(reduce_virtual_shape)

    arrays = tuple(jnp.reshape(arr, (-1,) + arr.shape[len(axis) :]) for arr in arrays)
    print(reduce_virtual_shape)

    def loop_body(linear_ind, carry):
        carry_sum, virtual_subs = carry
        new_concrete_linear_inds = _sub_to_linear(virtual_subs, reduce_concrete_shapes)

        fun_args = tuple(
            arr[cind] for cind, arr in zip(new_concrete_linear_inds, arrays)
        )
        new_virtual_subs = _increment_subs(virtual_subs, 1, reduce_virtual_shape)

        return (
            carry_sum + jnp.sum(map_fun(*fun_args), axis=[0]),
            new_virtual_subs,
        )

    init_virtual_linear_inds = jnp.arange(1, dtype=jnp.int32)
    init_virtual_subs = jnp.unravel_index(
        init_virtual_linear_inds,
        reduce_virtual_shape,
    )

    return jax.lax.fori_loop(
        0,
        reduce_count,
        loop_body,
        (
            jnp.zeros(out_shape_type.shape, dtype=out_shape_type.dtype),
            init_virtual_subs,
        ),
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
