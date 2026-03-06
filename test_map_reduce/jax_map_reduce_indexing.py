import math

from typing import Callable, Sequence
import jax
import jax.numpy as jnp


# TODO: Data type of indices


# @jax.jit(static_argnames=("concrete_shapes",))
def _sub_to_linear(
    virtual_subs: jax.Array,
    concrete_shapes: tuple[tuple[int, ...], ...],
) -> jax.Array:
    stride_matrix = []

    for shape in concrete_shapes:
        strides = []
        current_stride = 1
        for dim_size in reversed(shape):
            strides.append(current_stride if dim_size > 1 else 0)
            current_stride *= dim_size
        stride_matrix.append(strides[::-1])

    print("SM", jnp.array(stride_matrix))
    print("VS", virtual_subs)
    return jnp.array(stride_matrix) @ virtual_subs


# @jax.jit(static_argnames=("linear_stride", "virtual_shape"))
def _increment_subs(
    virtual_subs: jax.Array,
    linear_stride: int,
    virtual_shape: tuple[int, ...],
) -> jax.Array:
    print("VS init", virtual_subs)

    current_stride = linear_stride
    strides: list[int] = []
    for dim_size in reversed(virtual_shape):
        strides.append(current_stride % dim_size)
        current_stride //= dim_size
    strides = strides[::-1]

    virtual_subs += jnp.array(strides)[:, None]
    print("VS after", virtual_subs)
    for n in reversed(range(len(strides))):
        needs_wrap = virtual_subs[n] >= virtual_shape[n]
        virtual_subs = virtual_subs.at[n].set(
            jax.lax.select(
                needs_wrap,
                virtual_subs[n] - virtual_shape[n],
                virtual_subs[n],
            )
        )
        virtual_subs = virtual_subs.at[n - 1].set(
            jax.lax.select(
                needs_wrap,
                virtual_subs[n - 1] + 1,
                virtual_subs[n - 1],
            )
        )

    return virtual_subs


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
    reduce_count = math.prod(reduce_virtual_shape)

    arrays = tuple(jnp.reshape(arr, (-1,) + arr.shape[len(axis) :]) for arr in arrays)

    def loop_body(_, carry):
        carry_sum, virtual_subs = carry
        concrete_linear_inds = _sub_to_linear(virtual_subs, reduce_concrete_shapes)
        print("CLI", concrete_linear_inds)
        fun_args = tuple(arr[cind] for cind, arr in zip(concrete_linear_inds, arrays))
        print("FA", fun_args)
        return (
            carry_sum + jnp.sum(map_fun(*fun_args), axis=[0]),
            _increment_subs(virtual_subs, 1, reduce_virtual_shape),
        )

    init_virtual_linear_inds = jnp.arange(1, dtype=jnp.int32)
    init_virtual_subs = jnp.stack(
        jnp.unravel_index(
            init_virtual_linear_inds,
            reduce_virtual_shape,
        )
    )
    print("IVS", init_virtual_subs)

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
