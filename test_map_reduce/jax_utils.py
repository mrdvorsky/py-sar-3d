from contextlib import contextmanager
import time

import jax
from jaxlib import xla_client


@contextmanager
def jax_timer(label="Execution"):
    # Ensure all previous device activity is finished
    jax.effects_barrier()
    start = time.perf_counter()
    try:
        yield
    finally:
        # This is the "secret sauce" for JAX
        # We need to wait for the computation to actually finish
        jax.effects_barrier()
        end = time.perf_counter()
        print(f"{label}: {end - start:.6f} seconds")


def time_it(func, *args, warm_up=True, **kwargs):
    """
    A wrapper to time a JAX function properly.
    """
    if warm_up:
        # Run once to JIT compile without timing
        _ = func(*args, **kwargs).block_until_ready()

    with jax_timer(label=f"Timing {func.__name__}"):
        result = func(*args, **kwargs).block_until_ready()

    return result


def exportGraph(fn: jax.stages.Wrapped, *args, folder="dot_files"):
    lowered = fn.lower(*args)
    compiled = lowered.compile()
    modules = compiled.runtime_executable().hlo_modules()  # type: ignore

    if len(modules) != 1:
        raise ValueError(f"Number of modules should be (1), but is actually ({len(modules)})")

    proto = modules[0].as_serialized_hlo_module_proto()
    compute = xla_client.XlaComputation(proto)

    filename = f"{folder}/{fn.__qualname__}.dot"
    with open(filename, "w") as f:
        f.write(compute.as_hlo_dot_graph())
