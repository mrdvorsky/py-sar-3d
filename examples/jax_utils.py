import jax
from jaxlib import xla_client
import time
from contextlib import contextmanager

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

def exportGraph(filename: str, fn, *args):
    lowered = jax.jit(fn).lower(*args)
    compiled = lowered.compile()
    hlo_modules = compiled.runtime_executable().hlo_modules()
    # print(len(hlo_modules))
    dotFile = xla_client._xla.hlo_module_to_dot_graph(hlo_modules[0])
    with open(filename, "w") as f:
        f.write(dotFile)

