import jax
from jaxlib import xla_client


def exportGraph(filename: str, fn, *args):
    lowered = jax.jit(fn).lower(*args)
    compiled = lowered.compile()
    hlo_module = compiled.runtime_executable().hlo_modules()[0]
    dotFile = xla_client._xla.hlo_module_to_dot_graph(hlo_module)
    with open(filename, "w") as f:
        f.write(dotFile)
