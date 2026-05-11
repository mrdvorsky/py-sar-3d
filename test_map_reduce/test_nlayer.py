import jax
import jax.numpy as jnp


from jax_map_reduce import map_reduce
from jax_utils import time_it, export_graph


M = 16
N = 300
Nf = 1000


def _kernel(kr, k):
    return jnp.exp(-1j * 10 * k * kr)


@jax.jit
def calculateSingle(Ai, kr, k):
    gamVal = _kernel(kr, k)
    # A = jnp.sum(Ai * gamVal, axis=[0])
    A = jnp.vecdot(Ai, gamVal, axis=-1)
    idMat = jnp.eye(A.shape[-1])
    # print(idMat, A)

    Etop = idMat - A
    Ebot = idMat + A
    Ebot = Ebot[:, :, 0, None]
    # print(Ebot)

    Smn = jnp.linalg.solve(Etop, Ebot)
    print(Smn)

    return Smn[0, 0]


key = jax.random.PRNGKey(10)
kr = jax.random.normal(key, [1, 1, 1, N])
k = jax.random.normal(key, [Nf, 1, 1, 1])
Ai = jax.random.normal(key, [1, M, M, N])


time_it(calculate, Ai, kr, k, num_reps=1)
export_graph(calculate, Ai, kr, k)
