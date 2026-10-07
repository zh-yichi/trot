"""
RHF measurement kernels on the restricted hamiltonian with the local energy scanned over
chunks of cholesky vectors (kernel suffix _rw_rh / _uw_rh: restricted / unrestricted
walker on the restricted hamiltonian, as meas/rhf.py names them).

meas/rhf.py sums the two body term over every cholesky vector at once (memory_mode
"high", a (n_chol, nocc, nocc) intermediate per walker) or over fori_loop batches
("low"). Here the vectors come in (n_chunks, chunk, nocc, norb) chunks (cholesky.
chunk_rot_chol, zero padded) and the energy is a lax.scan of plain einsums over them,
as meas/uhf_uh.py does on the unrestricted hamiltonian and as afqmc's
wavefunctions_restricted.rhf._calc_energy_restricted does:

    lg   (k, nocc, nocc)   L_g,pr G_qr                     one chunk of k vectors
    e2   2 sum_g (tr lg_g)^2 - sum_g tr(lg_g lg_g)         restricted walker

so the peak memory per walker is set by the chunk, not by n_chol. A zero padded vector
contributes nothing. The half green's function, the context (meas.rhf.RhfMeasCtx: the
half rotated h1 and chol), the overlap, the force bias and the observables are
meas/rhf.py's; only the energy kernels differ. For an unrestricted walker the energy is
meas.uhf_uh.u_rot_energy with the one rotated hamiltonian serving both spins.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

from ..cholesky import chunk_rot_chol
from ..core.ops import MeasOps, k_energy
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.rhf import RhfTrial
from .rhf import RhfMeasCtx, _half_green_from_overlap_matrix, make_rhf_meas_ops
from .uhf_uh import u_rot_energy

# cholesky vectors per lax.scan step in the energy kernels; None sums them in one chunk
DEFAULT_NCHOL_CHUNK: int | None = None


def rot_energy_r(
    green: jax.Array, h0: jax.Array, rot_h1: jax.Array, rot_chol: jax.Array
) -> jax.Array:
    """
    Local energy <T|H|phi>/<T|phi> of a restricted walker against the half rotated
    restricted hamiltonian, from its half green's function (nocc, norb).

    rot_chol is (n_chunks, chunk, nocc, norb); a plain (n_chol, nocc, norb) is accepted
    and treated as a single chunk. The two body term is reduced with lax.scan over the
    chunks. Both spins share green, so the coulomb term carries the factor 2 of the
    doubly occupied orbitals squared and the exchange term the factor 2 once.
    """
    if rot_chol.ndim == 3:
        rot_chol = rot_chol.reshape(1, *rot_chol.shape)

    e1 = 2.0 * jnp.einsum("pq,pq->", rot_h1, green, optimize="optimal")

    zero = jnp.array(0.0, dtype=jnp.result_type(rot_chol, green))

    def scanned_fun(carry: jax.Array, chol_c: jax.Array) -> tuple[jax.Array, None]:
        lg_c = jnp.einsum("gpr,qr->gpq", chol_c, green, optimize="optimal")  # (chunk, nocc, nocc)
        tr_c = jnp.einsum("gpp->g", lg_c, optimize="optimal")
        e2_c = 2.0 * jnp.sum(tr_c * tr_c) - jnp.einsum("gpq,gqp->", lg_c, lg_c, optimize="optimal")
        return carry + e2_c, None

    e2, _ = lax.scan(scanned_fun, zero, rot_chol)

    return h0 + e1 + e2


def energy_kernel_rw_rh(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: RhfMeasCtx,
    trial_data: RhfTrial,
    *,
    nchol_chunk: int | None = DEFAULT_NCHOL_CHUNK,
) -> jax.Array:
    m = trial_data.mo_coeff.conj().T @ walker
    green = _half_green_from_overlap_matrix(walker, m)  # (nocc, norb)
    return rot_energy_r(
        green, ham_data.h0, meas_ctx.rot_h1, chunk_rot_chol(meas_ctx.rot_chol, nchol_chunk)
    )


def energy_kernel_uw_rh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamChol,
    meas_ctx: RhfMeasCtx,
    trial_data: RhfTrial,
    *,
    nchol_chunk: int | None = DEFAULT_NCHOL_CHUNK,
) -> jax.Array:
    bra = (trial_data.mo_coeff, trial_data.mo_coeff)
    rot_chol = chunk_rot_chol(meas_ctx.rot_chol, nchol_chunk)
    return u_rot_energy(
        bra, walker, ham_data.h0, (meas_ctx.rot_h1, meas_ctx.rot_h1), (rot_chol, rot_chol)
    )


def make_rhf_meas_ops_rh(sys: System, *, nchol_chunk: int | None = DEFAULT_NCHOL_CHUNK) -> MeasOps:
    """
    meas.rhf.make_rhf_meas_ops with the local energy scanned over chunks of nchol_chunk
    cholesky vectors (None: all at once). Everything else (overlap, force bias, context,
    observables) is the same object's. A generalized walker has no RHF energy kernel
    there either.
    """
    base = make_rhf_meas_ops(sys, memory_mode="high")
    wk = sys.walker_kind.lower()
    kernels: dict[str, Any] = dict(base.kernels)
    if wk == "restricted":
        kernels[k_energy] = partial(energy_kernel_rw_rh, nchol_chunk=nchol_chunk)
    elif wk == "unrestricted":
        kernels[k_energy] = partial(energy_kernel_uw_rh, nchol_chunk=nchol_chunk)
    return MeasOps(
        overlap=base.overlap,
        build_meas_ctx=base.build_meas_ctx,
        kernels=kernels,
        observables=base.observables,
    )
