"""
meas/rhf_rh.py: the RHF local energy on the restricted hamiltonian scanned over chunks
of cholesky vectors.

Checks:
  - restricted and unrestricted walkers: the chunked kernels reproduce meas/rhf.py's
    kernels (every vector at once) on unstructured random walkers, for chunks that
    divide the cholesky index and ones that zero pad it;
  - the restricted kernel reproduces afqmc's rhf._calc_energy_restricted written out
    term by term;
  - the mixed recipe's rhf guide gets these kernels with the trial's chunk.
"""

from trot import config

config.configure_once()

import contextlib
import io
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.core.ops import k_energy
from trot.core.system import System
from trot.meas.rhf import build_meas_ctx, energy_kernel_rw_rh as energy_rw_at_once
from trot.meas.rhf import energy_kernel_uw_rh as energy_uw_at_once
from trot.meas.rhf import make_rhf_meas_ops
from trot.meas.rhf_rh import energy_kernel_rw_rh, energy_kernel_uw_rh, make_rhf_meas_ops_rh
from trot.testing import make_random_ham_chol
from trot.trial.rhf import RhfTrial

jax.config.update("jax_enable_x64", True)


def _energy_afqmc_reference(walker, ham, trial, nchol_chunk):
    """afqmc's wavefunctions_restricted.rhf._calc_energy_restricted, as written there."""
    nocc = trial.mo_coeff.shape[1]
    cH = trial.mo_coeff.conj().T
    rot_h1 = (cH @ ham.h1)[:nocc, :]
    rot_chol = jnp.einsum("pi,gij->gpj", cH, ham.chol)[:, :nocc, :]
    green = (walker.dot(jnp.linalg.inv(trial.mo_coeff.T.conj() @ walker))).T
    e1 = 2 * jnp.einsum("pq,pq->", rot_h1, green)
    nchol = rot_chol.shape[0]
    nchunks = -(-nchol // nchol_chunk)
    pad = nchunks * nchol_chunk - nchol
    rot_chol = jnp.pad(rot_chol, ((0, pad), (0, 0), (0, 0)))
    chunks = rot_chol.reshape(nchunks, nchol_chunk, nocc, ham.chol.shape[1])

    def scanned_fun(carry, chol_c):
        lg_c = jnp.einsum("gpr,qr->gpq", chol_c, green)
        tr_c = jnp.einsum("gpp->g", lg_c)
        return carry + 2 * jnp.sum(tr_c**2) - jnp.einsum("gpq,gqp->", lg_c, lg_c), 0.0

    e2, _ = jax.lax.scan(scanned_fun, 0.0, chunks)
    return ham.h0 + e1 + e2


@pytest.mark.parametrize("norb, nocc, nchol", [(6, 3, 7), (9, 4, 12), (5, 1, 3), (8, 7, 5)])
@pytest.mark.parametrize("walker_kind", ["near_reference", "orthonormal", "gaussian"])
def test_chunked_energy_matches_rhf_kernels(norb, nocc, nchol, walker_kind):
    key = jax.random.PRNGKey(norb * 100 + nocc * 10 + nchol)
    k_ham, k_mo, k_w = jax.random.split(key, 3)
    ham = make_random_ham_chol(k_ham, norb=norb, n_chol=nchol)
    mo = jnp.linalg.qr(jax.random.normal(k_mo, (norb, nocc)))[0]
    trial = RhfTrial(mo_coeff=mo)
    ctx = build_meas_ctx(ham, trial)
    n_walkers = 12
    rng = np.random.default_rng(norb + nchol)

    def walkers():
        x = rng.normal(size=(n_walkers, norb, nocc)) + 1j * rng.normal(size=(n_walkers, norb, nocc))
        if walker_kind == "orthonormal":
            x = np.linalg.qr(x)[0]
        elif walker_kind == "near_reference":
            x = np.asarray(mo)[None] + 0.2 * x
        return jnp.asarray(x)

    w, w_dn = walkers(), walkers()
    e_ref = jax.vmap(energy_rw_at_once, (0, None, None, None))(w, ham, ctx, trial)
    e_ref_u = jax.vmap(energy_uw_at_once, (0, None, None, None))((w, w_dn), ham, ctx, trial)
    # chunks that divide nchol, ones that pad the last chunk, and the whole index at once
    for nchol_chunk in (1, 2, 3, nchol - 1, nchol, None):
        if nchol_chunk is not None and nchol_chunk < 1:
            continue
        e = jax.vmap(partial(energy_kernel_rw_rh, nchol_chunk=nchol_chunk), (0, None, None, None))(
            w, ham, ctx, trial
        )
        np.testing.assert_allclose(np.asarray(e), np.asarray(e_ref), rtol=1e-11)
        e_u = jax.vmap(
            partial(energy_kernel_uw_rh, nchol_chunk=nchol_chunk), (0, None, None, None)
        )((w, w_dn), ham, ctx, trial)
        np.testing.assert_allclose(np.asarray(e_u), np.asarray(e_ref_u), rtol=1e-11)
        # an unrestricted walker with equal spins is the restricted one
        e_uu = jax.vmap(
            partial(energy_kernel_uw_rh, nchol_chunk=nchol_chunk), (0, None, None, None)
        )((w, w), ham, ctx, trial)
        np.testing.assert_allclose(np.asarray(e_uu), np.asarray(e_ref), rtol=1e-11)
        if nchol_chunk is not None:
            e_afqmc = jax.vmap(_energy_afqmc_reference, (0, None, None, None))(
                w, ham, trial, nchol_chunk
            )
            np.testing.assert_allclose(np.asarray(e), np.asarray(e_afqmc), rtol=1e-11)


def test_meas_ops_keep_everything_but_the_energy():
    for walker_kind in ("restricted", "unrestricted", "generalized"):
        sys = System(norb=6, nelec=(3, 3), walker_kind=walker_kind)
        base = make_rhf_meas_ops(sys)
        ops = make_rhf_meas_ops_rh(sys, nchol_chunk=4)
        assert ops.overlap is base.overlap
        assert set(ops.observables) == set(base.observables)
        assert set(ops.kernels) == set(base.kernels)
        if walker_kind == "generalized":
            assert not ops.has_kernel(k_energy)
        else:
            kernel = ops.require_kernel(k_energy)
            assert isinstance(kernel, partial) and kernel.keywords == {"nchol_chunk": 4}


def test_mixed_rhf_guide_runs_the_chunked_energy():
    from pyscf import cc, gto, scf

    from trot.afqmc import AfqmcMixed

    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.4; H 0 0 2.8; H 0 0 4.2", basis="sto-6g", unit="b", verbose=0
    )
    mf = scf.RHF(mol)
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.kernel()
    af = AfqmcMixed(mycc, guide="rhf", trial="pt2ccsd_bar", n_walkers=6, n_blocks=2, nchol_chunk=3)
    with contextlib.redirect_stdout(io.StringIO()):
        job = af.build_job()
    kernel = job.meas_ops.require_kernel(k_energy)
    assert isinstance(kernel, partial) and kernel.func is energy_kernel_rw_rh
    assert kernel.keywords == {"nchol_chunk": 3} and job.guide_nchol_chunk == 3
