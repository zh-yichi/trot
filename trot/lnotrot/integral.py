from __future__ import annotations

import time
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
from pyscf import lib, scf

from ..cholesky import _core_uveff, joint_df2chol
from ..staging import HamInput, HamInputU

print = partial(print, flush=True)

# The fragment hamiltonian, term for term as afqmc's lno_afqmc/integral.py
# (get_lno_integral_joint, called from prep_lno_integral):
#
#   1. the effective core energy and one-electron integrals come from the frozen
#      occupied LNOs, with J and K built from the DF tensor on the device
#      (afqmc h1e_ras / h1e_uas, here lno_effective_core);
#   2. the cholesky vectors of the local active space are compressed straight from the
#      whole system's DF 3-index tensor mf.with_df, rotated into the active space one aux
#      block at a time (cderi2mo) and then decomposed there: a pivoted modified cholesky
#      of the pair Gram matrix for a restricted fragment (df2chol_gpu), and for an
#      unrestricted one trot.cholesky.joint_df2chol over both spins' pair spaces, so the
#      two spins share one auxiliary index as HamCholU needs.
#
# mf must be density fitted. The orbitals come ordered [frz_occ | act_occ | act_vir |
# frz_vir] from las.make_las, so the core is the leading ncore columns and the active
# block is contiguous; get_las_idx checks that instead of assuming it.


# --------------------------------------------------------------------------- partition


def _las_idx_one(nao: int, nocc: int, frozen: Any) -> tuple[int, int, int, NDArray]:
    frozen = set(int(i) for i in (np.atleast_1d(frozen) if not isinstance(frozen, int) else []))
    actfrag = np.array([i for i in range(nao) if i not in frozen], dtype=np.int64)
    nfrzocc = sum(1 for i in range(nocc) if i in frozen)
    nactocc = sum(1 for i in range(nocc) if i not in frozen)
    ncas = int(actfrag.size)
    if not np.array_equal(actfrag, np.arange(nfrzocc, nfrzocc + ncas)):
        raise ValueError(
            "the fragment orbitals must be ordered [frozen occ | active | frozen vir] so that "
            "the core is the leading block; got frozen indices "
            f"{sorted(frozen)} for nocc={nocc}, nao={nao}."
        )
    return nfrzocc, nactocc, ncas, actfrag


def get_las_idx(mf: Any, lno_frozen: Any):
    """
    (ncore, nocc, ncas, actfrag) of the fragment: frozen occupied count, active occupied
    count, active orbital count and the active indices. Each is a pair for a UHF mf.
    """
    nao = mf.mol.nao
    if isinstance(mf, scf.uhf.UHF):
        out = [
            _las_idx_one(nao, int(np.count_nonzero(mf.mo_occ[s])), lno_frozen[s]) for s in range(2)
        ]
        return tuple(tuple(o[k] for o in out) for k in range(4))
    if isinstance(mf, scf.rhf.RHF):
        return _las_idx_one(nao, int(np.count_nonzero(mf.mo_occ)), lno_frozen)
    raise TypeError(f"unsupported mean-field type: {type(mf)}")


# --------------------------------------------------------------------------- core


def _require_df_x64(mf: Any) -> None:
    if getattr(mf, "with_df", None) is None:
        raise NotImplementedError(
            "LNO-AFQMC builds the fragment integrals from the density fitting tensor; "
            "use a density fitted mean field (mf.density_fit())."
        )
    if not jax.config.read("jax_enable_x64"):
        raise RuntimeError(
            "jax_enable_x64 is off: the fragment integrals would be built in single precision. "
            "Call trot.config.configure_once() (importing trot.afqmc does) before staging."
        )


def lno_effective_core(mf: Any, core: Any, act: Any) -> tuple[float, Any]:
    """
    E_core and h1eff of the fragment, exactly as afqmc's h1e_ras / h1e_uas.

    core / act are the frozen occupied and the active LNO coefficients, (C_c, C_a) for a
    restricted mf or ((C_c^a, C_c^b), (C_a^a, C_a^b)) for a UHF one. With
    D^s = C_c^s C_c^s^T and V^s = J[D^a + D^b] - K[D^s] (trot.cholesky._core_uveff, the
    DF tensor on the device):

        E_core = E_nuc + sum_s tr(D^s h) + 1/2 sum_s tr(D^s V^s)
        h1eff^s = C_a^s^T (h + V^s) C_a^s

    which for D^a = D^b = D is the restricted E_nuc + 2 tr(D h) + tr(D V), V = 2J - K.
    """
    _require_df_x64(mf)
    hcore = np.asarray(mf.get_hcore())
    e_core = float(mf.energy_nuc())

    if isinstance(mf, scf.uhf.UHF):
        cores = [np.asarray(c) for c in core]
        acts = [np.asarray(c) for c in act]
        if cores[0].shape[1] == 0 and cores[1].shape[1] == 0:
            veff = (0.0, 0.0)
        else:
            dms = [c @ c.conj().T for c in cores]
            veff = _core_uveff(mf, dms[0], dms[1])
            for s in range(2):
                e_core += float(np.einsum("ij,ji->", dms[s], hcore).real)
                e_core += 0.5 * float(np.einsum("ij,ji->", dms[s], veff[s]).real)
        h1eff = tuple(acts[s].conj().T @ (hcore + veff[s]) @ acts[s] for s in range(2))
        return e_core, h1eff

    core = np.asarray(core)
    act = np.asarray(act)
    if core.shape[1] == 0:
        veff = 0.0
    else:
        dm = core @ core.conj().T
        veff = _core_uveff(mf, dm, dm)[0]  # J[2D] - K[D]
        e_core += 2.0 * float(np.einsum("ij,ji->", dm, hcore).real)
        e_core += float(np.einsum("ij,ji->", dm, veff).real)
    h1eff = act.conj().T @ (hcore + veff) @ act
    return e_core, h1eff


# --------------------------------------------------------------------------- cholesky


@jax.jit
def cderi2mo(cderi: jax.Array, coeff: jax.Array) -> jax.Array:
    """One (blk, nao, nao) DF block rotated into an orbital set and packed, (blk, npair)."""
    cderi_mo = jnp.einsum("pr,grs,sq->gpq", coeff.T, cderi, coeff, optimize="optimal")
    n = coeff.shape[1]
    rows, cols = jnp.tril_indices(n)
    return cderi_mo[:, rows, cols]


@jax.jit
def df2chol_gpu(dferi: jax.Array, max_error: float = 1e-6) -> tuple[jax.Array, jax.Array]:
    """
    Pivoted modified cholesky of the pair Gram matrix dferi^T dferi, compiled.

    Returns a zero padded (n_aux, norb, norb) array of vectors plus the number that are
    valid, so the caller slices [:nchol] outside the jit. Same algorithm and stopping
    rule as trot.cholesky.df2chol (numpy).
    """
    n_aux, n_pair = dferi.shape
    diag = jnp.sum(dferi**2, axis=0)
    norb = int(((-1 + (1 + 8 * n_pair) ** 0.5) / 2))

    chol_vecs = jnp.zeros((n_aux, n_pair))
    m_approx = jnp.zeros(n_pair)
    init_state = (0, chol_vecs, m_approx, diag, jnp.max(diag))

    def cond_fun(state):
        nchol, _, _, _, max_val = state
        return jnp.logical_and(nchol < n_aux, max_val >= max_error)

    def body_fun(state):
        nchol, chol_vecs_loop, m_approx_loop, diag_res_loop, _ = state
        nu = jnp.argmax(diag_res_loop)
        delta_max = diag_res_loop[nu]
        row_nu = jnp.dot(dferi.T, dferi[:, nu])
        r = jnp.dot(chol_vecs_loop[:, nu], chol_vecs_loop)
        new_vec = (row_nu - r) / jnp.sqrt(jnp.maximum(delta_max, 1e-12))
        chol_vecs_loop = chol_vecs_loop.at[nchol].set(new_vec)
        m_approx_loop = m_approx_loop + new_vec**2
        diag_res_loop = jnp.abs(diag - m_approx_loop)
        return (nchol + 1, chol_vecs_loop, m_approx_loop, diag_res_loop, jnp.max(diag_res_loop))

    final_nchol, final_chol_vecs, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)

    chol_out = jnp.zeros((n_aux, norb, norb))
    row_idx, col_idx = jnp.tril_indices(norb)
    chol_out = chol_out.at[:, row_idx, col_idx].set(final_chol_vecs)
    chol_out = chol_out.at[:, col_idx, row_idx].set(final_chol_vecs)
    return chol_out, final_nchol


def active_df(mf: Any, coeffs: list[NDArray]) -> list[NDArray]:
    """
    The DF tensor of mf rotated and packed into each orbital set of coeffs, one aux
    block at a time on the device: a (naux, n(n+1)/2) array per set.
    """
    naux = int(mf.with_df.get_naoaux())
    coeffs_j = [jnp.asarray(c) for c in coeffs]
    outs = [np.zeros((naux, c.shape[1] * (c.shape[1] + 1) // 2)) for c in coeffs]
    p1 = 0
    for cderi in mf.with_df.loop():
        cderi = jnp.asarray(lib.unpack_tril(cderi, axis=-1))
        p0, p1 = p1, p1 + cderi.shape[0]
        for out, c in zip(outs, coeffs_j):
            out[p0:p1] = np.asarray(cderi2mo(cderi, c))
    if p1 != naux:
        raise RuntimeError(f"DF iterator yielded {p1} auxiliaries; expected {naux}.")
    return outs


def _sym(h1: Any) -> NDArray:
    h1 = np.asarray(h1)
    return 0.5 * (h1 + h1.T.conj())


def build_ham_lno_df(mf: Any, lno_coeff: Any, lno_frozen: Any, *, chol_cut: float) -> HamInput:
    """The restricted fragment hamiltonian (afqmc get_lno_integral_joint, RHF branch)."""
    _require_df_x64(mf)
    lno_coeff = np.asarray(lno_coeff)
    ncore, nocc, ncas, actfrag = get_las_idx(mf, lno_frozen)
    print(f"[lnotrot] fragment space: nocc={nocc} ncas={ncas} ncore={ncore}")

    t0 = time.time()
    h0, h1 = lno_effective_core(mf, lno_coeff[:, :ncore], lno_coeff[:, actfrag])
    print(f"[lnotrot] effective core and h1 in {time.time() - t0:.2f}s (E_core={h0:.10f})")

    t0 = time.time()
    (cderi_las,) = active_df(mf, [lno_coeff[:, actfrag]])
    print(f"[lnotrot] DF tensor in the active space {cderi_las.shape} in {time.time() - t0:.2f}s")

    t0 = time.time()
    chol_full, nchol = df2chol_gpu(jnp.asarray(cderi_las), max_error=chol_cut)
    nchol = int(nchol)
    chol = np.asarray(chol_full[:nchol])
    print(f"[lnotrot] cholesky: nchol={nchol} (cut {chol_cut:g}) in {time.time() - t0:.2f}s")

    return HamInput(
        h0=float(h0),
        h1=_sym(h1),
        chol=chol,
        nelec=(int(nocc), int(nocc)),
        norb=int(ncas),
        chol_cut=float(chol_cut),
        frozen=np.asarray(lno_frozen, dtype=np.int64).reshape(-1)
        if not isinstance(lno_frozen, int)
        else np.zeros((0,), dtype=np.int64),
        source_kind="mf",
        basis="restricted",
    )


def build_ham_ulno_df(mf: Any, lno_coeff: Any, lno_frozen: Any, *, chol_cut: float) -> HamInputU:
    """
    The unrestricted fragment hamiltonian (afqmc get_lno_integral_joint, UHF branch):
    each spin's active DF tensor, factored jointly so both share the auxiliary index.
    """
    _require_df_x64(mf)
    coeffs = [np.asarray(c) for c in lno_coeff]
    ncore, nocc, ncas, _ = get_las_idx(mf, lno_frozen)
    print(f"[lnotrot] fragment space: nocc={nocc} ncas={ncas} ncore={ncore}")

    t0 = time.time()
    core = [coeffs[s][:, : ncore[s]] for s in range(2)]
    act = [coeffs[s][:, ncore[s] : ncore[s] + ncas[s]] for s in range(2)]
    h0, (h1a, h1b) = lno_effective_core(mf, core, act)
    print(f"[lnotrot] effective core and h1 in {time.time() - t0:.2f}s (E_core={h0:.10f})")

    t0 = time.time()
    df_a, df_b = active_df(mf, act)
    print(f"[lnotrot] DF tensors in the active spaces {df_a.shape}, {df_b.shape} in {time.time() - t0:.2f}s")

    t0 = time.time()
    uc = joint_df2chol(df_a, df_b, chol_cut=chol_cut)
    print(
        f"[lnotrot] joint cholesky: nchol={uc.nchol} residual_max={uc.residual_max:.2e} "
        f"(cut {chol_cut:g}) in {time.time() - t0:.2f}s"
    )

    frozen = tuple(
        np.asarray(f, dtype=np.int64).reshape(-1) if not isinstance(f, int) else np.zeros((0,), dtype=np.int64)
        for f in lno_frozen
    )
    return HamInputU(
        h0=float(h0),
        h1_a=_sym(h1a),
        h1_b=_sym(h1b),
        chol_a=np.asarray(uc.chol_a),
        chol_b=np.asarray(uc.chol_b),
        nelec=(int(nocc[0]), int(nocc[1])),
        norb=(int(ncas[0]), int(ncas[1])),
        chol_cut=float(chol_cut),
        frozen=frozen[0],
        source_kind="mf",
        basis="uchol",
    )
