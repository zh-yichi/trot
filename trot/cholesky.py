from __future__ import annotations

import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Tuple, TypeAlias

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import ArrayLike, NDArray

print = partial(print, flush=True)

# Everything trot does to a cholesky tensor that is not a measurement: building it,
# rotating it into an orbital basis, folding a frozen core into it, and splitting it into
# chunks. Collected here so a module can import the one piece it needs rather than
# reaching into staging.
#
#   building      modified_cholesky   from a packed ERI matrix
#                 chunked_cholesky    from a pyscf mol, shell by shell (AO basis)
#                 df2chol             from a density fitting tensor
#                 df_cderi            the DF tensor of a mean field, if it has one
#                 ao_cholesky         DF when available, else chunked_cholesky
#   rotating      rotate_chol_to_mo, rotate_chol_to_ghf_mo
#   frozen core   freeze_core_from_mo_cholesky      closed shell
#                 freeze_core_from_mo_cholesky_uh   per spin
#   chunking      equal_chunks, max_equal_chunk_pad, chunk_rot_chol
#   joint DF      joint_df_pair_cholesky   factor the alpha and beta pair spaces together
#                 build_joint_df_cholesky  the same, straight from a density fitted mf
#   common space  common_active_space            the union of the two active spaces
#                 build_common_space_df_cholesky decompose there once, project per spin
#
# The builders and the frozen core routines are numpy and run at staging time; the
# chunking helpers are shape arithmetic used by the jax kernels.

Array: TypeAlias = NDArray[Any]


# ======================================================================================
# building
# ======================================================================================


def modified_cholesky(
    mat: Array,
    max_error: float = 1e-6,
) -> Array:
    """Modified cholesky decomposition for a given matrix.

    Args:
        mat (Array): Matrix to decompose.
        max_error (float, optional): Maximum error allowed. Defaults to 1e-6.

    Returns:
        Array: Cholesky vectors.
    """
    diag = mat.diagonal()
    norb = int(((-1 + (1 + 8 * mat.shape[0]) ** 0.5) / 2))
    size = mat.shape[0]
    nchol_max = size
    chol_vecs = np.zeros((nchol_max, nchol_max))
    # ndiag = 0
    nu = np.argmax(diag)
    delta_max = diag[nu]
    Mapprox = np.zeros(size)
    chol_vecs[0] = np.copy(mat[nu]) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error and (nchol + 1) < nchol_max:
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (mat[nu] - R) / (delta_max + 1e-10) ** 0.5
        nchol += 1

    chol0 = chol_vecs[:nchol]
    nchol = chol0.shape[0]
    chol = np.zeros((nchol, norb, norb))
    for i in range(nchol):
        for m in range(norb):
            for n in range(m + 1):
                triind = m * (m + 1) // 2 + n
                chol[i, m, n] = chol0[i, triind]
                chol[i, n, m] = chol0[i, triind]
    return chol


def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10) -> NDArray:
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems. (copied from pauxy)

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`numpy.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao
    chol_vecs = np.zeros((nchol_max, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs.")
        print("# max number of cholesky vectors = %d" % nchol_max)
        print("# iteration %5d: delta_max = %f" % (0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor("int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor(
            "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
        )
        # Select correct ERI chunk from shell.
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (Munu0 - R) / (delta_max) ** 0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e" % info)

    return chol_vecs[:nchol]


def df2chol(dferi: NDArray, max_error: float = 1e-6) -> NDArray:
    """
    Modified cholesky decomposition of a density fitting tensor.

    Args:
        dferi: packed 3-index DF integrals, shape (n_aux, n_pair) with n_pair the lower
            triangle of the AO pair index (pyscf lib.pack_tril ordering).
        max_error: stop when the residual diagonal falls below this.

    Returns:
        (n_chol, norb, norb) cholesky vectors.
    """
    dferi = np.asarray(dferi)
    n_aux, n_pair = dferi.shape
    norb = int(round((-1 + (1 + 8 * n_pair) ** 0.5) / 2))
    if norb * (norb + 1) // 2 != n_pair:
        raise ValueError(f"n_pair={n_pair} is not a valid packed lower triangle size")

    diag = (dferi**2).sum(axis=0)
    chol_vecs = np.zeros((n_aux, n_pair))
    m_approx = np.zeros(n_pair)
    diag_residual = diag.copy()

    nchol = 0
    while nchol < n_aux:
        nu = int(np.argmax(diag_residual))
        delta_max = diag_residual[nu]
        if delta_max < max_error:
            break

        row_nu = dferi.T @ dferi[:, nu]
        if nchol == 0:
            chol_vecs[nchol] = row_nu / delta_max**0.5
        else:
            r = chol_vecs[:nchol, nu] @ chol_vecs[:nchol, :]
            chol_vecs[nchol] = (row_nu - r) / delta_max**0.5

        m_approx += chol_vecs[nchol] ** 2
        diag_residual = np.abs(diag - m_approx)
        nchol += 1

    chol = np.zeros((nchol, norb, norb))
    row_idx, col_idx = np.tril_indices(norb)
    chol[:, row_idx, col_idx] = chol_vecs[:nchol]
    chol[:, col_idx, row_idx] = chol_vecs[:nchol]
    return chol


def df_cderi(mf: Any) -> NDArray | None:
    """Packed (n_aux, n_pair) DF tensor if this mean field is density fitted, else None."""
    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        return None
    blocks = [np.asarray(b) for b in with_df.loop()]
    if not blocks:
        return None
    return np.vstack(blocks)


def ao_cholesky(mf: Any, *, chol_cut: float, verbose: bool = False) -> NDArray:
    """
    AO cholesky vectors of a mean field, flattened to (n_chol, nao*nao).

    Uses the density fitting tensor when the mean field carries one, otherwise falls back
    to the modified cholesky decomposition of the AO ERIs of mf.mol.
    """
    mol = mf.mol
    t0 = time.time()
    cderi = df_cderi(mf)

    if cderi is not None:
        print(
            f"[stage] cholesky from density fitting (n_aux={cderi.shape[0]}), "
            f"max_error={chol_cut:g} ..."
        )
        chol = df2chol(cderi, max_error=chol_cut)
        nao = int(chol.shape[1])
        chol = chol.reshape(chol.shape[0], nao * nao)
        source = "density fitting"
    else:
        print(f"[stage] AO modified cholesky, max_error={chol_cut:g} ...")
        chol = np.asarray(chunked_cholesky(mol, max_error=chol_cut, verbose=verbose))
        source = "AO modified cholesky"

    print(f"[stage] {source}: nchol={chol.shape[0]} in {time.time() - t0:.2f}s")
    return chol


# ======================================================================================
# rotating into an orbital basis
# ======================================================================================


def rotate_chol_to_mo(chol_vec: Array, basis_coeff: Array) -> Array:
    """Rotate AO-space Cholesky into an MO basis."""
    C = np.asarray(basis_coeff)
    nao, norb = C.shape
    nchol = int(chol_vec.shape[0])
    out_dtype = np.result_type(chol_vec.dtype, C.dtype)

    reuse_storage = nao == norb and out_dtype == chol_vec.dtype
    if reuse_storage:
        chol = chol_vec.reshape(nchol, nao, nao)
    else:
        chol = np.empty((nchol, norb, norb), dtype=out_dtype)

    Cdag = np.asarray(C.conj().T)
    tmp = np.empty((nao, norb), dtype=out_dtype)
    for i in range(nchol):
        chol_i_ao = chol_vec[i].reshape(nao, nao)
        np.dot(chol_i_ao, C, out=tmp)
        np.dot(Cdag, tmp, out=chol[i])

    return chol


def rotate_chol_to_ghf_mo(chol_vec: Array, basis_coeff: Array) -> Array:
    """Rotate spatial AO Cholesky factors into a generalized-spin MO basis."""
    C = np.asarray(basis_coeff)
    nao2, nmo = C.shape
    if nao2 % 2 != 0:
        raise ValueError(f"Expected even GHF AO dimension, got {nao2}")

    nao = nao2 // 2
    nchol = int(chol_vec.shape[0])
    out_dtype = np.result_type(chol_vec.dtype, C.dtype)
    chol = np.empty((nchol, nmo, nmo), dtype=out_dtype)

    Cdag = np.asarray(C.conj().T)
    chol_i_full = np.zeros((nao2, nao2), dtype=out_dtype)
    tmp = np.empty((nao2, nmo), dtype=out_dtype)
    for i in range(nchol):
        chol_i = chol_vec[i].reshape(nao, nao)
        chol_i_full.fill(0)
        chol_i_full[:nao, :nao] = chol_i
        chol_i_full[nao:, nao:] = chol_i
        np.dot(chol_i_full, C, out=tmp)
        np.dot(Cdag, tmp, out=chol[i])

    return chol


# ======================================================================================
# frozen core
# ======================================================================================


def freeze_core_from_mo_cholesky(
    *,
    h0: float,
    h1: NDArray,
    chol: NDArray,
    norb_frozen: int,
    nelec: Tuple[int, int],
) -> tuple[float, NDArray, NDArray, Tuple[int, int]]:
    nmo = int(h1.shape[0])
    if h1.shape != (nmo, nmo):
        raise ValueError(f"h1 must be square, got shape {h1.shape}.")
    if chol.ndim != 3 or chol.shape[1:] != (nmo, nmo):
        raise ValueError(f"chol must have shape (nchol, {nmo}, {nmo}), got {chol.shape}.")
    if norb_frozen < 0:
        raise ValueError(f"norb_frozen must be non-negative, got {norb_frozen}.")
    if norb_frozen > min(nelec):
        raise ValueError(f"norb_frozen={norb_frozen} exceeds min(nelec)={min(nelec)}")
    if norb_frozen >= nmo:
        raise ValueError(f"norb_frozen={norb_frozen} leaves no active orbitals (nmo={nmo}).")
    if norb_frozen == 0:
        return float(h0), np.asarray(h1), np.asarray(chol), nelec

    nelec_active = (int(nelec[0] - norb_frozen), int(nelec[1] - norb_frozen))
    if nelec_active[0] < 0 or nelec_active[1] < 0:
        raise ValueError(
            f"norb_frozen={norb_frozen} leaves negative active electron count "
            f"{nelec_active} from nelec={nelec}."
        )
    if sum(nelec_active) <= 0:
        raise ValueError("Frozen core left no active electrons.")

    core = slice(0, norb_frozen)
    act = slice(norb_frozen, nmo)

    chol_core = np.asarray(chol[:, core, core])
    chol_act = np.asarray(chol[:, act, act])
    chol_act_core = np.asarray(chol[:, act, core])
    chol_core_act = np.asarray(chol[:, core, act])

    core_trace = np.trace(chol_core, axis1=1, axis2=2)
    vj = 2.0 * np.einsum("x,xpq->pq", core_trace, chol_act, optimize=True)
    vk = np.einsum("xpi,xiq->pq", chol_act_core, chol_core_act, optimize=True)

    h1_eff = np.asarray(h1[act, act]) + vj - vk

    e1_core = 2.0 * np.trace(np.asarray(h1[core, core]))
    ej_core = 2.0 * np.dot(core_trace, core_trace)
    ek_core = np.einsum("xij,xji->", chol_core, chol_core, optimize=True)
    ecore = float(np.real(h0 + e1_core + ej_core - ek_core))

    return ecore, np.asarray(h1_eff), np.array(chol_act, copy=True), nelec_active


@jax.jit
def _ujk_from_cderi(cderi: jax.Array, dm_a: jax.Array, dm_b: jax.Array):
    """J[D^a + D^b], K[D^a], K[D^b] from one (naux, nao, nao) block of DF vectors."""
    dm_tot = dm_a + dm_b
    vj = jnp.einsum("g,gij->ij", jnp.einsum("gkl,lk->g", cderi, dm_tot), cderi)
    vk_a = jnp.einsum("gik,kl,glj->ij", cderi, dm_a, cderi, optimize=True)
    vk_b = jnp.einsum("gik,kl,glj->ij", cderi, dm_b, cderi, optimize=True)
    return vj, vk_a, vk_b


def _core_uveff(mf: Any, dm_a: NDArray, dm_b: NDArray) -> tuple[NDArray, NDArray]:
    """
    V^s = J[D^a + D^b] - K[D^s] for the core densities, in the AO basis.

    A density fitted mf is contracted block by block on the device, which is much faster
    than pyscf's DF get_jk; otherwise this is mf.get_jk with the exact ERIs. Without
    jax_enable_x64 the device path would be single precision, so it falls back to
    mf.get_jk (still density fitted) there too.
    """
    with_df = getattr(mf, "with_df", None)
    if with_df is None or not jax.config.read("jax_enable_x64"):
        vj, vk = mf.get_jk(mf.mol, np.asarray([dm_a, dm_b]), hermi=1)
        return vj[0] + vj[1] - vk[0], vj[0] + vj[1] - vk[1]

    from pyscf import lib

    dm_a = jnp.asarray(dm_a)
    dm_b = jnp.asarray(dm_b)
    vj = jnp.zeros_like(dm_a)
    vk_a = jnp.zeros_like(dm_a)
    vk_b = jnp.zeros_like(dm_b)
    for cderi in with_df.loop():
        cderi = jnp.asarray(lib.unpack_tril(cderi, axis=-1))
        dvj, dvk_a, dvk_b = _ujk_from_cderi(cderi, dm_a, dm_b)
        vj += dvj
        vk_a += dvk_a
        vk_b += dvk_b
    return np.asarray(vj - vk_a), np.asarray(vj - vk_b)


def freeze_core_from_mo_cholesky_uh(
    *,
    mf: Any,
    basis_a: NDArray,
    basis_b: NDArray,
    chol_a: NDArray,
    chol_b: NDArray,
    norb_frozen: int,
    nelec: Tuple[int, int],
) -> tuple[float, NDArray, NDArray, NDArray, NDArray, Tuple[int, int]]:
    """
    Unrestricted frozen core.

    The core energy and the core potential are built from the AO integrals of mf
    (the DF tensor on the GPU if mf is density fitted, else mf.get_jk), not from the
    cholesky vectors, so h0 and h1 do not depend on how, or how tightly, the ERIs were
    decomposed. With the core
    densities D^s = C^s_c C^s_c^dag and V^s = J[D^a + D^b] - K[D^s],

        E_core   = E_nuc + sum_s tr[D^s (h + V^s / 2)]
        h1_eff^s = C^s_a^dag (h + V^s) C^s_a

    basis_a / basis_b are the full (nao, nmo_s) bases chol_a / chol_b were rotated into;
    their first norb_frozen columns are the core. The cholesky vectors are only cut down
    to the active block. Setting alpha == beta reduces to the restricted expressions.
    """
    basis_a = np.asarray(basis_a)
    basis_b = np.asarray(basis_b)
    nmo_a, nmo_b = int(basis_a.shape[1]), int(basis_b.shape[1])
    for spin, chol, nmo in (("a", chol_a, nmo_a), ("b", chol_b, nmo_b)):
        if chol.ndim != 3 or chol.shape[1:] != (nmo, nmo):
            raise ValueError(
                f"chol_{spin} must have shape (nchol, {nmo}, {nmo}), got {chol.shape}."
            )
    if norb_frozen < 0:
        raise ValueError(f"norb_frozen must be non-negative, got {norb_frozen}.")
    if norb_frozen > min(nelec):
        raise ValueError(f"norb_frozen={norb_frozen} exceeds min(nelec)={min(nelec)}")
    if norb_frozen >= min(nmo_a, nmo_b):
        raise ValueError(
            f"norb_frozen={norb_frozen} leaves no active orbitals "
            f"(norb_a={nmo_a}, norb_b={nmo_b})."
        )

    nelec_active = (int(nelec[0] - norb_frozen), int(nelec[1] - norb_frozen))
    if min(nelec_active) < 0 or sum(nelec_active) <= 0:
        raise ValueError("Frozen core left no active electrons.")

    hcore = np.asarray(mf.get_hcore())
    ecore = float(mf.energy_nuc())

    core_a, act_a = basis_a[:, :norb_frozen], basis_a[:, norb_frozen:]
    core_b, act_b = basis_b[:, :norb_frozen], basis_b[:, norb_frozen:]

    if norb_frozen == 0:
        veff_a = veff_b = 0.0
    else:
        dm_a = core_a @ core_a.conj().T
        dm_b = core_b @ core_b.conj().T
        veff_a, veff_b = _core_uveff(mf, dm_a, dm_b)

        e_core = (
            np.einsum("ij,ji->", dm_a, hcore + 0.5 * veff_a)
            + np.einsum("ij,ji->", dm_b, hcore + 0.5 * veff_b)
        ) 
        ecore += float(np.real(e_core))

    h1_eff_a = act_a.conj().T @ (hcore + veff_a) @ act_a
    h1_eff_b = act_b.conj().T @ (hcore + veff_b) @ act_b

    act = slice(norb_frozen, None)
    return (
        ecore,
        np.asarray(h1_eff_a),
        np.asarray(h1_eff_b),
        np.array(chol_a[:, act, act], copy=True),
        np.array(chol_b[:, act, act], copy=True),
        nelec_active,
    )


# ======================================================================================
# chunking
# ======================================================================================


def equal_chunks(n: int, max_chunk: int) -> tuple[int, int, int]:
    """
    Split n cholesky vectors into equal chunks of at most max_chunk, returning
    (n_chunks, chunk, n_pad).

    Take the fewest chunks the cap allows, then divide evenly. That gives the same number
    of scan steps as slicing at exactly max_chunk and padding the remainder, but spreads
    the vectors out, so the zero padding is the minimum a fixed scan shape admits: with
    nchol=1600 and a cap of 300 it pads 2 vectors rather than 200.

    Padding is always < n_chunks, and is bounded over all caps by about sqrt(n) -- worst
    when the chunk count and the chunk size meet, e.g. 19 vectors at n=381, cap=20.

    Every chunking kernel uses this, on the whole cholesky set and on the subsets the
    semistochastic kernels form (their head and their sampled tail). Subsets are why an
    exactly-max_chunk rule will not do: with a cap sized for the full set, a short head
    would pad out to one whole chunk and run the T2 contractions on mostly zeros.
    """
    if max_chunk < 1:
        raise ValueError(f"max_chunk must be >= 1, got {max_chunk}")
    n_chunks = max(1, -(-n // max_chunk))
    chunk = max(1, -(-n // n_chunks))  # max(1, ...) keeps n == 0 from giving a zero axis
    return n_chunks, chunk, n_chunks * chunk - n


def max_equal_chunk_pad(n: int) -> int:
    """
    Worst-case padding from equal_chunks over every cap, i.e. how many zero cholesky
    vectors the padded copy can carry whatever chunk size is chosen. Small (about sqrt(n)),
    and k independent, which is what lets the memory model charge it once up front instead
    of scaling it with the chunk.
    """
    if n <= 0:
        return 1
    return max(equal_chunks(n, cap)[2] for cap in range(1, n + 1))


def chunk_rot_chol(rot_chol: jax.Array, nchol_chunk: int | None) -> jax.Array:
    """(n_chol, nocc, norb) -> (n_chunks, nchol_chunk, nocc, norb), zero padded."""
    n_chol = int(rot_chol.shape[0])
    chunk = n_chol if nchol_chunk is None else int(nchol_chunk)
    chunk = max(1, min(chunk, n_chol))
    n_chunks = -(-n_chol // chunk)
    pad = n_chunks * chunk - n_chol
    if pad:
        rot_chol = jnp.pad(rot_chol, ((0, pad), (0, 0), (0, 0)))
    return rot_chol.reshape(n_chunks, chunk, *rot_chol.shape[1:])


# ======================================================================================
# joint alpha/beta density fitting
#
# For an unrestricted hamiltonian the two spins need cholesky vectors over one shared
# auxiliary index while living in different orbital spaces. build_ham_uchol gets that by
# projecting one set of AO vectors into each basis; these factor the two pair spaces
# together instead, pivoting on whichever spin carries the largest residual, which is
# what an unrestricted LNO fragment needs when the alpha and beta active spaces are
# chosen independently.
#
# Pulled from main_trot/trot/lno_uhf.py.
# ======================================================================================


@dataclass(frozen=True, slots=True)
class UCholesky:
    """
    Cholesky vectors for an unrestricted hamiltonian: one set per spin over a shared
    auxiliary index, which is how HamCholU carries them. Every unrestricted construction
    in this module returns this.

      chol_a, chol_b  (nchol, norb_s, norb_s), sharing the auxiliary index
      chol_cut        the cutoff they were built to
      residual_max    the largest residual the decomposition left behind, i.e. how far
                      the vectors still are from the tensor they factor. It is below
                      chol_cut when the decomposition ran to convergence rather than
                      stopping at a rank limit
      common          the common active space, when the construction went through one
    """

    chol_a: NDArray
    chol_b: NDArray
    chol_cut: float
    residual_max: float
    common: "CommonActiveSpace | None" = None

    @property
    def nchol(self) -> int:
        return int(self.chol_a.shape[0])


def _as_packed_df(df: ArrayLike) -> NDArray:
    array = np.asarray(df)
    if array.ndim == 2:
        return array
    if array.ndim != 3 or array.shape[1] != array.shape[2]:
        raise ValueError("DF tensors must have shape (naux,npair) or (naux,norb,norb).")
    rows, cols = np.tril_indices(array.shape[1])
    return array[:, rows, cols]


def _unpack_symmetric(packed: NDArray, norb: int) -> NDArray:
    expected = norb * (norb + 1) // 2
    if packed.ndim != 2 or packed.shape[1] != expected:
        raise ValueError(f"packed tensor must have {expected} columns, got {packed.shape}.")
    output = np.zeros((packed.shape[0], norb, norb), dtype=packed.dtype)
    rows, cols = np.tril_indices(norb)
    output[:, rows, cols] = packed
    output[:, cols, rows] = packed
    return output


def joint_df2chol(
    df_a: ArrayLike,
    df_b: ArrayLike,
    *,
    chol_cut: float = 1e-5,
    max_chol: int | None = None,
) -> UCholesky:
    """Factor the joint A/B pair Gram matrix without forming that matrix."""
    a = _as_packed_df(df_a)
    b = _as_packed_df(df_b)
    if a.shape[0] != b.shape[0]:
        raise ValueError("alpha and beta DF tensors must use the same auxiliary basis.")
    if chol_cut <= 0:
        raise ValueError("chol_cut must be positive.")

    naux = int(a.shape[0])
    npa, npb = int(a.shape[1]), int(b.shape[1])
    npair = npa + npb
    rank_limit = min(naux, npair) if max_chol is None else min(int(max_chol), npair)
    if rank_limit <= 0:
        raise ValueError("max_chol must be positive.")

    diag = np.concatenate(
        (
            np.einsum("pi,pi->i", a.conj(), a, optimize=True).real,
            np.einsum("pi,pi->i", b.conj(), b, optimize=True).real,
        )
    )
    capacity = min(rank_limit, 64)
    factors = np.empty((npair, capacity), dtype=np.result_type(a, b))
    rank = 0

    while rank < rank_limit:
        pivot = int(np.argmax(diag))
        if diag[pivot] <= chol_cut:
            break
        pivot_df = a[:, pivot] if pivot < npa else b[:, pivot - npa]
        column = np.concatenate((a.conj().T @ pivot_df, b.conj().T @ pivot_df))
        if rank:
            column -= factors[:, :rank] @ factors[pivot, :rank].conj()
        pivot_residual = float(column[pivot].real)
        if pivot_residual <= chol_cut:
            diag[pivot] = max(0.0, pivot_residual)
            continue

        if rank == factors.shape[1]:
            new_capacity = min(rank_limit, max(rank + 1, 2 * rank))
            grown = np.empty((npair, new_capacity), dtype=factors.dtype)
            grown[:, :rank] = factors
            factors = grown
        factors[:, rank] = column / np.sqrt(pivot_residual)
        diag -= np.abs(factors[:, rank]) ** 2
        np.maximum(diag, 0.0, out=diag)
        rank += 1

    residual_max = float(np.max(diag, initial=0.0))
    if residual_max > chol_cut and rank == rank_limit:
        raise RuntimeError(
            f"joint Cholesky reached rank {rank_limit} with residual {residual_max:.3e}."
        )

    rows = factors[:, :rank].T
    na = _pair_dimension(npa)
    nb = _pair_dimension(npb)
    return UCholesky(
        chol_a=_unpack_symmetric(rows[:, :npa], na),
        chol_b=_unpack_symmetric(rows[:, npa:], nb),
        chol_cut=float(chol_cut),
        residual_max=residual_max,
    )


def _pair_dimension(npair: int) -> int:
    norb = int((np.sqrt(8 * npair + 1) - 1) // 2)
    if norb * (norb + 1) // 2 != npair:
        raise ValueError(f"{npair} is not a triangular pair dimension.")
    return norb


def _df_block_to_pairs(block: NDArray, coeff: NDArray) -> NDArray:
    nao, norb = coeff.shape
    if np.iscomplexobj(coeff):
        from pyscf import lib

        ao = lib.unpack_tril(block) if block.shape[1] != nao * nao else block.reshape(-1, nao, nao)
        transformed = np.einsum("Pmn,mp,nq->Ppq", ao, coeff.conj(), coeff, optimize=True)
    else:
        from pyscf.ao2mo import _ao2mo

        if block.shape[1] == nao * (nao + 1) // 2:
            mo = np.asarray(coeff, order="F")
            transformed = _ao2mo.nr_e2(
                block,
                mo,
                (0, norb, 0, norb),
                aosym="s2",
                mosym="s1",
            ).reshape(-1, norb, norb)
        else:
            ao = block.reshape(-1, nao, nao)
            transformed = np.einsum("Pmn,mp,nq->Ppq", ao, coeff, coeff, optimize=True)
    rows, cols = np.tril_indices(norb)
    return np.asarray(transformed[:, rows, cols])


def build_joint_df2chol(
    mf: Any,
    coeff: tuple[NDArray, NDArray],
    *,
    chol_cut: float = 1e-5,
    max_chol: int | None = None,
) -> UCholesky:
    """Transform molecular DF tensors and jointly factor the two pair spaces."""
    if any(np.iscomplexobj(block) for block in coeff):
        raise NotImplementedError(
            "split-LIS AFQMC currently requires real alpha and beta orbital coefficients."
        )
    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        raise TypeError("UHF LNO preparation requires a density-fitted mean field.")
    naux = int(with_df.get_naoaux())
    npa = coeff[0].shape[1] * (coeff[0].shape[1] + 1) // 2
    npb = coeff[1].shape[1] * (coeff[1].shape[1] + 1) // 2
    dtype = np.result_type(coeff[0], coeff[1], np.float64)
    df_a = np.empty((naux, npa), dtype=dtype)
    df_b = np.empty((naux, npb), dtype=dtype)
    offset = 0
    for raw_block in with_df.loop():
        block = np.asarray(raw_block)
        stop = offset + block.shape[0]
        df_a[offset:stop] = _df_block_to_pairs(block, coeff[0])
        df_b[offset:stop] = _df_block_to_pairs(block, coeff[1])
        offset = stop
    if offset != naux:
        raise RuntimeError(f"DF iterator yielded {offset} auxiliaries; expected {naux}.")
    return joint_df2chol(df_a, df_b, chol_cut=chol_cut, max_chol=max_chol)


# ======================================================================================
# common active space density fitting
#
# The other route to one shared auxiliary index, ported from afqmc's
# lno_afqmc/integral.py (common_las and the UHF branch of get_lno_integral). Instead of
# factoring the two pair spaces together, it factors once in the space they span jointly:
#
#   1. cLAS: hstack the alpha and beta active orbitals, drop the linear dependence
#      between them, and orthonormalize what is left -- the union of the two spaces
#   2. transform the DF tensor into that one space and decompose it there, so a single
#      set of cholesky vectors covers both spins
#   3. project those vectors back into each spin's active space with <C|A> and <C|B>
#
# This is what an unrestricted LNO fragment wants: alpha and beta freeze different
# orbitals, so their active spaces overlap heavily but are not the same, and the union is
# barely bigger than either one. Compare build_joint_df_cholesky, which pivots across the
# two pair spaces instead, and build_ham_uchol, which projects one AO factorization into
# each spin separately.
# ======================================================================================


@dataclass(frozen=True, slots=True)
class CommonActiveSpace:
    """The union of the alpha and beta active spaces, and the maps into each of them."""

    coeff: NDArray  # (nao, nclas), orthonormal in the AO metric
    a2c: NDArray  # (nclas, ncas_a), <C|A>
    b2c: NDArray  # (nclas, ncas_b), <C|B>
    rank: int  # nclas
    smallest_kept: float  # smallest retained Gram eigenvalue
    largest_dropped: float  # largest discarded one, 0.0 if the spaces were independent


def _mo_span_loss(mo1: NDArray, s1e: NDArray, mo2: NDArray) -> float:
    """
    How much of span(mo2) escapes span(mo1), as the largest element of

        olp12^H olp12 - olp22,   olp_ij = mo_i^H S mo_j

    which vanishes exactly when mo1 contains mo2. afqmc's tools.mo_span.
    """
    olp12 = mo1.conj().T @ s1e @ mo2
    olp22 = mo2.conj().T @ s1e @ mo2
    return float(np.abs(olp12.conj().T @ olp12 - olp22).max())


def common_active_space(
    s1e: NDArray,
    act_a: NDArray,
    act_b: NDArray,
    *,
    thresh: float = 1e-5,
) -> CommonActiveSpace:
    """
    Orthonormal basis for span(alpha active) U span(beta active).

    act_a and act_b are that spin's active AO coefficients, i.e.
    coeff[:, ncore : ncore + ncas] for each spin, and s1e is the AO overlap.

    Stacking the two blocks gives a spanning set that is linearly dependent wherever the
    two spaces agree -- which is nearly everywhere, since the spins usually differ only in
    which orbitals they freeze. Diagonalizing its overlap (Gram) matrix and keeping the
    eigenvectors above thresh removes exactly that dependence: eigh on a symmetric
    positive semidefinite matrix is the SVD, and the discarded directions are the ones the
    stacked set covers twice.

    thresh is meant to match the cholesky cutoff the vectors are later built with: a
    direction too weak to matter for the decomposition is also too weak to keep here.
    That reasoning holds when the two spaces are nearly the same, so the discarded
    eigenvalues are the redundancy between them and sit far below the kept ones. It fails
    when the spectrum runs continuously through the cutoff, which is what a genuinely
    spin polarized reference gives: measured on high spin Fe(H2O)6(2+), thresh = chol_cut
    = 1e-5 kept 63 of 90 directions and left a 4e-4 error in the reconstructed ERIs,
    while thresh = 1e-7 kept 86 and brought that to 2e-5. The diagnostics on the returned
    CommonActiveSpace (smallest_kept against largest_dropped) are what to check: no gap
    between them means the truncation is cutting into the hamiltonian, and note that the
    resulting loss never shows up in UCholesky.residual_max, which only measures the
    decomposition that follows.
    """
    act = np.hstack([np.asarray(act_a), np.asarray(act_b)])
    s1e = np.asarray(s1e)

    gram = act.conj().T @ s1e @ act
    w, v = np.linalg.eigh(gram)
    w, v = w[::-1], v[:, ::-1]  # descending
    keep = w > thresh
    rank = int(keep.sum())
    if rank == 0:
        raise ValueError(f"no active directions survive thresh={thresh:g}.")

    coeff = act @ (v[:, keep] / np.sqrt(w[keep]))

    # orthonormal by construction; a failure here means thresh kept a direction the
    # metric cannot normalize
    tol = max(float(thresh), 1e-8)
    if not np.allclose(coeff.conj().T @ s1e @ coeff, np.eye(rank), atol=tol):
        raise RuntimeError(f"the common active space is not orthonormal to {tol:g}; raise thresh.")

    # and it must contain each spin's active space, or the projection below loses part of
    # the hamiltonian rather than just its redundancy
    for name, block in (("alpha", act_a), ("beta", act_b)):
        loss = _mo_span_loss(coeff, s1e, np.asarray(block))
        if loss > tol:
            raise RuntimeError(
                f"the common active space misses {loss:.2e} of the {name} active space; "
                f"lower thresh (now {thresh:g})."
            )

    return CommonActiveSpace(
        coeff=coeff,
        a2c=coeff.conj().T @ s1e @ np.asarray(act_a),  # <C|A>
        b2c=coeff.conj().T @ s1e @ np.asarray(act_b),  # <C|B>
        rank=rank,
        smallest_kept=float(w[keep][-1]),
        largest_dropped=float(w[rank]) if rank < w.size else 0.0,
    )


def build_union_df2chol(
    mf: Any,
    act_a: NDArray,
    act_b: NDArray,
    *,
    chol_cut: float = 1e-5,
    thresh: float | None = None,
) -> UCholesky:
    """
    Cholesky vectors for both spins, decomposed once in the common active space.

    act_a and act_b are the two spins' active AO coefficients; mf must be density fitted.
    thresh is the cutoff for the common space itself and defaults to chol_cut.

    The DF tensor is transformed into the common space, decomposed there once, and the
    resulting vectors are projected into each spin's active space,

        L^a_g = <A|C> L^c_g <C|A>,      L^b_g = <B|C> L^c_g <C|B>,

    so both spins come out over one shared auxiliary index g, which is what HamCholU
    needs. Only one decomposition is ever done, in a space barely larger than either
    spin's own.
    """
    thresh = chol_cut if thresh is None else thresh
    common = common_active_space(np.asarray(mf.get_ovlp()), act_a, act_b, thresh=thresh)

    with_df = getattr(mf, "with_df", None)
    if with_df is None:
        raise TypeError(
            "the common active space cholesky reads the DF tensor; use a density fitted "
            "mean field (mf.density_fit())."
        )

    nclas = common.rank
    naux = int(with_df.get_naoaux())
    cderi = np.empty(
        (naux, nclas * (nclas + 1) // 2), dtype=np.result_type(common.coeff, np.float64)
    )
    offset = 0
    for raw_block in with_df.loop():
        block = np.asarray(raw_block)
        stop = offset + block.shape[0]
        cderi[offset:stop] = _df_block_to_pairs(block, common.coeff)
        offset = stop
    if offset != naux:
        raise RuntimeError(f"DF iterator yielded {offset} auxiliaries; expected {naux}.")

    chol_c = df2chol(cderi, max_error=chol_cut)  # (nchol, nclas, nclas)

    # what the decomposition left behind, on the same packed pair diagonal df2chol
    # thresholds: sum_P cderi[P,i]^2 - sum_g L_g[i]^2
    packed = _as_packed_df(chol_c)
    residual_max = float(np.max(np.abs((cderi**2).sum(0) - (packed**2).sum(0))))

    flat = chol_c.reshape(chol_c.shape[0], nclas * nclas)

    # rotate_chol_to_mo rotates in place when the two dimensions agree, which they do
    # whenever one spin already spans the union, so each spin gets its own copy
    chol_a = rotate_chol_to_mo(np.array(flat, copy=True), common.a2c)
    chol_b = rotate_chol_to_mo(np.array(flat, copy=True), common.b2c)

    return UCholesky(
        chol_a=chol_a,
        chol_b=chol_b,
        chol_cut=float(chol_cut),
        residual_max=residual_max,
        common=common,
    )
