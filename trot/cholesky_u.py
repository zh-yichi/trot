"""
Cholesky helpers for the unrestricted (uchol) hamiltonian.

The companion of the cholesky code in staging.py for the case where alpha and beta keep
their own orbital basis. One common set of AO cholesky vectors is decomposed once and
projected into each spin's basis, so the auxiliary field index stays shared while the
orbital dimensions are per spin. See trot/ham/chol_u.py.

    ao_cholesky                    AO cholesky vectors, flattened to (n_chol, nao*nao)
    rotate_chol_to_mo              project them into an orbital basis (staging's rotation)
    normalize_frozen_core_uh       (n_core_a, n_core_b) from an int or a pair
    freeze_core_from_mo_cholesky_uh
                                   unrestricted frozen core from the MO cholesky vectors
    equal_chunks, max_equal_chunk_pad, chunk_rot_chol
                                   fixed shape chunking of a cholesky axis for lax.scan
"""

from __future__ import annotations

import time
from typing import Any, Tuple

import jax
import numpy as np
from numpy.typing import NDArray

from .staging import _rotate_chol_to_mo, chunked_cholesky

Array = Any


# ======================================================================================
# building and rotating
# ======================================================================================


def ao_cholesky(mol: Any, *, chol_cut: float, verbose: bool = False) -> NDArray:
    """
    AO cholesky vectors of a molecule, flattened to (n_chol, nao*nao).

    The same modified cholesky decomposition of the exact AO ERIs that staging uses for
    the restricted hamiltonian, so a uchol hamiltonian built from the same molecule and
    cutoff carries the same vectors.
    """
    t0 = time.time()
    print(f"[stage] AO modified cholesky, max_error={chol_cut:g} ...")
    chol = np.asarray(chunked_cholesky(mol, max_error=chol_cut, verbose=verbose))
    print(f"[stage] AO modified cholesky: nchol={chol.shape[0]} in {time.time() - t0:.2f}s")
    return chol


def rotate_chol_to_mo(chol_vec: Array, basis_coeff: Array) -> Array:
    """
    Project AO cholesky vectors (n_chol, nao*nao) into an orbital basis (nao, norb):
    L_g -> C^dag L_g C, returning (n_chol, norb, norb).

    Note: staging's rotation reuses the input storage when nao == norb, so a caller that
    projects the same vectors into two bases must hand the first call a copy.
    """
    return _rotate_chol_to_mo(chol_vec, basis_coeff)


# ======================================================================================
# frozen core
# ======================================================================================


def normalize_frozen_core_uh(frozen: Any) -> Tuple[int, int]:
    """
    The number of frozen core orbitals per spin.

    An int freezes the same number of leading orbitals in both bases, a pair (n_a, n_b)
    freezes them independently. None is no frozen core. Orbital lists (LNO staging) are
    not supported by the canonical uchol path.
    """
    if frozen is None:
        return (0, 0)
    if isinstance(frozen, (int, np.integer)):
        n = int(frozen)
        return (n, n)
    if isinstance(frozen, (tuple, list, np.ndarray)):
        arr = np.asarray(frozen)
        if arr.ndim == 1 and arr.size == 2 and np.issubdtype(arr.dtype, np.integer):
            return (int(arr[0]), int(arr[1]))
        raise ValueError(
            "the unrestricted hamiltonian takes an integer frozen core, or a pair "
            f"(n_core_a, n_core_b); orbital lists are not supported, got {frozen!r}."
        )
    raise TypeError(f"Unsupported frozen core type: {type(frozen)}")


def freeze_core_from_mo_cholesky_uh(
    *,
    h0: float,
    h1_a: NDArray,
    h1_b: NDArray,
    chol_a: NDArray,
    chol_b: NDArray,
    norb_frozen: Tuple[int, int],
    nelec: Tuple[int, int],
) -> tuple[float, NDArray, NDArray, NDArray, NDArray, Tuple[int, int]]:
    """
    Unrestricted frozen core, built from the MO cholesky vectors as staging's
    _freeze_core_from_mo_cholesky does for the restricted hamiltonian.

    The core of spin s is the leading norb_frozen[s] orbitals of that spin's basis, so
    its density is the identity on that block and the core potential seen by the active
    orbitals of spin s is

        V^s_pq = sum_g L^s_g,pq (sum_t tr_core L^t_g) - sum_g sum_{i in core_s} L^s_g,pi L^s_g,iq

    the coulomb term summed over both spins through the shared field index and the
    exchange term within the spin. With c_g = sum_t tr_core L^t_g,

        E_core = h0 + sum_s tr_core h1^s + 1/2 sum_g c_g^2 - 1/2 sum_s sum_g tr(L^s_g,cc L^s_g,cc)

    Because the potential and the energy come from the same truncated cholesky vectors
    as the two body term, the energy of the reference determinant in the frozen core
    hamiltonian equals its energy in the full one to roundoff. Setting alpha == beta
    reduces to the restricted expressions exactly (the factors of two become sums of two
    equal terms).
    """
    n_core_a, n_core_b = (int(n) for n in norb_frozen)
    nmo_a, nmo_b = int(h1_a.shape[0]), int(h1_b.shape[0])
    for spin, h1, chol, nmo in (("a", h1_a, chol_a, nmo_a), ("b", h1_b, chol_b, nmo_b)):
        if h1.shape != (nmo, nmo):
            raise ValueError(f"h1_{spin} must be square, got shape {h1.shape}.")
        if chol.ndim != 3 or chol.shape[1:] != (nmo, nmo):
            raise ValueError(
                f"chol_{spin} must have shape (nchol, {nmo}, {nmo}), got {chol.shape}."
            )
    if chol_a.shape[0] != chol_b.shape[0]:
        raise ValueError("chol_a and chol_b must share the auxiliary field index.")
    if n_core_a < 0 or n_core_b < 0:
        raise ValueError(f"norb_frozen must be non-negative, got {norb_frozen}.")
    if n_core_a > nelec[0] or n_core_b > nelec[1]:
        raise ValueError(f"norb_frozen={norb_frozen} exceeds nelec={nelec}")
    if n_core_a >= nmo_a or n_core_b >= nmo_b:
        raise ValueError(
            f"norb_frozen={norb_frozen} leaves no active orbitals (norb={(nmo_a, nmo_b)})."
        )
    if n_core_a == 0 and n_core_b == 0:
        return (
            float(h0),
            np.asarray(h1_a),
            np.asarray(h1_b),
            np.asarray(chol_a),
            np.asarray(chol_b),
            nelec,
        )

    nelec_active = (int(nelec[0] - n_core_a), int(nelec[1] - n_core_b))
    if sum(nelec_active) <= 0:
        raise ValueError("Frozen core left no active electrons.")

    core_a, act_a = slice(0, n_core_a), slice(n_core_a, nmo_a)
    core_b, act_b = slice(0, n_core_b), slice(n_core_b, nmo_b)

    chol_core_a = np.asarray(chol_a[:, core_a, core_a])
    chol_core_b = np.asarray(chol_b[:, core_b, core_b])
    # c_g = sum over both spins of the core trace of L_g: one vector on the shared index
    core_trace = np.trace(chol_core_a, axis1=1, axis2=2) + np.trace(chol_core_b, axis1=1, axis2=2)

    def _one_spin(h1, chol, core, act, chol_core):
        chol_act = np.asarray(chol[:, act, act])
        vj = np.einsum("x,xpq->pq", core_trace, chol_act, optimize=True)
        vk = np.einsum(
            "xpi,xiq->pq",
            np.asarray(chol[:, act, core]),
            np.asarray(chol[:, core, act]),
            optimize=True,
        )
        h1_eff = np.asarray(h1[act, act]) + vj - vk
        e1_core = np.trace(np.asarray(h1[core, core]))
        ek_core = np.einsum("xij,xji->", chol_core, chol_core, optimize=True)
        return h1_eff, chol_act, e1_core, ek_core

    h1_eff_a, chol_act_a, e1_a, ek_a = _one_spin(h1_a, chol_a, core_a, act_a, chol_core_a)
    h1_eff_b, chol_act_b, e1_b, ek_b = _one_spin(h1_b, chol_b, core_b, act_b, chol_core_b)

    ej_core = 0.5 * np.dot(core_trace, core_trace)
    ecore = float(np.real(h0 + e1_a + e1_b + ej_core - 0.5 * (ek_a + ek_b)))

    return (
        ecore,
        np.asarray(h1_eff_a),
        np.asarray(h1_eff_b),
        np.array(chol_act_a, copy=True),
        np.array(chol_act_b, copy=True),
        nelec_active,
    )


# ======================================================================================
# chunking
# ======================================================================================


def equal_chunks(n: int, max_chunk: int) -> tuple[int, int, int]:
    """
    Split n items into equal chunks of at most max_chunk, with minimal zero padding.
    Returns (n_chunks, chunk, n_pad) with n_chunks * chunk == n + n_pad.
    """
    n = int(n)
    max_chunk = int(max_chunk)
    if n <= 0:
        return (1, max(max_chunk, 1), max(max_chunk, 1) - n)
    if max_chunk <= 0 or max_chunk >= n:
        return (1, n, 0)
    n_chunks = -(-n // max_chunk)
    chunk = -(-n // n_chunks)
    return (n_chunks, chunk, n_chunks * chunk - n)


def max_equal_chunk_pad(n: int) -> int:
    """
    The largest zero padding equal_chunks can leave for n items over every possible
    chunk cap, about sqrt(n). Memory models use it to bound the padded copy a kernel
    builds whatever chunk size is later chosen.
    """
    n = int(n)
    if n <= 0:
        return 0
    return max(equal_chunks(n, cap)[2] for cap in range(1, n + 1))


def chunk_rot_chol(rot_chol: jax.Array, nchol_chunk: int | None) -> jax.Array:
    """
    (n_chol, nocc, norb) -> (n_chunks, chunk, nocc, norb), zero padded, so that a
    measurement kernel can lax.scan over the chunks with a fixed shape. None keeps the
    whole axis as one chunk.
    """
    import jax.numpy as jnp

    n_chol = int(rot_chol.shape[0])
    if nchol_chunk is None or nchol_chunk >= n_chol:
        return rot_chol.reshape(1, *rot_chol.shape)
    n_chunks, chunk, n_pad = equal_chunks(n_chol, nchol_chunk)
    if n_pad:
        pad = jnp.zeros((n_pad, *rot_chol.shape[1:]), dtype=rot_chol.dtype)
        rot_chol = jnp.concatenate([rot_chol, pad], axis=0)
    return rot_chol.reshape(n_chunks, chunk, *rot_chol.shape[1:])
