"""
S^2 and S_z of a saved AFQMC wavefunction (wavefunction_io).

The guided walker population represents

    |Psi> = sum_i c_i |phi_i>,     c_i = w_i / <G|phi_i>

with every |phi_i> a determinant of the same (N_alpha, N_beta). S_z = (N_alpha - N_beta)/2
is therefore m on every determinant and on any combination of them, and with
S^2 = S_- S_+ + S_z^2 + S_z the transition value between two determinants is

    <phi|S^2|phi'> / <phi|phi'> = m (m + 1) + N_beta
                                  - Tr[ M_a^-1 (Phi_a^H S Phi'_b) M_b^-1 (Phi_b^H S Phi'_a) ]

    M_s = Phi_s^H S Phi'_s,     <phi|phi'> = det(M_a) det(M_b)

where Phi_s (nao, N_s) are the occupied orbitals of spin s in the AO basis and S is the
AO overlap matrix. The alpha and beta orbitals are compared with each other, so both have
to be in one common basis: the AOs, with the frozen core put back (occupied_ao_orbitals).

transition_spin       overlap and S^2 of one pair of determinants
pair_spin             the same for one determinant against a stack
occupied_ao_orbitals  every walker's occupied orbitals in the AO basis, core first
population_spin       the mixed estimator <G|S^2|Psi>/<G|Psi>, the expectation value
                      <Psi|S^2|Psi>/<Psi|Psi> over all walker pairs, and <S_z>
guide_from_snapshots  the guide determinant of a run whose walkers start from it
spin_along_tau        population_spin of every snapshot of a run, in order of tau
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np
from numpy.typing import NDArray
from pyscf import lib

from .wavefunction_io import load_wavefunction


def transition_spin(
    phi_a: NDArray, phi_b: NDArray, phip_a: NDArray, phip_b: NDArray, s_ao: NDArray
) -> tuple[Any, Any]:
    """
    (<phi|phi'>, <phi|S^2|phi'>/<phi|phi'>) for two determinants with occupied orbitals
    phi_s (nao, N_s) and phip_s in an AO basis with overlap matrix s_ao. The determinants
    share (N_alpha, N_beta), so S_z = m on both and S_z^2 + S_z = m (m + 1).
    """
    na, nb = phi_a.shape[1], phi_b.shape[1]
    m = 0.5 * (na - nb)
    m_a = phi_a.conj().T @ s_ao @ phip_a
    m_b = phi_b.conj().T @ s_ao @ phip_b
    overlap = np.linalg.det(m_a) * np.linalg.det(m_b)
    x_ab = phi_a.conj().T @ s_ao @ phip_b  # <phi_alpha | phi'_beta>
    x_ba = phi_b.conj().T @ s_ao @ phip_a  # <phi_beta  | phi'_alpha>
    flip = np.trace(np.linalg.solve(m_a, x_ab) @ np.linalg.solve(m_b, x_ba))
    return overlap, m * (m + 1) + nb - flip


def pair_spin(
    phi_a: NDArray, phi_b: NDArray, sphip_a: NDArray, sphip_b: NDArray
) -> tuple[NDArray, NDArray]:
    """
    transition_spin of one determinant (phi_s: (nao, N_s)) against a stack of them, given
    as S @ phi'_w (sphip_s: (n, nao, N_s)): the n overlaps and the n values of
    <phi|S^2|phi'_w>/<phi|phi'_w>.
    """
    na, nb = phi_a.shape[1], phi_b.shape[1]
    m = 0.5 * (na - nb)
    pa, pb = phi_a.conj().T, phi_b.conj().T
    m_a, m_b = pa @ sphip_a, pb @ sphip_b
    x_ab, x_ba = pa @ sphip_b, pb @ sphip_a
    overlap = np.linalg.det(m_a) * np.linalg.det(m_b)
    # the trace of the product, per determinant of the stack
    flip = np.sum(
        np.linalg.solve(m_a, x_ab) * np.linalg.solve(m_b, x_ba).transpose(0, 2, 1), axis=(1, 2)
    )
    return overlap, m * (m + 1) + nb - flip


def occupied_ao_orbitals(wf: dict[str, Any]) -> tuple[NDArray, NDArray]:
    """
    Every walker's occupied orbitals in the AO basis, the frozen core first: the
    (n_walkers, nao, N_alpha) and (n_walkers, nao, N_beta) arrays of a file read by
    load_wavefunction. A restricted hamiltonian carries one basis for both spins;
    restricted walkers are the same determinant for both.

    AFQMC runs in the orbital basis |p> with AO coefficients C^m (basis.coeff), so a
    walker in the AO basis |mu> is, per spin,

        C^ao_{mu,i} = sum_p C^m_{mu,p} C^w_{p,i}

    with C^w the walker in the whole orbital set, the frozen orbitals put back
    (wavefunction_io.walker_full_coefficients): the identity on the frozen occupied
    orbitals, the saved walker W on the active ones, zero on the frozen virtual ones,

        C^w = [[ 1, 0 ],      frozen occupied
               [ 0, W ],      active
               [ 0, 0 ]]      frozen virtual

    Splitting the sum over p into those three ranges, the frozen virtual part drops out
    and the two kinds of columns separate:

        C^ao_{mu,i} = C^m_{mu,i}                           i a frozen occupied orbital
        C^ao_{mu,i} = sum_{p in active} C^m_{mu,p} W_{p,i}  i an occupied orbital of the walker

    that is C^ao = [ C^m_frozen_occ | C^m_active @ W ], which is what is built here.
    """
    if wf["walker_kind"] == "generalized":
        raise ValueError("generalized walkers have no (N_alpha, N_beta); S_z is not fixed.")
    out = []
    for key, bkey in (("walkers_a", "basis_a"), ("walkers_b", "basis_b")):
        basis = wf[bkey] if bkey in wf else wf["basis"]
        walkers = wf[key] if key in wf else wf["walkers"]
        core = basis.frozen_occ_coeff  # (nao, ncore)
        act = lib.einsum("pq,wqi->wpi", basis.active_coeff, walkers)  # (n, nao, nocc_active)
        out.append(
            np.concatenate([np.broadcast_to(core, (act.shape[0],) + core.shape), act], axis=2)
        )
    return out[0], out[1]


def population_spin(
    wf: dict[str, Any], s_ao: NDArray, guide_a: NDArray, guide_b: NDArray
) -> dict[str, Any]:
    """
    The spin of the population |Psi> = sum_i c_i |phi_i>, c_i = w_i / <G|phi_i>, of a
    loaded file, with the guide determinant's occupied AO orbitals guide_s (nao, N_s):

        mixed       <G|S^2|Psi>/<G|Psi> = sum_i w_i s2(G, phi_i) / sum_i w_i
        pure        <Psi|S^2|Psi>/<Psi|Psi> over all walker pairs
        sz          <Psi|S_z|Psi>/<Psi|Psi>, which is m
        s2_walkers  s2(G, phi_i) of every walker
        norm        <Psi|Psi>

    The mixed estimator is that of a single determinant guide. The pair sum costs
    n_walkers^2 / 2 small determinants and solves: each pair is evaluated once, S^2
    being hermitian.
    """
    phi_a, phi_b = occupied_ao_orbitals(wf)
    n = phi_a.shape[0]
    sphi_a = lib.einsum("pq,wqi->wpi", s_ao, phi_a)
    sphi_b = lib.einsum("pq,wqi->wpi", s_ao, phi_b)
    w = np.asarray(wf["weights"])
    c = w / np.asarray(wf["overlaps"])
    m = 0.5 * (phi_a.shape[2] - phi_b.shape[2])
    # mixed: the guide determinant against every walker
    _, s2_g = pair_spin(guide_a, guide_b, sphi_a, sphi_b)
    mixed = np.sum(w * s2_g) / np.sum(w)
    # the population itself: all pairs
    ov = np.empty((n, n), dtype=complex)
    s2 = np.empty((n, n), dtype=complex)
    for i in range(n):
        ov_i, s2_i = pair_spin(phi_a[i], phi_b[i], sphi_a[i:], sphi_b[i:])
        ov[i, i:], s2[i, i:] = ov_i, s2_i
        # <phi_j|S^2|phi_i> = <phi_i|S^2|phi_j>^*, and the overlaps likewise
        ov[i:, i], s2[i:, i] = ov_i.conj(), s2_i.conj()
    weight = np.conj(c)[:, None] * c[None, :] * ov
    norm = weight.sum()
    return dict(
        mixed=mixed,
        pure=(weight * s2).sum() / norm,
        sz=(weight * m).sum() / norm,
        s2_walkers=s2_g,
        norm=norm,
    )


def guide_from_snapshots(snap_dir: Union[str, Path]) -> tuple[NDArray, NDArray]:
    """
    Walker 0 of the tau = 0 snapshot of a run, in the AO basis. The walkers of a run
    start from the guide determinant unless initial walkers were given, so for a single
    determinant guide (RHF, UHF) this is the guide.
    """
    snap_dir = Path(snap_dir)
    manifest = json.loads((snap_dir / "snapshots.json").read_text())
    phi_a, phi_b = occupied_ao_orbitals(load_wavefunction(snap_dir / manifest[0]["file"]))
    return phi_a[0], phi_b[0]


def spin_along_tau(
    snap_dir: Union[str, Path],
    s_ao: NDArray,
    guide: tuple[NDArray, NDArray] | None = None,
) -> list[dict[str, Any]]:
    """
    population_spin of every snapshot AfqmcMixed(save_wavefunction=snap_dir) wrote, in
    order of tau: one dict per snapshot with tau, phase, s2_mixed, s2_pure, sz (real
    parts), pure_imag (the imaginary part of s2_pure, a measure of the noise) and the
    range s2_walker_min, s2_walker_max of s2(G, phi_i). guide is the guide determinant's
    occupied AO orbitals per spin, by default guide_from_snapshots(snap_dir).
    """
    snap_dir = Path(snap_dir)
    guide_a, guide_b = guide_from_snapshots(snap_dir) if guide is None else guide
    rows = []
    for entry in json.loads((snap_dir / "snapshots.json").read_text()):
        r = population_spin(load_wavefunction(snap_dir / entry["file"]), s_ao, guide_a, guide_b)
        rows.append(
            dict(
                tau=entry["tau"],
                phase=entry["phase"],
                s2_mixed=r["mixed"].real,
                s2_pure=r["pure"].real,
                pure_imag=r["pure"].imag,
                sz=r["sz"].real,
                s2_walker_min=r["s2_walkers"].real.min(),
                s2_walker_max=r["s2_walkers"].real.max(),
            )
        )
    return rows
