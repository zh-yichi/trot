"""
S^2 and S_z of a saved AFQMC wavefunction: the transition formula against pyscf and
against spin-pure determinants, the stacked and hermitian-half evaluation against the
plain pair loop, and the snapshots of an unrestricted run.
"""

from trot import config

config.configure_once()

import contextlib
import io
from typing import Any

import numpy as np
import pytest
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed
from trot.wavefunction_io import (
    WavefunctionBasis,
    dump_wavefunction,
    load_wavefunction,
    walker_full_coefficients,
)
from trot.wavefunction_spin import (
    guide_from_snapshots,
    occupied_ao_orbitals,
    pair_spin,
    population_spin,
    spin_along_tau,
    transition_spin,
)

_NAO, _NA, _NB, _NCORE, _NW = 9, 4, 2, 1, 5


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _random_population(seed: int = 0) -> dict[str, Any]:
    """A random unrestricted population in a non-orthogonal AO basis, with a frozen core."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(_NAO, _NAO))
    s_ao = x @ x.T + _NAO * np.eye(_NAO)
    # two S-orthonormal orbital sets, alpha and beta, sharing the core column
    evals, evecs = np.linalg.eigh(s_ao)
    lowdin = evecs @ np.diag(evals**-0.5) @ evecs.T
    ca = lowdin @ np.linalg.qr(rng.normal(size=(_NAO, _NAO)))[0]
    cb = ca.copy()
    cb[:, _NCORE:] = ca[:, _NCORE:] @ np.linalg.qr(rng.normal(size=(_NAO - 1, _NAO - 1)))[0]
    norb = _NAO - _NCORE

    def walkers(nocc):
        w = rng.normal(size=(_NW, norb, nocc)) + 1j * rng.normal(size=(_NW, norb, nocc))
        return np.eye(norb)[:, :nocc] + 0.3 * w

    return dict(
        s_ao=s_ao,
        basis=(
            WavefunctionBasis.leading_core(ca, _NCORE),
            WavefunctionBasis.leading_core(cb, _NCORE),
        ),
        walkers=(walkers(_NA - _NCORE), walkers(_NB - _NCORE)),
        weights=rng.uniform(0.5, 1.5, size=_NW),
        overlaps=rng.normal(size=_NW) + 1j * rng.normal(size=_NW),
    )


@pytest.fixture(scope="module")
def population(tmp_path_factory):
    pop = _random_population()
    path = tmp_path_factory.mktemp("spin") / "wf.h5"
    dump_wavefunction(
        path,
        walkers=pop["walkers"],
        weights=pop["weights"],
        overlaps=pop["overlaps"],
        walker_kind="unrestricted",
        basis=pop["basis"],
        nelec=(_NA - _NCORE, _NB - _NCORE),
    )
    pop["wf"] = load_wavefunction(path)
    return pop


def test_occupied_ao_orbitals_put_the_core_back(population):
    wf, s_ao = population["wf"], population["s_ao"]
    phi_a, phi_b = occupied_ao_orbitals(wf)
    assert phi_a.shape == (_NW, _NAO, _NA) and phi_b.shape == (_NW, _NAO, _NB)
    for phi, basis in ((phi_a, wf["basis_a"]), (phi_b, wf["basis_b"])):
        np.testing.assert_allclose(phi[0][:, :_NCORE], basis.coeff[:, :_NCORE])
        # the active part is orthogonal to the core
        assert (
            np.abs(phi[:, :, :_NCORE].transpose(0, 2, 1) @ s_ao @ phi[:, :, _NCORE:]).max() < 1e-12
        )


def test_full_coefficients_are_the_occupied_orbitals(population):
    """coeff @ [[1, 0], [0, W]] is what the spin analysis uses, per spin."""
    wf = population["wf"]
    phi = occupied_ao_orbitals(wf)
    full = walker_full_coefficients(wf)
    for s, key in enumerate(("basis_a", "basis_b")):
        nocc = phi[s].shape[2]
        assert full[s].shape == (_NW, _NAO, nocc)
        np.testing.assert_array_equal(full[s][:, :_NCORE, :_NCORE], np.ones((_NW, 1, 1)))
        np.testing.assert_allclose(
            np.einsum("pq,wqi->wpi", wf[key].coeff, full[s]), phi[s], atol=1e-13
        )


def test_transition_spin_diagonal_matches_pyscf(population):
    """On one determinant with orthonormal orbitals the formula is the UHF <S^2>."""
    wf, s_ao = population["wf"], population["s_ao"]
    ca, cb = wf["basis_a"].coeff[:, :_NA], wf["basis_b"].coeff[:, :_NB]
    overlap, s2 = transition_spin(ca, cb, ca, cb, s_ao)
    ref, _ = scf.uhf.spin_square((ca, cb), s_ao)
    assert overlap == pytest.approx(1.0, abs=1e-12)
    assert s2 == pytest.approx(ref, abs=1e-12)
    m = 0.5 * (_NA - _NB)
    assert s2 > m * (m + 1)  # the two orbital sets differ: contaminated


def test_spin_pure_determinants(population):
    """Beta orbitals inside the alpha span: S = m exactly, for any pair of such determinants."""
    wf, s_ao = population["wf"], population["s_ao"]
    phi_a, _ = occupied_ao_orbitals(wf)
    rng = np.random.default_rng(1)
    m = 0.5 * (_NA - _NB)
    dets = []
    for i in range(2):
        mix = rng.normal(size=(_NA, _NB)) + 1j * rng.normal(size=(_NA, _NB))
        dets.append((phi_a[i], phi_a[i] @ mix))
    for bra in dets:
        _, s2 = transition_spin(*bra, *bra, s_ao)
        assert s2 == pytest.approx(m * (m + 1), abs=1e-10)
    _, s2 = transition_spin(*dets[0], *dets[1], s_ao)
    assert s2 == pytest.approx(m * (m + 1), abs=1e-9)


def test_pair_spin_matches_transition_spin(population):
    wf, s_ao = population["wf"], population["s_ao"]
    phi_a, phi_b = occupied_ao_orbitals(wf)
    ov, s2 = pair_spin(phi_a[0], phi_b[0], s_ao @ phi_a, s_ao @ phi_b)
    for j in range(_NW):
        ov_j, s2_j = transition_spin(phi_a[0], phi_b[0], phi_a[j], phi_b[j], s_ao)
        assert ov[j] == pytest.approx(ov_j, rel=1e-12)
        assert s2[j] == pytest.approx(s2_j, rel=1e-12)


def test_population_spin_matches_the_pair_loop(population):
    wf, s_ao = population["wf"], population["s_ao"]
    phi_a, phi_b = occupied_ao_orbitals(wf)
    guide_a, guide_b = wf["basis_a"].coeff[:, :_NA], wf["basis_b"].coeff[:, :_NB]
    r = population_spin(wf, s_ao, guide_a, guide_b)

    w, c = wf["weights"], wf["weights"] / wf["overlaps"]
    s2_g = np.array(
        [transition_spin(guide_a, guide_b, phi_a[i], phi_b[i], s_ao)[1] for i in range(_NW)]
    )
    np.testing.assert_allclose(r["s2_walkers"], s2_g, rtol=1e-12)
    assert r["mixed"] == pytest.approx(np.sum(w * s2_g) / np.sum(w), rel=1e-12)
    num = norm = 0.0
    for i in range(_NW):
        for j in range(_NW):
            ov, s2 = transition_spin(phi_a[i], phi_b[i], phi_a[j], phi_b[j], s_ao)
            norm += np.conj(c[i]) * c[j] * ov
            num += np.conj(c[i]) * c[j] * ov * s2
    assert r["norm"] == pytest.approx(norm, rel=1e-12)
    assert r["pure"] == pytest.approx(num / norm, rel=1e-10)
    # an expectation value of a hermitian operator
    assert abs(r["pure"].imag) < 1e-10 and abs(r["norm"].imag) < 1e-10 * abs(r["norm"])
    assert r["sz"] == pytest.approx(0.5 * (_NA - _NB), abs=1e-12)


def test_spin_along_tau_of_a_run(tmp_path):
    """NH2 doublet: tau = 0 is the UHF determinant, S_z = 1/2 in every snapshot."""
    mol = gto.M(
        atom="N 0 0 0; H 1.02259 0 0; H -0.22811936 0.99682088 0",
        basis="sto-6g",
        spin=1,
        verbose=0,
    )
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mycc = cc.CCSD(mf, frozen=1)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    snaps = tmp_path / "snaps"
    n_blocks, n_eql = 4, 2
    af = AfqmcMixed(
        mycc,
        trial="upt2ccsd_bar",
        mixed_precision=False,
        save_wavefunction=snaps,
        dt=0.005,
        n_walkers=6,
        n_prop_steps=4,
        n_blocks=n_blocks,
        n_eql_blocks=n_eql,
        seed=11,
    )
    _quiet(af.kernel)

    s_ao = mol.intor("int1e_ovlp")
    na, nb = mol.nelec
    guide_a, guide_b = guide_from_snapshots(snaps)
    assert guide_a.shape == (mol.nao, na) and guide_b.shape == (mol.nao, nb)
    s2_uhf, _ = mf.spin_square()
    _, s2_guide = transition_spin(guide_a, guide_b, guide_a, guide_b, s_ao)
    assert s2_guide == pytest.approx(s2_uhf, abs=1e-8)

    rows = spin_along_tau(snaps, s_ao)
    assert len(rows) == 1 + n_eql + n_blocks
    assert [r["tau"] for r in rows] == sorted(r["tau"] for r in rows)
    assert rows[0]["phase"] == "init"
    assert rows[0]["s2_mixed"] == pytest.approx(s2_uhf, abs=1e-8)
    assert rows[0]["s2_pure"] == pytest.approx(s2_uhf, abs=1e-8)
    for r in rows:
        assert r["sz"] == pytest.approx(0.5, abs=1e-10)
        assert abs(r["pure_imag"]) < 1e-8
        assert r["s2_walker_min"] <= r["s2_mixed"] <= r["s2_walker_max"]
        assert 0.7 < r["s2_pure"] < 0.8
    # an explicit guide gives the same rows
    rows2 = spin_along_tau(snaps, s_ao, guide=(guide_a, guide_b))
    assert rows2[-1]["s2_mixed"] == pytest.approx(rows[-1]["s2_mixed"], rel=1e-12)
