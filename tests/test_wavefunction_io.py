"""
Saving the guided AFQMC wavefunction: the final walker population with its guide
overlaps and weights, and the orbital basis (with the frozen core) it is expressed in,
from AfqmcMixed on the restricted and the unrestricted hamiltonian.
"""

from trot import config

config.configure_once()

import contextlib
import io
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed
from trot.wavefunction_io import (
    WavefunctionBasis,
    dump_wavefunction,
    load_wavefunction,
    walker_ao_coefficients,
)

jax.config.update("jax_enable_x64", True)

_PARAMS: dict[str, Any] = dict(
    dt=0.005, n_walkers=6, n_prop_steps=4, n_blocks=8, n_eql_blocks=2, seed=11
)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def h2o_frozen():
    mol = gto.M(
        atom="O 0 0 0; H 0.95623 0 0; H -0.23537916 0.92680767 0", basis="sto-6g", verbose=0
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mycc = cc.CCSD(mf, frozen=1)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def nh2_ucc():
    mol = gto.M(
        atom="N 0 0 0; H 1.02259 0 0; H -0.22811936 0.99682088 0",
        basis="sto-6g",
        spin=1,
        verbose=0,
    )
    mf = scf.UHF(mol).newton()
    mf.conv_tol = 1e-12
    mf.kernel()
    mycc = cc.UCCSD(mf, frozen=1)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    return mycc


def test_basis_layouts():
    coeff = np.arange(12.0).reshape(3, 4)
    b = WavefunctionBasis.leading_core(coeff, 1)
    assert b.norb == 3 and b.frozen_occ.tolist() == [0] and b.active.tolist() == [1, 2, 3]
    np.testing.assert_array_equal(b.active_coeff, coeff[:, 1:])
    np.testing.assert_array_equal(b.frozen_occ_coeff, coeff[:, :1])
    # an explicit frozen list splits at the occupied count: [frz_occ | act | frz_vir]
    b = WavefunctionBasis.from_frozen_indices(coeff, [0, 3], nocc_full=2, kind="lno")
    assert b.active.tolist() == [1, 2] and b.frozen_occ.tolist() == [0]
    assert b.frozen_vir.tolist() == [3] and b.kind == "lno"
    with pytest.raises(ValueError, match="active and frozen"):
        WavefunctionBasis(coeff, active=[0, 1], frozen_occ=[1], frozen_vir=[])
    with pytest.raises(ValueError, match="orbitals but the basis"):
        dump_wavefunction(
            "/nonexistent/x.h5",
            walkers=np.zeros((2, 4, 1)),
            weights=np.ones(2),
            overlaps=np.ones(2),
            walker_kind="restricted",
            basis=b,
            nelec=(1, 1),
        )


def _guide_overlaps(af, walkers):
    job = af.job
    return np.asarray(jax.vmap(job.meas_ops.overlap, in_axes=(0, None))(walkers, job.trial_data))


def test_restricted_snapshots_and_reconstruct(h2o_frozen, tmp_path):
    """
    A snapshot is written at tau = 0 and at every printed row, in order of tau, and listed
    in snapshots.json. The last one is the run's final population,
    |psi> = sum_i w_i/<G|phi_i> |phi_i>, in the active canonical MOs with the frozen core
    as the leading column, and the loaded overlaps are the guide's on the saved walkers.
    """
    import json

    snaps = tmp_path / "snaps"
    af = AfqmcMixed(
        h2o_frozen, trial="pt2ccsd_bar", mixed_precision=False, save_wavefunction=snaps, **_PARAMS
    )
    _quiet(af.kernel)
    # 1 (tau = 0) + one row per equilibration block + one per sampling block here
    n_snap = 1 + _PARAMS["n_eql_blocks"] + _PARAMS["n_blocks"]
    files = sorted(snaps.glob("wfn_*.h5"))
    assert [f.name for f in files] == [f"wfn_{k:04d}.h5" for k in range(n_snap)]
    manifest = json.loads((snaps / "snapshots.json").read_text())
    assert [m["file"] for m in manifest] == [f.name for f in files]
    block_time = _PARAMS["dt"] * _PARAMS["n_prop_steps"]
    taus = [m["tau"] for m in manifest]
    assert taus == pytest.approx([k * block_time for k in range(n_snap)])
    assert manifest[0]["phase"] == "init" and manifest[1]["phase"] == "eql"
    assert manifest[-1]["phase"] == "sample" and manifest[-1]["block"] == _PARAMS["n_blocks"]
    assert "trial_energy" in manifest[-1] and "guide_energy" in manifest[0]
    # tau = 0: the initial walkers, the reference determinant, with unit guide overlap
    wf0 = load_wavefunction(files[0])
    assert wf0["attrs"]["snapshot"] == 0 and wf0["attrs"]["tau"] == 0.0
    np.testing.assert_allclose(wf0["overlaps"], 1.0, atol=1e-12)
    # the last snapshot is the final population
    path = files[-1]
    wf = load_wavefunction(path)
    assert wf["attrs"]["snapshot"] == n_snap - 1 and wf["attrs"]["phase"] == "sample"
    mf = h2o_frozen._scf
    nmo, nocc_full = mf.mo_coeff.shape[1], mf.mol.nelec[0]
    assert wf["walker_kind"] == "restricted" and wf["nelec"] == (nocc_full - 1, nocc_full - 1)
    n = _PARAMS["n_walkers"]
    assert wf["walkers"].shape == (n, nmo - 1, nocc_full - 1)
    assert wf["weights"].shape == (n,) and wf["overlaps"].shape == (n,)
    basis = wf["basis"]
    assert basis.kind == "canonical_mo" and basis.norb == nmo - 1
    assert basis.frozen_occ.tolist() == [0] and basis.frozen_vir.size == 0
    np.testing.assert_allclose(basis.coeff, mf.mo_coeff, atol=1e-14)
    np.testing.assert_allclose(basis.active_coeff, mf.mo_coeff[:, 1:], atol=1e-14)
    # the population is the run's final state
    state = af.qmc_result.final_state
    np.testing.assert_allclose(wf["walkers"], np.asarray(state.walkers), atol=1e-14)
    np.testing.assert_allclose(wf["weights"], np.asarray(state.weights), atol=1e-14)
    np.testing.assert_allclose(wf["overlaps"], _guide_overlaps(af, jnp.asarray(wf["walkers"])))
    # the walkers' orbitals in the AO basis are orthonormal to the frozen core
    ao = walker_ao_coefficients(wf)
    s1e = mf.get_ovlp()
    assert ao.shape == (n, mf.mol.nao, nocc_full - 1)
    core = basis.frozen_occ_coeff
    assert np.abs(np.einsum("pi,pq,wqj->wij", core, s1e, ao)).max() < 1e-10
    # the metadata
    a = wf["attrs"]
    assert a["guide"] == "rhf" and a["trial"] == "pt2ccsd_bar" and a["ham_basis"] == "restricted"
    assert a["n_blocks"] == _PARAMS["n_blocks"] and a["seed"] == _PARAMS["seed"]
    assert a["tau"] == pytest.approx((2 + 8) * 4 * 0.005)
    assert a["nelectron"] == mf.mol.nelectron
    # the same population from the method, with the final energies
    path2 = af.save_wavefunction(tmp_path / "wf2.h5")
    wf2 = load_wavefunction(path2)
    np.testing.assert_allclose(wf2["walkers"], wf["walkers"])
    assert wf2["attrs"]["phase"] == "final" and wf2["attrs"]["e_tot"] == pytest.approx(af.e_tot)


def test_unrestricted_save_and_reconstruct(nh2_ucc, tmp_path):
    """The uchol hamiltonian: one basis and one walker set per spin, the same frozen core."""
    af = AfqmcMixed(nh2_ucc, trial="upt2ccsd_bar", mixed_precision=False, **_PARAMS)
    _quiet(af.kernel)
    path = af.save_wavefunction(tmp_path / "wf_u.h5")
    wf = load_wavefunction(path)
    mf = nh2_ucc._scf
    nmo = mf.mo_coeff[0].shape[1]
    na, nb = mf.mol.nelec
    assert wf["walker_kind"] == "unrestricted" and wf["nelec"] == (na - 1, nb - 1)
    n = _PARAMS["n_walkers"]
    assert wf["walkers_a"].shape == (n, nmo - 1, na - 1)
    assert wf["walkers_b"].shape == (n, nmo - 1, nb - 1)
    for s, key in enumerate(("basis_a", "basis_b")):
        b = wf[key]
        assert b.frozen_occ.tolist() == [0] and b.norb == nmo - 1
        np.testing.assert_allclose(b.coeff, mf.mo_coeff[s], atol=1e-14)
    np.testing.assert_allclose(
        wf["overlaps"],
        _guide_overlaps(af, (jnp.asarray(wf["walkers_a"]), jnp.asarray(wf["walkers_b"]))),
    )
    ao_a, ao_b = walker_ao_coefficients(wf)
    assert ao_a.shape == (n, mf.mol.nao, na - 1) and ao_b.shape == (n, mf.mol.nao, nb - 1)
    assert wf["attrs"]["ham_basis"] == "uchol" and wf["attrs"]["nelec"] == [na - 1, nb - 1]


def test_save_needs_a_finished_run(h2o_frozen, tmp_path):
    import h5py

    from trot.wavefunction_io import WAVEFUNCTION_SNAPSHOT_DIR

    af = AfqmcMixed(h2o_frozen, trial="pt2ccsd_bar", **_PARAMS)
    assert af.wavefunction_dir is None
    assert AfqmcMixed(h2o_frozen, save_wavefunction=True, **_PARAMS).wavefunction_dir == Path(
        WAVEFUNCTION_SNAPSHOT_DIR
    )
    with pytest.raises(RuntimeError, match="kernel"):
        af.save_wavefunction(tmp_path / "x.h5")
    with h5py.File(tmp_path / "other.h5", "w") as f:
        f.create_dataset("x", data=np.zeros(2))
    with pytest.raises(ValueError, match="not a trot_afqmc_wavefunction"):
        load_wavefunction(tmp_path / "other.h5")
