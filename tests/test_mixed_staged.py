"""
AfqmcMixed from a staged file: save_staged writes the hamiltonian, the guide and the
measurement trial of a (guide, trial) pair, and from_staged runs from that file alone,
with no mean field or CC object, reproducing the run made from the CC object.
"""

from trot import config

config.configure_once()

import contextlib
import io
from typing import Any

import h5py
import jax
import numpy as np
import pytest
from pyscf import cc, gto, scf

from trot.afqmc import Afqmc, AfqmcMixed
from trot.wavefunction_io import load_wavefunction

jax.config.update("jax_enable_x64", True)

_PARAMS: dict[str, Any] = dict(
    dt=0.005, n_walkers=6, n_prop_steps=4, n_blocks=8, n_eql_blocks=2, seed=11
)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def h2o_cc():
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


@pytest.mark.parametrize(
    "system,guide,trial",
    [
        ("h2o_cc", "rhf", "pt2ccsd_bar"),
        ("h2o_cc", "cisd", "pt2ccsd"),
        ("nh2_ucc", "uhf", "upt2ccsd_bar"),
        ("nh2_ucc", "ucisd", "upt2ccsd"),
    ],
)
def test_from_staged_reproduces_the_run(system, guide, trial, tmp_path, request):
    mycc = request.getfixturevalue(system)
    direct = AfqmcMixed(mycc, guide=guide, trial=trial, mixed_precision=False, **_PARAMS)
    e_d, err_d = _quiet(direct.kernel)

    # the file is written by staging alone: no job is built
    writer = AfqmcMixed(mycc, guide=guide, trial=trial)
    path = _quiet(writer.save_staged, tmp_path / "mixed.h5")
    assert writer._job is None and path.exists()

    af = _quiet(AfqmcMixed.from_staged, path, mixed_precision=False, **_PARAMS)
    assert af._scf is None and af._cc is None
    assert (af.guide, af.trial) == (guide, trial)
    e_s, err_s = _quiet(af.kernel)
    assert e_s == pytest.approx(e_d, abs=1e-10)
    # nan for both when the run is too short for a blocking error
    assert err_s == pytest.approx(err_d, abs=1e-10, nan_ok=True)
    assert af.guide_e_tot == pytest.approx(direct.guide_e_tot, abs=1e-10)

    # the staged inputs are the ones the CC object gives
    np.testing.assert_array_equal(
        np.asarray(af.job.ham_data.h0), np.asarray(direct.job.ham_data.h0)
    )
    for key, value in direct.trial_input.data.items():
        np.testing.assert_array_equal(af.trial_input.data[key], np.asarray(value))
    assert af.job.staged.meta["chol_cut"] == direct.job.staged.meta["chol_cut"]

    # the wavefunction files need no mean field either: the basis comes from the file
    wf_s = load_wavefunction(af.save_wavefunction(tmp_path / "wf_staged.h5"))
    wf_d = load_wavefunction(direct.save_wavefunction(tmp_path / "wf_direct.h5"))
    for key in ("basis", "basis_a", "basis_b"):
        if key in wf_d:
            np.testing.assert_array_equal(wf_s[key].coeff, wf_d[key].coeff)
            np.testing.assert_array_equal(wf_s[key].frozen_occ, wf_d[key].frozen_occ)
            np.testing.assert_array_equal(wf_s[key].active, wf_d[key].active)
    for key in ("weights", "overlaps"):
        np.testing.assert_allclose(wf_s[key], wf_d[key], atol=1e-12)
    for key in ("nelectron", "spin", "charge", "guide", "trial"):
        assert wf_s["attrs"][key] == wf_d["attrs"][key]


def test_from_staged_trial_and_guide_choice(nh2_ucc, h2o_cc, tmp_path):
    path = _quiet(
        AfqmcMixed(nh2_ucc, guide="uhf", trial="upt2ccsd_bar").save_staged, tmp_path / "u.h5"
    )
    with h5py.File(path, "r") as f:
        assert f["mixed"].attrs["guide"] == "uhf" and f["mixed"].attrs["trial"] == "upt2ccsd_bar"
        assert set(f["mixed_trial/data"].keys()) == {"mo_t_a", "mo_t_b", "t2aa", "t2ab", "t2bb"}
        assert "basis_a" in f["mixed"] and "basis_b" in f["mixed"]
        assert f["mixed"].attrs["e_mf"] == pytest.approx(nh2_ucc._scf.e_tot)

    # the other estimator of the same staged trial runs from the same file
    other = _quiet(AfqmcMixed.from_staged, path, trial="upt2ccsd", mixed_precision=False, **_PARAMS)
    ref = AfqmcMixed(nh2_ucc, guide="uhf", trial="upt2ccsd", mixed_precision=False, **_PARAMS)
    assert _quiet(other.kernel)[0] == pytest.approx(_quiet(ref.kernel)[0], abs=1e-10)

    # a guide the file was not staged for, and a trial on the other hamiltonian
    with pytest.raises(ValueError, match="guide='ucisd' needs staged data"):
        _quiet(AfqmcMixed.from_staged, path, guide="ucisd")
    with pytest.raises(ValueError):
        _quiet(AfqmcMixed.from_staged, path, trial="pt2ccsd_bar")
    with pytest.raises(ValueError, match="tau_eql or n_eql_blocks"):
        AfqmcMixed.from_staged(path, tau_eql=1.0, n_eql_blocks=2)

    # tau_eql is honoured on a staged run
    af = _quiet(AfqmcMixed.from_staged, path, tau_eql=0.1, dt=0.005, n_prop_steps=4)
    assert _quiet(af.build_job).params.n_eql_blocks == 5

    # a plain staged file has no measurement trial
    plain = tmp_path / "plain.h5"
    _quiet(Afqmc(h2o_cc._scf).save_staged, plain)
    with pytest.raises(ValueError, match="carries no measurement trial"):
        AfqmcMixed.from_staged(plain)
