"""
The single determinant and CISD trials of the mixed guide/trial API (trot/mixed.py
"rhf" and "cisd"): one component kernels whose energy_fn returns the local energy, so
that the energy a plain Afqmc run with that trial would measure can be taken under any
guide. The tau = 0 energies of the HF population are exact: E_RHF for the RHF trial and
E_CCSD for the CISD trial built from the CC amplitudes (its singles and doubles are the
CCSD ones, and <HF|H|Psi> only sees those).
"""

from trot import config

config.configure_once()

import contextlib
import io

import jax
import numpy as np
import pytest
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed
from trot.mixed import (
    ONE_COMPONENT,
    available_mixed_recipes,
    get_mixed_recipe,
    single_component_energy,
    stage_cisd_trial,
    stage_rhf_trial,
)

jax.config.update("jax_enable_x64", True)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def h4():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()
    return mycc


def test_registry():
    pairs = available_mixed_recipes()
    for pair in (("rhf", "rhf"), ("cisd", "rhf"), ("rhf", "cisd"), ("cisd", "cisd")):
        assert pair in pairs
    for name in ("rhf", "cisd"):
        rec = get_mixed_recipe(name)
        assert rec.guide == "rhf" and rec.components == ONE_COMPONENT
        assert rec.energy_fn is single_component_energy
        assert rec.walker_kind == "restricted" and rec.ham_basis == "restricted"
        assert rec.plan_chunking is None
    assert single_component_energy(0.0, np.array([[1.5], [2.5]])).tolist() == [1.5, 2.5]


def test_stagers(h4):
    tr = stage_rhf_trial(h4)
    assert tr.kind == "rhf" and "mo" in tr.data and tr.source_kind == "mf"
    tc = stage_cisd_trial(h4)
    assert tc.kind == "cisd" and {"ci1", "ci2"} <= set(tc.data)
    np.testing.assert_allclose(tc.data["ci1"], h4.t1)


@pytest.mark.parametrize("guide", ["rhf", "cisd"])
@pytest.mark.parametrize("trial", ["rhf", "cisd"])
def test_tau0_energies(h4, guide, trial):
    af = AfqmcMixed(
        h4, guide=guide, trial=trial, n_walkers=4, n_blocks=1, n_eql_blocks=1, seed=3, chol_cut=1e-9
    )
    job = _quiet(af.build_job)
    from trot.driver_mixed import _init_trial_energy

    state = job.prop_ops.init_prop_state(
        sys=job.sys,
        ham_data=job.ham_data,
        trial_ops=job.trial_ops,
        trial_data=job.trial_data,
        meas_ops=job.meas_ops,
        params=job.params,
    )
    e0, _ = _init_trial_energy(
        state,
        job.ham_data,
        job.mix_trial_data,
        job.mix_trial_meas_ops,
        job.mix_meas_ctx(),
        job.params,
        job.recipe.components,
        job.recipe.energy_fn,
    )
    expected = h4._scf.e_tot if trial == "rhf" else h4.e_tot
    np.testing.assert_allclose(float(np.real(e0)), expected, atol=1e-8)


def test_short_run(h4):
    af = AfqmcMixed(
        h4,
        guide="rhf",
        trial="cisd",
        n_walkers=50,
        n_blocks=20,
        n_eql_blocks=10,
        seed=5,
        n_prop_steps=10,
    )
    e, err = _quiet(af.kernel)
    assert np.isfinite(e) and np.isfinite(err)
    assert af.qmc_result.trial_component_names == ONE_COMPONENT
    assert af.qmc_result.trial_block_components["energy"].shape == (20,)
    # the guide energy is the RHF estimator's, the trial energy the CISD one; a short
    # run only checks that both land in the right neighbourhood of E_CCSD
    assert abs(e - h4.e_tot) < 0.05 and abs(af.guide_e_tot - h4.e_tot) < 0.05
