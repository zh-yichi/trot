"""
Mixed guide/trial AFQMC beyond pt2CCSD: any registered guide with any registered trial
through one block function, driver and statistics.

The strongest check is that a mixed run whose guide and trial are the same wavefunction
reproduces the plain Afqmc run block for block: same walkers, same weights, same energies.
"""

from __future__ import annotations

import contextlib
import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.afqmc import Afqmc, AfqmcMixed
from trot.driver import _init_trial_energy
from trot.mixed import (
    GUIDES,
    TRIALS,
    available_guides,
    available_mixed_recipes,
    available_trials,
    get_mixed_recipe,
)
from trot.stat_utils import (
    clean_components,
    clean_pt2ccsd,
    make_component_blocking,
    pt2ccsd_blocking,
    pt2ccsd_energy_fn,
)

pytestmark = pytest.mark.slow

jax.config.update("jax_enable_x64", True)

_PARAMS = dict(dt=0.01, n_walkers=4, n_prop_steps=2, n_blocks=12, n_eql_blocks=2, seed=11)


@pytest.fixture(scope="module")
def h4():
    from pyscf import cc, gto, scf

    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8", basis="sto-6g", unit="b", verbose=0
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    return {"mf": mf, "cc": mycc}


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_registry_and_pairing():
    assert set(available_guides()) == {"rhf", "uhf", "cisd", "ucisd"}
    for name in ("rhf", "uhf", "cisd", "ucisd", "pt2ccsd", "upt2ccsd"):
        assert name in available_trials()
    pairs = available_mixed_recipes()
    for pair in (
        ("rhf", "pt2ccsd"),
        ("cisd", "pt2ccsd"),
        ("rhf", "cisd"),
        ("cisd", "rhf"),
        ("cisd", "cisd"),
        ("uhf", "upt2ccsd"),
        ("uhf", "ucisd"),
        ("ucisd", "uhf"),
    ):
        assert pair in pairs

    # the default guide is the corresponding HF
    assert get_mixed_recipe("cisd").guide == "rhf"
    assert get_mixed_recipe("ucisd").guide == "uhf"
    assert get_mixed_recipe("upt2ccsd").guide == "uhf"

    rec = get_mixed_recipe("pt2ccsd", guide="cisd")
    assert rec.walker_kind == "restricted" and rec.components == ("t2", "e0", "e1")
    rec = get_mixed_recipe("rhf", guide="cisd")
    assert rec.components == ("e_loc",)

    # incompatible hamiltonians: the CISD guide is on the one-basis hamiltonian, the
    # unrestricted pt2CCSD trial on the two-basis one
    with pytest.raises(ValueError, match="hamiltonians"):
        get_mixed_recipe("upt2ccsd", guide="cisd")
    # no walker kind in common
    with pytest.raises(ValueError, match="walker kind"):
        get_mixed_recipe("cisd", guide="ucisd")
    with pytest.raises(ValueError, match="unknown mixed guide"):
        get_mixed_recipe("pt2ccsd", guide="no_such_guide")


def test_trial_kwargs_typo_is_an_error(h4):
    af = AfqmcMixed(h4["cc"], trial="rhf", trial_kwargs={"n_chol_head": 3}, **_PARAMS)
    with pytest.raises(ValueError, match="takes no option"):
        _quiet(af.build_job)


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------


def test_component_blocking_matches_pt2ccsd_blocking():
    rng = np.random.default_rng(3)
    n = 60
    w = jnp.asarray(rng.normal(1.0, 0.1, n) + 1j * rng.normal(0.0, 0.02, n))
    t2 = jnp.asarray(rng.normal(0.1, 0.01, n) + 1j * rng.normal(0.0, 0.002, n))
    e0 = jnp.asarray(rng.normal(-2.0, 0.05, n) + 1j * rng.normal(0.0, 0.01, n))
    e1 = jnp.asarray(rng.normal(-0.1, 0.01, n) + 1j * rng.normal(0.0, 0.002, n))
    h0 = 0.7
    generic = make_component_blocking(pt2ccsd_energy_fn)
    for final in (False, True):
        ref = pt2ccsd_blocking(h0, w, t2, e0, e1, final=final)
        got = generic(h0, w, t2, e0, e1, final=final)
        assert float(got[0]) == pytest.approx(float(ref[0]), abs=1e-12)
        assert float(got[1]) == pytest.approx(float(ref[1]), rel=1e-9)

    e_sp = (h0 + e0 + e1 - t2 * e0).real
    ref_clean = clean_pt2ccsd(e_sp, w, t2, e0, e1)
    got_clean = clean_components(e_sp, w, t2, e0, e1)
    for a, b in zip(ref_clean, got_clean):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_eloc_blocking_is_the_weighted_mean():
    from trot.stat_utils import eloc_energy_fn

    rng = np.random.default_rng(5)
    n = 40
    w = jnp.asarray(rng.normal(1.0, 0.1, n) + 0j)
    e = jnp.asarray(rng.normal(-1.0, 0.05, n) + 0j)
    blocking = make_component_blocking(eloc_energy_fn)
    mean, err = blocking(0.0, w, e, final=False)
    assert float(mean) == pytest.approx(float((jnp.sum(w * e) / jnp.sum(w)).real), abs=1e-12)
    # delta method of a ratio of means: the influence values
    wm, em = jnp.sum(w), jnp.sum(w * e)
    infl = ((w * e) / wm - em * w / wm**2).real
    ref = float(jnp.sqrt(jnp.sum(infl**2) * n / (n - 1)))
    assert float(err) == pytest.approx(ref, rel=1e-9)


# ---------------------------------------------------------------------------
# guide == trial reproduces the plain run
# ---------------------------------------------------------------------------


def _plain(obj, **kw):
    # Afqmc takes no n_prop_steps argument but its params are built from attributes
    params = {k: v for k, v in _PARAMS.items() if k != "n_prop_steps"}
    af = Afqmc(obj, **params, **kw)
    af.n_prop_steps = _PARAMS["n_prop_steps"]
    af.mixed_precision = False
    af.e_tot, af.e_err = _quiet(af.kernel)
    return af


def _mixed(obj, guide, trial, **kw):
    af = AfqmcMixed(obj, guide=guide, trial=trial, **_PARAMS, **kw)
    _quiet(af.kernel)
    return af


def test_rhf_guide_rhf_trial_reproduces_plain_afqmc(h4):
    plain = _plain(h4["mf"])
    mixed = _mixed(h4["mf"], "rhf", "rhf")
    res = mixed.qmc_result
    # the guide side is the plain run: same walkers, same weights, same block energies
    ref = plain.qmc_result
    np.testing.assert_allclose(
        np.asarray(res.guide_block_energies), np.asarray(ref.block_energies), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(res.guide_block_weights), np.asarray(ref.block_weights), rtol=0, atol=1e-12
    )
    # and the trial, being the same wavefunction, measures the same local energies
    n_eq = _PARAMS["n_eql_blocks"] + 1
    trial_e = np.asarray(res.trial_block_components["e_loc"]).real
    np.testing.assert_allclose(trial_e, np.asarray(ref.block_energies)[n_eq:], atol=1e-10)
    assert mixed.e_tot == pytest.approx(plain.e_tot, abs=1e-8)


def test_cisd_guide_cisd_trial_reproduces_plain_afqmc(h4):
    plain = _plain(h4["cc"])
    mixed = _mixed(h4["cc"], "cisd", "cisd", memory_mode="high")
    res = mixed.qmc_result
    ref = plain.qmc_result
    np.testing.assert_allclose(
        np.asarray(res.guide_block_energies), np.asarray(ref.block_energies), rtol=0, atol=1e-12
    )
    n_eq = _PARAMS["n_eql_blocks"] + 1
    trial_e = np.asarray(res.trial_block_components["e_loc"]).real
    np.testing.assert_allclose(trial_e, np.asarray(ref.block_energies)[n_eq:], atol=1e-10)
    assert mixed.e_tot == pytest.approx(plain.e_tot, abs=1e-8)


# ---------------------------------------------------------------------------
# tau = 0: the trial energy of the initial (guide) walkers
# ---------------------------------------------------------------------------


def _tau0(af):
    job = _quiet(af.build_job)
    state = job.prop_ops.init_prop_state(
        sys=job.sys,
        ham_data=job.ham_data,
        trial_ops=job.trial_ops,
        trial_data=job.trial_data,
        meas_ops=job.meas_ops,
        params=job.params,
    )
    e, w = _init_trial_energy(
        state,
        job.ham_data,
        job.mix_trial_data,
        job.mix_trial_meas_ops,
        job.mix_meas_ctx(),
        job.params,
        job.recipe.components,
        job.recipe.energy_fn,
    )
    return float(e.real), complex(w)


def test_tau0_energies_under_the_hf_guide(h4):
    mf, mycc = h4["mf"], h4["cc"]
    # HF walkers measured against HF: E_HF
    # tolerances at the cholesky truncation level (chol_cut = 1e-5)
    e, w = _tau0(AfqmcMixed(mf, trial="rhf", **_PARAMS))
    assert e == pytest.approx(mf.e_tot, abs=1e-6)
    assert w == pytest.approx(_PARAMS["n_walkers"], abs=1e-10)
    # HF walkers against the CC-derived CISD: <CISD|H|HF>/<CISD|HF> is the CCSD energy
    e, _ = _tau0(AfqmcMixed(mycc, trial="cisd", **_PARAMS))
    assert e == pytest.approx(mycc.e_tot, abs=1e-6)
    # and against pt2CCSD, the same
    e, _ = _tau0(AfqmcMixed(mycc, trial="pt2ccsd", **_PARAMS))
    assert e == pytest.approx(mycc.e_tot, abs=1e-6)


# ---------------------------------------------------------------------------
# the new pairs run
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("guide,trial", [("cisd", "pt2ccsd"), ("cisd", "rhf"), ("rhf", "cisd")])
def test_new_pairs_run(h4, guide, trial):
    af = _mixed(h4["cc"], guide, trial)
    assert np.isfinite(af.e_tot) and np.isfinite(af.e_err)
    assert np.isfinite(af.guide_e_tot)
    res = af.qmc_result
    assert set(res.trial_block_components) == set(af.recipe.components)
    assert res.trial_block_weights.shape[0] == _PARAMS["n_blocks"]
    if trial.startswith("pt2ccsd"):
        assert res.trial_block_t2s is not None
    else:
        assert res.trial_block_t2s is None


def test_guide_needs_cc_object(h4):
    with pytest.raises(ValueError, match="needs a pyscf CC object"):
        AfqmcMixed(h4["mf"], guide="cisd", trial="rhf")
    with pytest.raises(ValueError, match="needs a pyscf CC object"):
        AfqmcMixed(h4["mf"], trial="cisd")
