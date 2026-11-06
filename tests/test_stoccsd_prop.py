"""
Propagation under the stochastically sampled CCSD guide (prop/afqmc_stocc.py,
prop/blocks_mixed_stocc.py) and its registration as the mixed guide "stoccsd": with no
T2 modes and one determinant the walk is the RHF-guided one with the Thouless reference
as the guide; the reconfiguration carries each walker's sample; AfqmcMixed runs end to
end with the rhf, cisd and pt2ccsd trials.
"""

from trot import config

config.configure_once()

import contextlib
import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed
from trot.core.system import System
from trot.meas.rhf import make_rhf_meas_ops
from trot.meas.stoccsd import make_stoccsd_meas_ops
from trot.mixed import GUIDES, available_mixed_recipes, get_mixed_recipe
from trot.prop.afqmc import make_prop_ops
from trot.prop.afqmc_stocc import (
    PropStateStocc,
    guide_overlaps,
    init_prop_state_stocc,
    make_stocc_prop_ops,
    sample_walker_dets,
)
from trot.prop.blocks_mixed_stocc import _reconfigure, block_mixed_stocc
from trot.prop.types import PropState, QmcParams
from trot.testing import make_random_ham_chol
from trot.trial.rhf import RhfTrial, make_rhf_trial_ops
from trot.trial.stoccsd import StoccsdGuide, decompose_t2, make_stoccsd_trial_ops
from trot.walkers import init_walkers

jax.config.update("jax_enable_x64", True)

_NORB, _NOCC, _NCHOL, _NW = 6, 2, 7, 5


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _thouless_np(nocc, norb, t1):
    exp_t1 = np.eye(norb)
    exp_t1[:nocc, nocc:] = t1
    return exp_t1.T @ np.eye(norb)[:, :nocc]


@pytest.fixture(scope="module")
def case():
    key = jax.random.PRNGKey(31)
    k_ham, k_t1, k_t2 = jax.random.split(key, 3)
    ham = make_random_ham_chol(k_ham, norb=_NORB, n_chol=_NCHOL)
    nvir = _NORB - _NOCC
    t1 = 0.05 * np.asarray(jax.random.normal(k_t1, (_NOCC, nvir)))
    t2 = 0.05 * np.asarray(jax.random.normal(k_t2, (_NOCC, nvir, _NOCC, nvir)))
    t2 = 0.5 * (t2 + t2.transpose(2, 3, 0, 1))
    mo_t = jnp.asarray(_thouless_np(_NOCC, _NORB, t1))
    sys = System(norb=_NORB, nelec=(_NOCC, _NOCC), walker_kind="restricted")
    params = QmcParams(dt=0.01, n_walkers=_NW, n_prop_steps=4, n_chunks=1, seed=7)
    return dict(ham=ham, sys=sys, params=params, mo_t=mo_t, t2=t2)


def _guide(case, tau, n_samples, key):
    mo_t = case["mo_t"]
    dets = jax.vmap(lambda k: jnp.asarray(mo_t, dtype=jnp.complex128))(jnp.arange(n_samples))
    g = StoccsdGuide(mo_t=mo_t, tau=jnp.asarray(tau, dtype=jnp.complex128), dets=dets)
    return g


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------


def test_step_with_no_modes_is_the_rhf_guided_walk(case):
    """
    One determinant and no T2 modes: every sample is the Thouless reference, R_G = 1,
    and the fields are drawn from the same subkey as prop.afqmc.afqmc_step, so the two
    walks agree to roundoff (the force bias contractions are ordered differently).
    """
    ham, sys, params, mo_t = case["ham"], case["sys"], case["params"], case["mo_t"]
    tau0 = jnp.zeros((0, _NOCC, _NORB - _NOCC), dtype=jnp.complex128)
    guide = _guide(case, tau0, 1, None)
    rhf = RhfTrial(mo_coeff=jnp.asarray(mo_t, dtype=jnp.complex128))

    rhf_meas = make_rhf_meas_ops(sys)
    sto_meas = make_stoccsd_meas_ops(sys)
    rhf_prop = make_prop_ops("restricted", "restricted")
    sto_prop = make_stocc_prop_ops("restricted", "restricted")
    rdm1 = make_rhf_trial_ops(sys).get_rdm1(rhf)
    prop_ctx = rhf_prop.build_prop_ctx(ham, rdm1, params)
    rhf_ctx = rhf_meas.build_meas_ctx(ham, rhf)
    sto_ctx = sto_meas.build_meas_ctx(ham, guide)

    walkers = init_walkers(sys=sys, rdm1=rdm1, n_walkers=_NW)
    walkers = walkers + 0.05j * jax.random.normal(jax.random.PRNGKey(1), walkers.shape)
    walkers, _ = jnp.linalg.qr(walkers)
    key = jax.random.PRNGKey(params.seed)
    e0 = jnp.asarray(-1.0)
    olp = jax.vmap(rhf_meas.overlap, in_axes=(0, None))(walkers, rhf)
    st_rhf = PropState(walkers, jnp.ones(_NW), olp, key, e0, e0, jnp.asarray(0))
    dets = sample_walker_dets(guide, jax.random.PRNGKey(0), _NW)
    assert dets.shape == (_NW, 1, _NORB, _NOCC)
    np.testing.assert_allclose(dets[:, 0], np.broadcast_to(mo_t, (_NW, _NORB, _NOCC)))
    olp_s = guide_overlaps(sto_meas, walkers, dets, guide, n_chunks=1)
    np.testing.assert_allclose(olp_s, olp, rtol=1e-13)
    st_sto = PropStateStocc(walkers, jnp.ones(_NW), olp_s, key, e0, e0, jnp.asarray(0), dets)

    step_rhf = jax.jit(
        lambda st: rhf_prop.step(
            st,
            params=params,
            ham_data=ham,
            trial_data=rhf,
            trial_ops=None,
            meas_ops=rhf_meas,
            meas_ctx=rhf_ctx,
            prop_ctx=prop_ctx,
        )
    )
    step_sto = jax.jit(
        lambda st: sto_prop.step(
            st,
            params=params,
            ham_data=ham,
            trial_data=guide,
            trial_ops=None,
            meas_ops=sto_meas,
            meas_ctx=sto_ctx,
            prop_ctx=prop_ctx,
        )
    )
    for _ in range(10):
        st_rhf = step_rhf(st_rhf)
        st_sto = step_sto(st_sto)
        np.testing.assert_allclose(st_sto.walkers, st_rhf.walkers, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(st_sto.weights, st_rhf.weights, rtol=1e-10)
        np.testing.assert_allclose(st_sto.overlaps, st_rhf.overlaps, rtol=1e-10)
        np.testing.assert_array_equal(st_sto.rng_key, st_rhf.rng_key)
        assert int(st_sto.node_encounters) == int(st_rhf.node_encounters)
    assert float(jnp.sum(st_sto.weights)) > 0


def test_step_redraws_the_guide_per_walker(case):
    ham, sys, params = case["ham"], case["sys"], case["params"]
    tau = decompose_t2(case["t2"], thresh=1e-12)
    guide = _guide(case, tau, 3, None)
    sto_meas = make_stoccsd_meas_ops(sys)
    sto_prop = make_stocc_prop_ops("restricted", "restricted")
    sto_ctx = sto_meas.build_meas_ctx(ham, guide)
    rdm1 = make_stoccsd_trial_ops(sys).get_rdm1(guide)
    prop_ctx = sto_prop.build_prop_ctx(ham, rdm1, params)
    state = init_prop_state_stocc(
        sys=sys,
        ham_data=ham,
        trial_ops=make_stoccsd_trial_ops(sys),
        trial_data=guide,
        meas_ops=sto_meas,
        params=params,
        meas_ctx=sto_ctx,
    )
    assert state.guide_dets.shape == (_NW, 3, _NORB, _NOCC)
    # every walker has its own sample
    d = state.guide_dets
    assert not np.allclose(d[0], d[1]) and not np.allclose(d[0, 0], d[0, 1])
    np.testing.assert_allclose(
        state.overlaps, guide_overlaps(sto_meas, state.walkers, d, guide, n_chunks=1)
    )
    new = sto_prop.step(
        state,
        params=params,
        ham_data=ham,
        trial_data=guide,
        trial_ops=None,
        meas_ops=sto_meas,
        meas_ctx=sto_ctx,
        prop_ctx=prop_ctx,
    )
    assert not np.allclose(new.guide_dets, state.guide_dets)
    np.testing.assert_allclose(
        new.overlaps, guide_overlaps(sto_meas, new.walkers, new.guide_dets, guide, n_chunks=1)
    )
    assert np.all(np.isfinite(new.weights)) and float(jnp.sum(new.weights)) > 0
    # the walk is deterministic in the key
    again = sto_prop.step(
        state,
        params=params,
        ham_data=ham,
        trial_data=guide,
        trial_ops=None,
        meas_ops=sto_meas,
        meas_ctx=sto_ctx,
        prop_ctx=prop_ctx,
    )
    np.testing.assert_array_equal(again.guide_dets, new.guide_dets)
    np.testing.assert_array_equal(again.weights, new.weights)


def test_reconfiguration_carries_the_samples(case):
    mo_t = case["mo_t"]
    tau = decompose_t2(case["t2"], thresh=1e-12)
    guide = _guide(case, tau, 2, None)
    dets = sample_walker_dets(guide, jax.random.PRNGKey(3), _NW)
    walkers = jnp.stack(
        [jnp.asarray(mo_t, dtype=jnp.complex128) * (1 + 0.1 * i) for i in range(_NW)]
    )
    overlaps = jnp.arange(_NW, dtype=jnp.complex128) + 1.0
    weights = jnp.zeros(_NW).at[2].set(3.0)  # only walker 2 survives
    state = PropStateStocc(
        walkers, weights, overlaps, jax.random.PRNGKey(0), 0.0, 0.0, jnp.asarray(0), dets
    )
    new = _reconfigure(state, jnp.asarray(0.5))
    for w in range(_NW):
        np.testing.assert_array_equal(new.walkers[w], walkers[2])
        np.testing.assert_array_equal(new.guide_dets[w], dets[2])
        assert complex(new.overlaps[w]) == complex(overlaps[2])
    np.testing.assert_allclose(new.weights, 3.0 / _NW)


# ---------------------------------------------------------------------------
# AfqmcMixed
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h4():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    return cc.CCSD(mf).run()


def test_registry():
    spec = GUIDES["stoccsd"]
    assert spec.stage is not None and spec.bundle is not None
    assert spec.mixed_block_fn is block_mixed_stocc
    pairs = available_mixed_recipes()
    for trial in ("rhf", "cisd", "pt2ccsd", "pt2ccsd_bar"):
        assert ("stoccsd", trial) in pairs
        assert get_mixed_recipe(trial, guide="stoccsd").mixed_block_fn is block_mixed_stocc
    assert get_mixed_recipe("pt2ccsd").mixed_block_fn is not block_mixed_stocc
    with pytest.raises(ValueError, match="hamiltonians"):
        get_mixed_recipe("upt2ccsd", guide="stoccsd")


@pytest.mark.parametrize("trial", ["rhf", "cisd", "pt2ccsd"])
def test_afqmc_mixed_stoccsd(h4, trial):
    af = AfqmcMixed(
        h4,
        guide="stoccsd",
        trial=trial,
        guide_n_samples=3,
        guide_t2_thresh=1e-6,
        n_walkers=20,
        n_blocks=20,
        n_eql_blocks=5,
        n_prop_steps=5,
        seed=11,
        chol_cut=1e-8,
    )
    assert af.guide_options == {"n_samples": 3, "t2_thresh": 1e-6}
    job = _quiet(af.build_job)
    assert isinstance(job.trial_data, StoccsdGuide) and job.trial_data.n_samples == 3
    assert job.staged.trial.kind == "stoccsd"
    assert job.recipe.mixed_block_fn is block_mixed_stocc
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        e, err = af.kernel()
    text = out.getvalue()
    assert "guide_options" in text and "stoccsd" in text
    assert np.isfinite(e) and np.isfinite(err)
    assert isinstance(af.qmc_result.final_state, PropStateStocc)
    assert af.qmc_result.final_state.guide_dets.shape == (20, 3, job.sys.norb, 2)
    # tau = 0.5 only: the energies have left E_RHF towards E_CCSD (the rhf trial starts
    # at E_RHF, the CC trials at E_CCSD) and stay in between, within the noise
    e_rhf, e_cc = h4._scf.e_tot, h4.e_tot
    for value in (e, af.guide_e_tot):
        assert e_cc - 0.05 < value < e_rhf + 0.02, (value, e_rhf, e_cc)


def test_tau0_trial_energy_is_exact_under_the_sampled_guide(h4):
    """The HF population's trial energy does not depend on the guide: E_RHF / E_CCSD."""
    from trot.driver_mixed import _init_trial_energy

    for trial, expected in (("rhf", h4._scf.e_tot), ("cisd", h4.e_tot)):
        af = AfqmcMixed(
            h4,
            guide="stoccsd",
            trial=trial,
            guide_n_samples=2,
            n_walkers=4,
            n_blocks=1,
            n_eql_blocks=1,
            seed=3,
            chol_cut=1e-9,
        )
        job = _quiet(af.build_job)
        state = job.prop_ops.init_prop_state(
            sys=job.sys,
            ham_data=job.ham_data,
            trial_ops=job.trial_ops,
            trial_data=job.trial_data,
            meas_ops=job.meas_ops,
            params=job.params,
        )
        assert isinstance(state, PropStateStocc)
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
        np.testing.assert_allclose(float(np.real(e0)), expected, atol=1e-8)
