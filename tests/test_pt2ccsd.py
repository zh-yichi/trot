from trot import config

config.configure_once()

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, scf

from trot.driver_mixed import run_mixed_qmc
from trot.ham.chol import HamChol
from trot.meas.pt2ccsd import (
    Pt2ccsdMeasCfg,
    build_meas_ctx,
    combine_first_order_energy,
    make_pt2ccsd_meas_ops,
)
from trot.meas.rhf import make_rhf_meas_ops
from trot.prop.afqmc import make_prop_ops
from trot.prop.blocks_mixed import block_mixed
from trot.prop.types import QmcParams
from trot.staging import StagedMfOrCc, _stage_pt2ccsd_input, stage
from trot import walkers as wk
from trot.core.ops import k_energy
from trot.core.system import System
from trot.trial.pt2ccsd import Pt2ccsdTrial
from trot.trial.rhf import RhfTrial, make_rhf_trial_ops

# ---------------------------------------------------------------------------
# Module-level fixtures — built once for the whole test file
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h8_system():
    """8 H2 molecules (nc=8, na=2) in sto-6g, well-separated clusters.

    Returns a dict containing the pyscf objects and all pre-built TROT
    inputs so each parametrised test can reuse them without re-running SCF/CC.
    """
    a = 2  # bond length inside each H2 dimer (Bohr)
    d = 100  # centre-to-centre distance between dimers (Bohr)
    unit = "b"
    na, nc = 2, 8
    elmt, basis = "H", "sto6g"

    atoms = ""
    for n in range(nc * na):
        shift = ((n - n % na) // na) * (d - a)
        atoms += f"{elmt} {n*a+shift:.5f} 0.00000 0.00000 \n"

    mol = gto.M(atom=atoms, basis=basis, unit=unit, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()

    mycc = cc.CCSD(mf)
    mycc.kernel()

    # Stage the guide (RHF) Hamiltonian
    staged_guide = stage(mycc._scf)
    ham = staged_guide.ham
    sys = System(norb=int(ham.norb), nelec=ham.nelec, walker_kind="restricted")
    ham_data = HamChol(
        jnp.asarray(ham.h0),
        jnp.asarray(ham.h1),
        jnp.asarray(ham.chol),
        basis=ham.basis,
    )

    # Guide wavefunction (RHF MOs)
    guide_data = RhfTrial(
        mo_coeff=jnp.array(staged_guide.trial.data["mo"][:, : sys.nup]),
    )

    # Trial wavefunction (PT2-CCSD)
    trial_obj = StagedMfOrCc(mycc, frozen=None)
    staged_trial = _stage_pt2ccsd_input(trial_obj)
    trial_data = Pt2ccsdTrial(
        mo_t=jnp.array(staged_trial.data["mo_t"]),
        t2=jnp.array(staged_trial.data["t2"]),
    )

    # Measurement context (built once; seed-independent)
    trial_cfg = Pt2ccsdMeasCfg(memory_mode="low")
    trial_meas_ctx = build_meas_ctx(ham_data, trial_data, trial_cfg)

    # Operator factories
    guide_ops = make_rhf_trial_ops(sys)
    guide_meas_ops = make_rhf_meas_ops(sys)
    guide_prop_ops = make_prop_ops(ham_data.basis, sys.walker_kind)
    trial_meas_ops = make_pt2ccsd_meas_ops(sys, mixed_precision=False)

    return dict(
        mycc=mycc,
        sys=sys,
        ham_data=ham_data,
        guide_data=guide_data,
        trial_data=trial_data,
        trial_meas_ctx=trial_meas_ctx,
        guide_ops=guide_ops,
        guide_meas_ops=guide_meas_ops,
        guide_prop_ops=guide_prop_ops,
        trial_meas_ops=trial_meas_ops,
    )


# ---------------------------------------------------------------------------
# Reference energies (dt=0.005, n_walkers=1, n_prop_steps=1, n_blocks=50,
# n_eql_blocks=1) - generated once from a known-good code version.
# Update only after a deliberate algorithmic change; never update silently.
# The errors of that version came from a delta-method estimate; the mixed driver now
# reports the jackknife blocking error of the component ratios, so only the energies
# (the same weighted component ratios) are compared.
# ---------------------------------------------------------------------------

_REFERENCES = {
    1: -8.771210912150202,
    2: -8.769501242545516,
    3: -8.769769708305297,
    4: -8.771395845249032,
}


# ---------------------------------------------------------------------------
# Parametrised test — one run per seed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [1, 2, 3, 4])
def test_pt2ccsd_energy_matches_reference(h8_system, seed):
    """Run PT2-CCSD AFQMC with a fixed seed and check against a reference value.

    QmcParams: dt=0.005, n_walkers=1, n_prop_steps=1, n_blocks=100, n_eql_blocks=5.
    Reference energies and errors were obtained from a known-good run and are
    used as a regression tripwire — any change to propagation, trial, or
    measurement that alters the trajectory will fail this test.
    """
    s = h8_system
    e_ref = _REFERENCES[seed]

    params = QmcParams(
        dt=0.005,
        n_walkers=1,
        n_prop_steps=1,
        n_blocks=50,
        n_eql_blocks=1,
        seed=seed,
    )

    result = run_mixed_qmc(
        sys=s["sys"],
        params=params,
        ham_data=s["ham_data"],
        guide_data=s["guide_data"],
        guide_ops=s["guide_ops"],
        guide_prop_ops=s["guide_prop_ops"],
        guide_meas_ops=s["guide_meas_ops"],
        trial_data=s["trial_data"],
        trial_meas_ops=s["trial_meas_ops"],
        mix_block_fn=block_mixed,
        energy_fn=combine_first_order_energy,
    )

    e_mean = float(result.trial_mean_energy)
    e_err = float(result.trial_stderr_energy)

    print(f"seed={seed}  E={e_mean:.6f}  err={e_err:.6f}  ref={e_ref:.6f}")

    # Reference values are stored to 6 decimal places; match to that precision
    assert jnp.isclose(jnp.array(e_mean), jnp.array(e_ref), atol=1e-6), (
        f"seed={seed}: energy {e_mean:.6f} != reference {e_ref:.6f} " f"(diff={e_mean - e_ref:.2e})"
    )
    assert jnp.isfinite(e_err)


# ---------------------------------------------------------------------------
# Outlier handling of block_mixed: the guide local energy alone flags a walker, and the
# flag is applied to every sampled quantity of that walker (guide energy and the three
# trial components).
# ---------------------------------------------------------------------------


def _block_mixed_kwargs(s, params):
    guide_meas_ctx = s["guide_meas_ops"].build_meas_ctx(s["ham_data"], s["guide_data"])
    guide_prop_ctx = s["guide_prop_ops"].build_prop_ctx(
        s["ham_data"], s["guide_ops"].get_rdm1(s["guide_data"]), params
    )
    return dict(
        sys=s["sys"],
        params=params,
        ham_data=s["ham_data"],
        guide_data=s["guide_data"],
        guide_ops=s["guide_ops"],
        guide_meas_ops=s["guide_meas_ops"],
        guide_meas_ctx=guide_meas_ctx,
        guide_prop_ops=s["guide_prop_ops"],
        guide_prop_ctx=guide_prop_ctx,
        trial_data=s["trial_data"],
        trial_meas_ops=s["trial_meas_ops"],
        trial_meas_ctx=s["trial_meas_ctx"],
    )


def _per_walker_samples(s, kw, state):
    """The per-walker quantities block_mixed measures, computed independently."""
    walkers = wk.orthonormalize(state.walkers, s["sys"].walker_kind)
    n = kw["params"].n_chunks
    guide_e = wk.vmap_chunked(
        s["guide_meas_ops"].require_kernel(k_energy), n_chunks=n, in_axes=(0, None, None, None)
    )(walkers, s["ham_data"], kw["guide_meas_ctx"], s["guide_data"])
    guide_ov = wk.vmap_chunked(s["guide_meas_ops"].overlap, n_chunks=n, in_axes=(0, None))(
        walkers, s["guide_data"]
    )
    trial_ov = wk.vmap_chunked(s["trial_meas_ops"].overlap, n_chunks=n, in_axes=(0, None))(
        walkers, s["trial_data"]
    )
    comps = wk.vmap_chunked(
        s["trial_meas_ops"].require_kernel(k_energy), n_chunks=n, in_axes=(0, None, None, None)
    )(walkers, s["ham_data"], s["trial_meas_ctx"], s["trial_data"])
    comps = np.asarray(comps).reshape(wk.n_walkers(walkers), -1)
    return (
        np.real(np.asarray(guide_e)),
        np.asarray(guide_ov),
        np.asarray(trial_ov),
        comps,
        np.asarray(state.weights),
    )


def _identity_sr(walkers, weights, zeta, walker_kind):
    return walkers, weights


def test_block_mixed_outliers_follow_the_guide_mask(h8_system):
    """Clamp and reject, checked against a hand-built mask on the guide local energy.

    The population is spread by a few equilibration blocks, then measured with
    n_prop_steps=0 and dt chosen so that sqrt(2/dt) splits the walkers' guide energies
    around their median: part of the population is an outlier, part is not.
    """
    s = h8_system
    params = QmcParams(dt=0.005, n_walkers=12, n_prop_steps=10, seed=7)
    kw = _block_mixed_kwargs(s, params)
    state = s["guide_prop_ops"].init_prop_state(
        sys=s["sys"],
        ham_data=s["ham_data"],
        trial_ops=s["guide_ops"],
        trial_data=s["guide_data"],
        meas_ops=s["guide_meas_ops"],
        params=params,
        meas_ctx=kw["guide_meas_ctx"],
    )
    for _ in range(3):
        state, _ = block_mixed(state, measure_trial=False, **kw)

    guide_e, guide_ov, trial_ov, comps, w = _per_walker_samples(s, kw, state)
    e_ref = float(np.median(guide_e))
    dev = np.abs(guide_e - e_ref)
    thresh = float(np.sort(dev)[len(dev) // 2 + 1]) * 0.999
    assert thresh > 0.0
    dt = 2.0 / thresh**2
    is_bad = dev > thresh
    assert 0 < int(is_bad.sum()) < len(is_bad), "need a split population"

    state = state._replace(e_estimate=jnp.asarray(e_ref))
    wp = w * trial_ov / guide_ov
    good = ~is_bad

    def run(mode):
        p = replace(params, n_prop_steps=0, dt=dt, mixed_outlier=mode)
        k = dict(kw, params=p, sr_fn=_identity_sr)
        _, obs = block_mixed(state, **k)
        return {key: np.asarray(v) for key, v in obs.scalars.items()}

    # clamp: the walker keeps its weight; its guide energy becomes e_ref and its three
    # component samples the wp-average over the non-outlier walkers
    out = run("clamp")
    e_clamped = np.where(is_bad, e_ref, guide_e)
    assert np.isclose(out["guide_weight"], w.sum())
    assert np.isclose(out["guide_energy"], (w * e_clamped).sum() / w.sum())
    comps_good = (wp[good, None] * comps[good]).sum(0) / wp[good].sum()
    comps_clamped = np.where(is_bad[:, None], comps_good[None, :], comps)
    assert np.isclose(out["trial_weight"], wp.sum())
    expect = (wp[:, None] * comps_clamped).sum(0) / wp.sum()
    assert np.allclose(out["trial_components"], expect, rtol=1e-10, atol=1e-12)
    # the clamped walkers' rows differ from the raw ones, so the clamp did something
    assert not np.allclose(expect, (wp[:, None] * comps).sum(0) / wp.sum())

    # reject: the walker's weight is zeroed everywhere
    out = run("reject")
    assert np.isclose(out["guide_weight"], w[good].sum())
    assert np.isclose(out["guide_energy"], (w[good] * guide_e[good]).sum() / w[good].sum())
    assert np.isclose(out["trial_weight"], wp[good].sum())
    assert np.allclose(out["trial_components"], comps_good, rtol=1e-10, atol=1e-12)

    # none: the raw samples, nothing flagged
    out = run("none")
    assert np.isclose(out["guide_weight"], w.sum())
    assert np.isclose(out["guide_energy"], (w * guide_e).sum() / w.sum())
    assert np.isclose(out["trial_weight"], wp.sum())
    assert np.allclose(
        out["trial_components"], (wp[:, None] * comps).sum(0) / wp.sum(), rtol=1e-10, atol=1e-12
    )

    # clamp with every walker an outlier: no average to clamp to, so the trial estimator
    # rejects them (zero trial weight) and the guide energy falls back to e_ref
    state_far = state._replace(e_estimate=jnp.asarray(e_ref + 1.0e3))
    p = replace(params, n_prop_steps=0, dt=dt, mixed_outlier="clamp")
    _, obs = block_mixed(state_far, **dict(kw, params=p, sr_fn=_identity_sr))
    assert np.isclose(np.asarray(obs.scalars["guide_weight"]), w.sum())
    assert np.isclose(np.asarray(obs.scalars["guide_energy"]), e_ref + 1.0e3)
    assert np.asarray(obs.scalars["trial_weight"]) == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
