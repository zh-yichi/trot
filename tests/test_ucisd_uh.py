"""
The UCISD guide on the unrestricted (uchol) hamiltonian: meas.ucisd_uh against afqmc's
wavefunctions_unrestricted.ucisd (the reference implementation), against the UHF kernels
when every CI coefficient is zero, and against the UCCSD energy on the reference
determinant.
"""

from __future__ import annotations

import contextlib
import io

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.afqmc import AfqmcMixed
from trot.meas.ucisd_uh import (
    build_meas_ctx_uh,
    energy_kernel_uw_uh,
    force_bias_kernel_uw_uh,
    overlap_uw_uh,
)
from trot.meas.uhf import build_meas_ctx_uh as uhf_build_meas_ctx_uh
from trot.meas.uhf import energy_kernel_uw_uh as uhf_energy_kernel_uw_uh
from trot.meas.uhf import force_bias_kernel_uw_uh as uhf_force_bias_kernel_uw_uh
from trot.meas.uhf import overlap_u as uhf_overlap_u
from trot.mixed import available_mixed_recipes, get_mixed_recipe
from trot.trial.ucisd import UcisdTrial
from trot.trial.uhf import UhfTrial

pytestmark = pytest.mark.slow

jax.config.update("jax_enable_x64", True)

_PARAMS = dict(dt=0.01, n_walkers=4, n_prop_steps=2, n_blocks=12, n_eql_blocks=2, seed=3)


@pytest.fixture(scope="module")
def oh():
    """OH radical, STO-3G, one frozen core: 5 active orbitals, (4, 3) active electrons."""
    from pyscf import cc, gto, scf

    mol = gto.M(atom="O 0 0 0; H 0 0 0.97", basis="sto-3g", spin=1, verbose=0)
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mycc = cc.UCCSD(mf, frozen=1)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    af = AfqmcMixed(mycc, guide="ucisd", trial="upt2ccsd", **_PARAMS)
    with contextlib.redirect_stdout(io.StringIO()):
        job = af.build_job()
    return {"mf": mf, "cc": mycc, "af": af, "job": job}


def _random_walkers(job, n, seed=0):
    rng = np.random.default_rng(seed)
    norb_a, norb_b = job.sys.norb
    nup, ndn = job.sys.nelec
    out = []
    for _ in range(n):
        wa = rng.normal(size=(norb_a, nup)) + 0.3j * rng.normal(size=(norb_a, nup))
        wb = rng.normal(size=(norb_b, ndn)) + 0.3j * rng.normal(size=(norb_b, ndn))
        # keep the reference block well conditioned, as propagated walkers are
        wa[:nup] += 2.0 * np.eye(nup)
        wb[:ndn] += 2.0 * np.eye(ndn)
        out.append((jnp.asarray(wa), jnp.asarray(wb)))
    return out


def test_pair_is_registered():
    assert ("ucisd", "upt2ccsd") in available_mixed_recipes()
    rec = get_mixed_recipe("upt2ccsd", guide="ucisd")
    assert rec.walker_kind == "unrestricted" and rec.ham_basis == "uchol"


def test_guide_bundle_is_ucisd(oh):
    job = oh["job"]
    assert isinstance(job.trial_data, UcisdTrial)
    assert job.meas_ops.overlap is overlap_uw_uh
    assert job.meas_ops.kernels["force_bias"] is force_bias_kernel_uw_uh
    assert job.meas_ops.kernels["energy"] is energy_kernel_uw_uh
    # the reference of each spin is the identity in its own basis
    nup, ndn = job.sys.nelec
    np.testing.assert_array_equal(
        np.asarray(job.trial_data.mo_coeff_a), np.eye(job.sys.norb[0])[:, :nup]
    )
    np.testing.assert_array_equal(
        np.asarray(job.trial_data.mo_coeff_b), np.eye(job.sys.norb[1])[:, :ndn]
    )


def test_matches_afqmc_reference_implementation(oh):
    ref_mod = pytest.importorskip(
        "afqmc.wavefunctions.wavefunctions_unrestricted",
        reason="the reference afqmc package is not on the path",
    )
    job = oh["job"]
    ham, td = job.ham_data, job.trial_data
    ctx = build_meas_ctx_uh(ham, td)
    norb = int(job.sys.norb[0])
    assert job.sys.norb[0] == job.sys.norb[1]  # the reference class has one norb
    nelec = tuple(int(n) for n in job.sys.nelec)

    ref = ref_mod.ucisd(norb, nelec)
    ham_dict = {
        "h0": ham.h0,
        "h1": [ham.h1_a, ham.h1_b],
        "chol": [
            ham.chol_a.reshape(ham.chol_a.shape[0], -1),
            ham.chol_b.reshape(ham.chol_b.shape[0], -1),
        ],
    }
    wave_data = {
        "ci1A": td.c1a,
        "ci1B": td.c1b,
        "ci2AA": td.c2aa,
        "ci2AB": td.c2ab,
        "ci2BB": td.c2bb,
        "mo_coeff": [td.mo_coeff_a, td.mo_coeff_b],
    }
    ham_dict = ref._build_measurement_intermediates(ham_dict, wave_data)

    for wa, wb in _random_walkers(job, 4):
        o_ref = complex(ref._calc_overlap(wa, wb, wave_data))
        o_new = complex(overlap_uw_uh((wa, wb), td))
        assert o_new == pytest.approx(o_ref, rel=1e-10)
        fb_ref = np.asarray(ref._calc_force_bias(wa, wb, ham_dict, wave_data))
        fb_new = np.asarray(force_bias_kernel_uw_uh((wa, wb), ham, ctx, td))
        np.testing.assert_allclose(fb_new, fb_ref, rtol=1e-10, atol=1e-12)
        e_ref = complex(ref._calc_energy(wa, wb, ham_dict, wave_data))
        e_new = complex(energy_kernel_uw_uh((wa, wb), ham, ctx, td))
        assert e_new == pytest.approx(e_ref, rel=1e-10)


def test_zero_coefficients_reduce_to_uhf(oh):
    job = oh["job"]
    ham, td = job.ham_data, job.trial_data
    td0 = UcisdTrial(
        mo_coeff_a=td.mo_coeff_a,
        mo_coeff_b=td.mo_coeff_b,
        c1a=jnp.zeros_like(td.c1a),
        c1b=jnp.zeros_like(td.c1b),
        c2aa=jnp.zeros_like(td.c2aa),
        c2ab=jnp.zeros_like(td.c2ab),
        c2bb=jnp.zeros_like(td.c2bb),
    )
    ctx0 = build_meas_ctx_uh(ham, td0)
    uhf = UhfTrial(td.mo_coeff_a, td.mo_coeff_b)
    uctx = uhf_build_meas_ctx_uh(ham, uhf)
    for w in _random_walkers(job, 3, seed=1):
        assert complex(overlap_uw_uh(w, td0)) == pytest.approx(
            complex(uhf_overlap_u(w, uhf)), rel=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(force_bias_kernel_uw_uh(w, ham, ctx0, td0)),
            np.asarray(uhf_force_bias_kernel_uw_uh(w, ham, uctx, uhf)),
            rtol=1e-11,
            atol=1e-13,
        )
        assert complex(energy_kernel_uw_uh(w, ham, ctx0, td0)) == pytest.approx(
            complex(uhf_energy_kernel_uw_uh(w, ham, uctx, uhf)), rel=1e-11
        )


def test_energy_on_the_reference_determinant_is_uccsd(oh):
    """<CISD|H|HF>/<CISD|HF> with the CC-derived coefficients is the UCCSD energy."""
    job = oh["job"]
    ham, td = job.ham_data, job.trial_data
    ctx = build_meas_ctx_uh(ham, td)
    w = (td.mo_coeff_a + 0j, td.mo_coeff_b + 0j)
    e = complex(energy_kernel_uw_uh(w, ham, ctx, td))
    assert complex(overlap_uw_uh(w, td)) == pytest.approx(1.0, abs=1e-12)
    assert e.real == pytest.approx(oh["cc"].e_tot, abs=1e-5)  # cholesky truncation


def test_ucisd_guide_upt2ccsd_trial_runs(oh):
    af = oh["af"]
    with contextlib.redirect_stdout(io.StringIO()):
        e, err = af.kernel()
    assert np.isfinite(e) and np.isfinite(err) and np.isfinite(af.guide_e_tot)
    assert set(af.qmc_result.trial_block_components) == {"t2", "e0", "e1"}
