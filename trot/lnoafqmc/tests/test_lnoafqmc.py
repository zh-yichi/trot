"""
Tests of trot.lnoafqmc on O2 (sto-3g, density fitted), two atomic fragments.

Run from the repository root:

    pytest trot/lnoafqmc/tests -q                 # fast checks
    pytest trot/lnoafqmc/tests -q --run-slow      # + the end-to-end LNO-AFQMC run

With a tight LNO threshold each fragment's local active space is the whole active space,
so the two fragments' projectors add up to the identity and their energies must add up
to full-space values: exactly at tau = 0 (the CCSD energy), within error bars after
sampling (the AFQMC/pt2CCSD energy). Everything below the propagator is deterministic
and checked to machine precision.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import trot.lnoafqmc  # noqa: F401  (allocator, before jax)
from trot import config

config.configure_once()

import io
import contextlib

import jax.numpy as jnp
from typing import Any

import numpy as np
import pytest
from pyscf import cc, gto, scf
from pyscf.data import elements

from trot.core.system import System
from trot.ham.chol import HamChol
from trot.lnoafqmc import LnoAfqmcMixed, LnoFragMixed, iao_fragment
from trot.lnoafqmc import integral as li
from trot.lnoafqmc import las, pipeline, solvers
from trot.lnoafqmc import staging as lst
from trot.lnoafqmc import stat_utils as su
from trot.lnoafqmc.meas import pt2ccsd as lm
from trot.lnoafqmc.mixed import available_mixed_recipes, get_mixed_recipe
from trot.lnoafqmc.trial.pt2ccsd import make_pt2ccsd_trial_data, overlap_r
from trot.meas.pt2ccsd import Pt2ccsdMeasCfg
from trot.meas.pt2ccsd import build_meas_ctx as trot_build_ctx
from trot.meas.pt2ccsd import energy_kernel_rw_rh_bar as trot_bar_kernel
from trot.stat_utils import pt2ccsd_blocking
from trot.trial.pt2ccsd import Pt2ccsdTrial as TrotPt2ccsdTrial

CHOL_CUT = 1e-6


@pytest.fixture(scope="module")
def o2():
    mol = gto.M(atom="O 0 0 0; O 0 0 1.208", basis="sto-3g", spin=0, verbose=0)
    mf: Any = scf.RHF(mol).density_fit()
    mf.kernel()
    nfrozen = int(elements.chemcore(mol))
    lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen, frag_type="atom")
    mlno = solvers.get_lnoccsd(mf, lo_coeff, frag_list, nfrozen, 1e-12)
    mlno.verbose = 0
    eris = mlno.ao2mo()
    nfrag = len(frag_list)
    frags = [
        pipeline.cpu_stage(
            mlno,
            mf,
            lo_coeff,
            frag_list[i],
            mlno.lno_thresh,
            [None, None],
            [[None, None]] * nfrag,
            ["1h", "1h"],
            eris,
            i,
            i,
            frag_name[i],
            True,
            True,
            nfrozen,
        )
        for i in range(nfrag)
    ]
    mycc: Any = cc.CCSD(mf, frozen=nfrozen)
    mycc.kernel()
    return dict(
        mol=mol,
        mf=mf,
        nfrozen=nfrozen,
        lo_coeff=lo_coeff,
        frag_list=frag_list,
        frag_name=frag_name,
        frags=frags,
        e_ccsd=float(mycc.e_corr),
    )


@pytest.fixture(scope="module")
def frag0(o2):
    """Fragment 0's hamiltonian, trial and meas ctx on the device."""
    mf, frag = o2["mf"], o2["frags"][0]
    with contextlib.redirect_stdout(io.StringIO()):
        ham = li.build_ham_lno_df(mf, frag.lno_coeff, frag.lno_frozen, chol_cut=CHOL_CUT)
    sys_ = System(norb=ham.norb, nelec=ham.nelec, walker_kind="restricted")
    ham_data = HamChol(
        jnp.asarray(ham.h0), jnp.asarray(ham.h1), jnp.asarray(ham.chol), basis="restricted"
    )
    tin = lst.stage_pt2ccsd_trial(frag)
    trial_data = make_pt2ccsd_trial_data(tin.data, sys_)
    ops = lm.make_pt2ccsd_meas_ops(sys_, measure_type="bar", nchol_chunk=3)
    ctx = ops.build_meas_ctx(ham_data, trial_data)
    t2_full = np.asarray(frag.t2).transpose(0, 2, 1, 3)  # unprojected, (i,a,j,b)
    return dict(
        ham=ham,
        sys=sys_,
        ham_data=ham_data,
        tin=tin,
        trial_data=trial_data,
        ops=ops,
        ctx=ctx,
        t2_full=t2_full,
    )


def _random_walkers(norb, nocc, n, seed=7):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield jnp.asarray(
            np.linalg.qr(rng.normal(size=(norb, nocc)) + 0.3j * rng.normal(size=(norb, nocc)))[0]
        )


# ----------------------------------------------------------------------------- CPU stage


def test_fragments_sum_to_canonical_mp2_ccsd(o2):
    frags = o2["frags"]
    assert o2["frag_name"] == ["O0", "O1"]
    assert all(f.nact == 8 for f in frags)
    e_mp = sum(f.efrag_mp for f in frags)
    e_cc = sum(f.efrag_cc for f in frags)
    from pyscf import mp

    mmp: Any = mp.MP2(o2["mf"], frozen=o2["nfrozen"])
    mmp.kernel()
    assert abs(e_mp - mmp.e_corr) < 1e-8
    assert abs(e_cc - o2["e_ccsd"]) < 1e-6


def test_check_span_passes_and_fails(o2):
    las.check_span(o2["mf"], o2["lo_coeff"], o2["nfrozen"])
    with pytest.raises(ValueError):
        las.check_span(o2["mf"], o2["lo_coeff"][:, :3], o2["nfrozen"])


# ----------------------------------------------------------------------------- integrals


def test_fragment_hamiltonian_reproduces_hf_energy(frag0, o2):
    """The full-space LAS with the DF-exact core: E_HF of the fragment ham is mf.e_tot."""
    ham = frag0["ham"]
    nocc = ham.nelec[0]
    lo = ham.chol[:, :nocc, :nocc]
    e_hf = (
        ham.h0
        + 2 * np.trace(ham.h1[:nocc, :nocc])
        + 2 * np.sum(np.trace(lo, axis1=1, axis2=2) ** 2)
    )
    e_hf -= np.einsum("gij,gji->", lo, lo)
    assert abs(e_hf - o2["mf"].e_tot) < 1e-6
    assert ham.norb == 8 and ham.nelec == (6, 6)


def test_las_ordering_is_asserted(o2):
    with pytest.raises(ValueError):
        li.get_las_idx(o2["mf"], np.array([0, 3]))


# ----------------------------------------------------------------------------- kernel


def test_kernel_prjlo_one_limit_matches_trot_bar_kernel(frag0):
    """With the identity projector the fragment kernel is trot's bar kernel; e0frg is e0 - E_HF."""
    sys_, ham_data, tin, ops = frag0["sys"], frag0["ham_data"], frag0["tin"], frag0["ops"]
    nocc, norb = sys_.nup, sys_.norb
    t2_full = frag0["t2_full"]
    d = dict(tin.data)
    d["prjlo"] = np.eye(nocc)
    d["t2"] = t2_full
    td = make_pt2ccsd_trial_data(d, sys_)
    ctx = ops.build_meas_ctx(ham_data, td)
    ttrial = TrotPt2ccsdTrial(mo_t=td.mo_t, t2=jnp.asarray(t2_full))
    tctx = trot_build_ctx(ham_data, ttrial, Pt2ccsdMeasCfg(measure_type="bar", nchol_chunk=3))
    ham = frag0["ham"]
    lo = ham.chol[:, :nocc, :nocc]
    e_hf = 2 * np.trace(ham.h1[:nocc, :nocc]) + 2 * np.sum(np.trace(lo, axis1=1, axis2=2) ** 2)
    e_hf -= np.einsum("gij,gji->", lo, lo)
    for w in _random_walkers(norb, nocc, 4):
        out = np.asarray(lm.energy_kernel_rw_rh_bar(w, ham_data, ctx, td))
        ref = np.asarray(trot_bar_kernel(w, ham_data, tctx, ttrial))  # [t2, e0, e1]
        assert abs(out[0] - ref[0]) < 1e-9
        assert abs(out[3] - ref[1]) < 1e-9
        assert abs(out[2] - ref[2]) < 1e-9
        assert abs(out[1] - (ref[1] - e_hf)) < 1e-9


def test_kernel_partition_of_unity(frag0, o2):
    """The fragment projectors of O0 and O1 add up to 1: so do the fragment kernels."""
    sys_, ham_data, tin, ops = frag0["sys"], frag0["ham_data"], frag0["tin"], frag0["ops"]
    mf, frag, lo_coeff, frag_list = o2["mf"], o2["frags"][0], o2["lo_coeff"], o2["frag_list"]
    nocc, norb = sys_.nup, sys_.norb
    nocc_full = int(np.count_nonzero(mf.mo_occ))
    ncore = int(np.sum(np.asarray(frag.lno_frozen) < nocc_full))  # frozen occupied count
    s1e = mf.get_ovlp()
    actocc = frag.lno_coeff[:, ncore : ncore + nocc]
    prj = []
    for f in range(len(frag_list)):
        u = actocc.T @ s1e @ lo_coeff[:, frag_list[f]]
        prj.append(u @ u.T)
    assert np.abs(sum(prj) - np.eye(nocc)).max() < 1e-10

    t2_full = frag0["t2_full"]
    tds, ctxs = [], []
    for p in prj + [np.eye(nocc)]:
        d = dict(tin.data)
        d["prjlo"] = p
        d["t2"] = np.einsum("iajb,ik->kajb", t2_full, p)
        td = make_pt2ccsd_trial_data(d, sys_)
        tds.append(td)
        ctxs.append(ops.build_meas_ctx(ham_data, td))
    for w in _random_walkers(norb, nocc, 4, seed=11):
        parts = [
            np.asarray(lm.energy_kernel_rw_rh_bar(w, ham_data, c, t)) for c, t in zip(ctxs, tds)
        ]
        full = parts[-1]
        s = parts[0] + parts[1]
        assert np.abs(s[:3] - full[:3]).max() < 1e-9  # t2frg, e0frg, e1frg are additive
        assert abs(parts[0][3] - full[3]) < 1e-9  # e0 is the same for every fragment


def test_overlap_ratio_is_trial_over_guide(frag0):
    """wp = w <T|phi>/<G|phi>: for the HF guide the ratio is afqmc's obar/o0."""
    sys_, td, ctx = frag0["sys"], frag0["trial_data"], frag0["ctx"]
    nocc, norb = sys_.nup, sys_.norb
    for w in _random_walkers(norb, nocc, 3, seed=3):
        obar = jnp.linalg.det((ctx.exp_t1 @ w)[:nocc, :]) ** 2
        o0 = jnp.linalg.det(w[:nocc, :]) ** 2
        assert abs(complex(overlap_r(w, td) / o0) - complex(obar / o0)) < 1e-10


def test_sto_chol_kernel_full_head_exact_parts_and_unbiased(frag0):
    """
    pt2ccsd_sto_chol: a full head is the bar kernel and draws no key; with a sampled tail
    t2frg, e0frg and e0 stay exact and e1frg is unbiased.
    """
    import jax

    sys_, ham_data, td, ctx = frag0["sys"], frag0["ham_data"], frag0["trial_data"], frag0["ctx"]
    full = lm.make_pt2ccsd_meas_ops(
        sys_, measure_type="sto_chol", nchol_chunk=3, n_chol_head="full"
    )
    sto = lm.make_pt2ccsd_meas_ops(
        sys_, measure_type="sto_chol", nchol_chunk=3, n_chol_head=8, n_chol_samples=8
    )
    assert not full.needs_rng("energy") and sto.needs_rng("energy")
    cf, cs = full.build_meas_ctx(ham_data, td), sto.build_meas_ctx(ham_data, td)
    w = next(_random_walkers(sys_.norb, sys_.nup, 1, seed=5))
    ref = np.asarray(lm.energy_kernel_rw_rh_bar(w, ham_data, ctx, td))
    assert np.abs(np.asarray(lm.energy_kernel_rw_rh_sto(w, ham_data, cf, td)) - ref).max() < 1e-10
    with pytest.raises(ValueError):
        lm.energy_kernel_rw_rh_sto(w, ham_data, cs, td)  # a sampled tail needs a key

    keys = jax.random.split(jax.random.PRNGKey(1), 2000)
    outs = np.asarray(jax.vmap(lambda k: lm.energy_kernel_rw_rh_sto(w, ham_data, cs, td, k))(keys))
    assert np.abs(outs[:, [0, 1, 3]] - ref[[0, 1, 3]]).max() < 1e-10
    e1_re, e1_im = np.real(outs[:, 2]), np.imag(outs[:, 2])
    assert e1_re.std() > 0  # it does sample
    assert abs(e1_re.mean() - np.real(ref[2])) < 5 * e1_re.std() / np.sqrt(len(keys))
    assert abs(e1_im.mean() - np.imag(ref[2])) < 5 * e1_im.std() / np.sqrt(len(keys))


# ----------------------------------------------------------------------------- statistics


def test_frag_blocking_mirrors_trot_when_e0frg_is_e0():
    rng = np.random.default_rng(3)
    n = 300
    w = jnp.asarray(rng.uniform(0.8, 1.2, n) + 0.05j * rng.normal(size=n))
    t2 = jnp.asarray(0.1 + 0.02 * rng.normal(size=n))
    e0 = jnp.asarray(-75.0 + 0.3 * rng.normal(size=n))
    e1 = jnp.asarray(-0.05 + 0.03 * rng.normal(size=n))
    for final in (False, True):
        e_t, err_t = pt2ccsd_blocking(0.0, w, t2, e0, e1, final=final)
        e_m, err_m = su.frag_pt2ccsd_blocking(w, t2, e0, e1, e0, final=final)
        assert abs(float(e_t) - float(e_m)) < 1e-12 and abs(float(err_t) - float(err_m)) < 1e-12
    assert su.frag_pt2ccsd_blocking(w[:1], t2[:1], e0[:1], e1[:1], e0[:1]) is None


def test_frag_delta_method_error_is_a_first_order_variance():
    """The delta-method error equals the jackknife-free linearization: check against a
    brute-force finite-difference influence function."""
    rng = np.random.default_rng(5)
    n = 50
    w = rng.uniform(0.5, 1.5, n)
    t2, e0f, e1f, e0 = (rng.normal(size=n) for _ in range(4))
    args = [jnp.asarray(x) for x in (w, t2, e0f, e1f, e0)]
    err = float(su._frag_pt2ccsd_delta_method_error(*args))

    def energy(ws):
        return float(su._frag_pt2ccsd_energy(jnp.asarray(ws), *args[1:]))

    # influence of sample i on E: derivative w.r.t. scaling sample i's contribution
    infl = np.zeros(n)
    eps = 1e-6
    for i in range(n):
        wp = w.copy()
        wp[i] *= 1 + eps
        wm = w.copy()
        wm[i] *= 1 - eps
        infl[i] = (energy(wp) - energy(wm)) / (2 * eps)
    ref = np.sqrt(np.sum(infl**2) * n / (n - 1))
    assert abs(err - ref) < 1e-6


def test_clean_frag_pt2ccsd_drops_outliers():
    n = 200
    rng = np.random.default_rng(1)
    ept = rng.normal(size=n) * 0.01
    ept[17] = 5.0
    arrs = [jnp.asarray(rng.normal(size=n)) for _ in range(5)]
    (w, *_), mask = su.clean_frag_pt2ccsd(jnp.asarray(ept), *arrs, zeta=20)
    assert int(mask.sum()) == n - 1 and not bool(mask[17]) and w.shape[0] == n - 1


# ----------------------------------------------------------------------------- recipes and files


def test_recipes():
    assert ("rhf", "pt2ccsd") in available_mixed_recipes()
    assert ("rhf", "pt2ccsd_sto_chol") in available_mixed_recipes()
    rec = get_mixed_recipe("pt2ccsd")
    assert rec.guide == "rhf" and rec.needs_amplitudes and rec.components == lm.TRIAL_COMPONENTS
    with pytest.raises(ValueError):
        get_mixed_recipe("pt2ccsd", guide="uhf")
    with pytest.raises(ValueError):
        get_mixed_recipe("cisd")


def test_frag_file_round_trip(o2, tmp_path):
    mf, frag = o2["mf"], o2["frags"][0]
    with contextlib.redirect_stdout(io.StringIO()):
        fm = LnoFragMixed(mf, frag, trial="pt2ccsd", chol_cut=CHOL_CUT)
        path = fm.save(tmp_path / "frag1.h5")
        frag2, staged, attrs = lst.load_frag(path)
    assert frag2.frag_name == frag.frag_name and frag2.nact == frag.nact
    assert np.allclose(frag2.t2, frag.t2) and np.allclose(frag2.uocc_loc, frag.uocc_loc)
    assert staged.ham.norb == 8 and staged.trial.kind == "rhf"
    assert abs(attrs["emf"] - mf.e_tot) < 1e-12
    # the guide staged by trot in the fragment basis is the identity on the active LNOs
    assert np.allclose(staged.trial.data["mo"], np.eye(8), atol=1e-10)
    with contextlib.redirect_stdout(io.StringIO()):
        fm2 = LnoFragMixed.from_frag_data(path, trial="pt2ccsd")
        job = fm2.build_job()
    assert job.sys.norb == 8 and job.mix_trial_data.nocc == 6


# ----------------------------------------------------------------------------- end to end


@pytest.mark.slow
def test_lno_afqmc_o2_end_to_end(o2, tmp_path, request):
    """
    Both O2 fragments: at tau = 0 the fragment energies add up to the (LNO-)CCSD
    energy exactly; after sampling they add up to the full-space AFQMC/pt2CCSD (bar)
    correlation energy within error bars; and a fragment re-run from its file with the
    same seed reproduces the loop.
    """
    if not request.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")
    from trot.afqmc import AfqmcMixed

    mf, nfrozen = o2["mf"], o2["nfrozen"]
    qmc: dict[str, Any] = dict(
        n_walkers=100, n_eql_blocks=10, n_blocks=60, dt=0.005, n_prop_steps=50
    )
    out = tmp_path
    with contextlib.redirect_stdout(io.StringIO()):
        lno = LnoAfqmcMixed(
            mf,
            o2["lo_coeff"],
            o2["frag_list"],
            frag_name=o2["frag_name"],
            lno_thresh=1e-12,
            nfrozen=nfrozen,
            trial="pt2ccsd",
            seed=17,
            chol_cut=CHOL_CUT,
            frag_output=str(out / "fragment.out"),
            lno_output=str(out / "lno_result.out"),
            save_frag_data=str(out / "frag_data"),
            keep_qmc_results=True,
            debug_memory=True,
            **qmc,
        )
        e_qmc, e_qmc_err = lno.kernel()
    init_sum = sum(r.frag_init_energy for r in lno.frag_qmc_results)
    assert abs(init_sum - lno.e_cc) < 1e-8
    assert abs(init_sum - o2["e_ccsd"]) < 1e-6
    assert (out / "fragment.out1").exists() and (out / "lno_result.out").exists()
    assert lno.lno_size == [8, 8] and lno.n_done == 2

    mycc = cc.CCSD(mf, frozen=nfrozen)
    mycc.kernel()
    with contextlib.redirect_stdout(io.StringIO()):
        af = AfqmcMixed(
            mycc, trial="pt2ccsd_bar", norb_frozen_core=nfrozen, seed=17, chol_cut=CHOL_CUT, **qmc
        )
        e_ref, err_ref = af.kernel()
    e_ref_corr = e_ref - mf.e_tot
    assert abs(e_qmc - e_ref_corr) < 3 * np.hypot(e_qmc_err, err_ref)

    with contextlib.redirect_stdout(io.StringIO()):
        fr = LnoFragMixed.from_frag_data(
            out / "frag_data" / "frag1.h5", seed=int(lno._seeds[0]), **qmc
        )
        e1, err1 = fr.kernel()
    assert abs(e1 - lno.lno_eqmc[0]) < 1e-10 and abs(err1 - lno.lno_eqmc_err[0]) < 1e-10
