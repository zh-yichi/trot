"""
Tests of the unrestricted half of trot.lnoafqmc (trial="upt2ccsd"), on triplet O2 (UHF,
sto-3g, density fitted, two atomic fragments, full-space LAS). The restricted half and
the statistics are in test_lnoafqmc.py.

    pytest trot/lnoafqmc/tests -q [--run-slow]
"""

from __future__ import annotations

import trot.lnoafqmc  # noqa: F401  (allocator, before jax)
from trot import config

config.configure_once()

import contextlib
import io

import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, mp, scf
from pyscf.data import elements

from trot.core.system import System
from trot.ham.chol_u import HamCholU
from trot.lnoafqmc import LnoAfqmcMixed, LnoFragMixed, iao_fragment
from trot.lnoafqmc import integral as li
from trot.lnoafqmc import pipeline, solvers
from trot.lnoafqmc import staging as lst
from trot.lnoafqmc.meas import upt2ccsd as lmu
from trot.lnoafqmc.mixed import available_mixed_recipes, get_mixed_recipe
from trot.lnoafqmc.trial.upt2ccsd import make_upt2ccsd_trial_data, overlap_u
from trot.meas.pt2ccsd import Pt2ccsdMeasCfg
from trot.meas.upt2ccsd import build_meas_ctx as trot_build_ctx
from trot.meas.upt2ccsd import energy_kernel_uw_uh_bar as trot_bar_kernel
from trot.trial.upt2ccsd import Upt2ccsdTrial as TrotUpt2ccsdTrial

CHOL_CUT = 1e-6


@pytest.fixture(scope="module")
def o2t():
    mol = gto.M(atom="O 0 0 0; O 0 0 1.208", basis="sto-3g", spin=2, verbose=0)
    mf = scf.UHF(mol).density_fit()
    mf.kernel()
    nfrozen = int(elements.chemcore(mol))
    lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen, frag_type="atom")
    mlno = solvers.get_lnoccsd(mf, lo_coeff, frag_list, nfrozen, 1e-12)
    mlno.verbose = 0
    eris = mlno.ao2mo()
    nfrag = len(frag_list)
    frags = [
        pipeline.cpu_stage(
            mlno, mf, lo_coeff, frag_list[i], mlno.lno_thresh, [None, None], [[None, None]] * nfrag,
            ["1h", "1h"], eris, i, i, frag_name[i], True, True, nfrozen,
        )
        for i in range(nfrag)
    ]
    mycc = cc.UCCSD(mf, frozen=nfrozen)
    mycc.kernel()
    return dict(
        mol=mol, mf=mf, nfrozen=nfrozen, lo_coeff=lo_coeff, frag_list=frag_list,
        frag_name=frag_name, frags=frags, e_ccsd=float(mycc.e_corr),
    )


@pytest.fixture(scope="module")
def ufrag0(o2t):
    """Fragment 0's uchol hamiltonian, trial and meas ops on the device."""
    mf, frag = o2t["mf"], o2t["frags"][0]
    with contextlib.redirect_stdout(io.StringIO()):
        ham = li.build_ham_ulno_df(mf, frag.lno_coeff, frag.lno_frozen, chol_cut=CHOL_CUT)
    sys_ = System(norb=ham.norb[0], nelec=ham.nelec, walker_kind="unrestricted")
    ham_data = HamCholU(
        jnp.asarray(ham.h0), jnp.asarray(ham.h1_a), jnp.asarray(ham.h1_b),
        jnp.asarray(ham.chol_a), jnp.asarray(ham.chol_b), basis="uchol",
    )
    tin = lst.stage_upt2ccsd_trial(frag)
    ops = lmu.make_upt2ccsd_meas_ops(sys_, measure_type="bar", nchol_chunk=3)
    # unprojected amplitudes in trot's (i,a,j,b) layout
    t2_full = tuple(np.asarray(t).transpose(0, 2, 1, 3) for t in frag.t2)
    return dict(ham=ham, sys=sys_, ham_data=ham_data, tin=tin, ops=ops, t2_full=t2_full)


def _e_hf_el(ham) -> float:
    na, nb = ham.nelec
    loa, lob = ham.chol_a[:, :na, :na], ham.chol_b[:, :nb, :nb]
    tr = np.trace(loa, axis1=1, axis2=2) + np.trace(lob, axis1=1, axis2=2)
    return float(
        np.trace(ham.h1_a[:na, :na]) + np.trace(ham.h1_b[:nb, :nb]) + 0.5 * np.sum(tr**2)
        - 0.5 * np.einsum("gij,gji->", loa, loa) - 0.5 * np.einsum("gij,gji->", lob, lob)
    )


def _random_walkers(norb, nelec, n, seed=7):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        yield tuple(
            jnp.asarray(np.linalg.qr(rng.normal(size=(no, ne)) + 0.3j * rng.normal(size=(no, ne)))[0])
            for no, ne in zip(norb, nelec)
        )


def _projected_trial(ufrag0, prj_a, prj_b):
    t2aa, t2ab, t2bb = ufrag0["t2_full"]
    d = dict(ufrag0["tin"].data)
    d.update(
        prjlo_a=prj_a,
        prjlo_b=prj_b,
        t2aa=np.einsum("iajb,ik->kajb", t2aa, prj_a),
        t2ab=np.einsum("iajb,ik->kajb", t2ab, prj_a),
        t2ba=np.einsum("jbia,ik->kajb", t2ab, prj_b),
        t2bb=np.einsum("iajb,ik->kajb", t2bb, prj_b),
    )
    td = make_upt2ccsd_trial_data(d, ufrag0["sys"])
    return td, ufrag0["ops"].build_meas_ctx(ufrag0["ham_data"], td)


# ----------------------------------------------------------------------------- CPU stage, integrals


def test_u_fragments_sum_to_canonical_mp2_ccsd(o2t):
    frags = o2t["frags"]
    assert all(f.unrestricted and f.nact == (8, 8) and tuple(f.nactocc) == (7, 5) for f in frags)
    mmp = mp.UMP2(o2t["mf"], frozen=o2t["nfrozen"])
    mmp.kernel()
    assert abs(sum(f.efrag_mp for f in frags) - mmp.e_corr) < 1e-8
    assert abs(sum(f.efrag_cc for f in frags) - o2t["e_ccsd"]) < 1e-6


def test_u_fragment_hamiltonian_reproduces_hf_energy(ufrag0, o2t):
    """Full-space LAS, DF-exact core, joint cholesky: E_UHF of the fragment ham is mf.e_tot."""
    ham = ufrag0["ham"]
    assert ham.norb == (8, 8) and ham.nelec == (7, 5) and ham.basis == "uchol"
    assert ham.chol_a.shape[0] == ham.chol_b.shape[0]  # one shared cholesky index
    assert abs(ham.h0 + _e_hf_el(ham) - o2t["mf"].e_tot) < 1e-5


# ----------------------------------------------------------------------------- kernel


def test_u_kernel_prjlo_one_limit_matches_trot_bar_kernel(ufrag0):
    """With identity projectors the fragment kernel is trot's upt2ccsd_bar kernel; e0frg is e0 - E_HF."""
    ham, ham_data = ufrag0["ham"], ufrag0["ham_data"]
    na, nb = ham.nelec
    td, ctx = _projected_trial(ufrag0, np.eye(na), np.eye(nb))
    t2aa, t2ab, t2bb = ufrag0["t2_full"]
    ttrial = TrotUpt2ccsdTrial(
        mo_t_a=td.mo_t_a, mo_t_b=td.mo_t_b, t2aa=jnp.asarray(t2aa), t2ab=jnp.asarray(t2ab), t2bb=jnp.asarray(t2bb)
    )
    tctx = trot_build_ctx(ham_data, ttrial, Pt2ccsdMeasCfg(measure_type="bar", nchol_chunk=3))
    e_hf = _e_hf_el(ham)
    for w in _random_walkers(ham.norb, ham.nelec, 4):
        out = np.asarray(lmu.energy_kernel_uw_uh_bar(w, ham_data, ctx, td))
        ref = np.asarray(trot_bar_kernel(w, ham_data, tctx, ttrial))  # [t2, e0, e1]
        assert abs(out[0] - ref[0]) < 1e-9
        assert abs(out[3] - ref[1]) < 1e-9
        assert abs(out[2] - ref[2]) < 1e-9
        assert abs(out[1] - (ref[1] - e_hf)) < 1e-9


def test_u_kernel_partition_of_unity(ufrag0, o2t):
    """The fragment projectors of O0 and O1 add up to 1 per spin: so do the fragment kernels."""
    mf, frag, lo_coeff, frag_list = o2t["mf"], o2t["frags"][0], o2t["lo_coeff"], o2t["frag_list"]
    ham, ham_data = ufrag0["ham"], ufrag0["ham_data"]
    ncore, nocc, _, _ = li.get_las_idx(mf, frag.lno_frozen)
    s1e = mf.get_ovlp()
    prj = []
    for f in range(len(frag_list)):
        pp = []
        for s in range(2):
            actocc = np.asarray(frag.lno_coeff[s])[:, ncore[s] : ncore[s] + nocc[s]]
            u = actocc.T @ s1e @ np.asarray(lo_coeff[s])[:, frag_list[f][s]]
            pp.append(u @ u.T)
        prj.append(pp)
    for s in range(2):
        assert np.abs(sum(p[s] for p in prj) - np.eye(nocc[s])).max() < 1e-10
    # the staged projector is fragment 0's
    assert np.abs(ufrag0["tin"].data["prjlo_a"] - prj[0][0]).max() < 1e-10
    assert np.abs(ufrag0["tin"].data["prjlo_b"] - prj[0][1]).max() < 1e-10

    parts = [_projected_trial(ufrag0, *p) for p in prj]
    td1, ctx1 = _projected_trial(ufrag0, np.eye(nocc[0]), np.eye(nocc[1]))
    for w in _random_walkers(ham.norb, ham.nelec, 4, seed=11):
        outs = [np.asarray(lmu.energy_kernel_uw_uh_bar(w, ham_data, c, t)) for t, c in parts]
        full = np.asarray(lmu.energy_kernel_uw_uh_bar(w, ham_data, ctx1, td1))
        s = outs[0] + outs[1]
        assert np.abs(s[:3] - full[:3]).max() < 1e-9  # t2frg, e0frg, e1frg are additive
        assert abs(outs[0][3] - full[3]) < 1e-9  # e0 is the same for every fragment


def test_u_overlap_is_exp_t1_reference(ufrag0):
    """<T|phi> = <HF|exp(T1) phi> per spin: with the UHF guide, wp/w is afqmc's obar/o0."""
    ham, sys_ = ufrag0["ham"], ufrag0["sys"]
    na, nb = ham.nelec
    td = make_upt2ccsd_trial_data(ufrag0["tin"].data, sys_)
    ctx = ufrag0["ops"].build_meas_ctx(ufrag0["ham_data"], td)
    for w in _random_walkers(ham.norb, ham.nelec, 3, seed=3):
        obar = jnp.linalg.det((ctx.exp_t1_a @ w[0])[:na, :]) * jnp.linalg.det((ctx.exp_t1_b @ w[1])[:nb, :])
        assert abs(complex(overlap_u(w, td)) - complex(obar)) < 1e-10


def test_u_sto_chol_kernel_full_head_exact_parts_and_unbiased(ufrag0):
    """
    upt2ccsd_sto_chol: a full head is the bar kernel and draws no key; with a sampled tail
    t2frg, e0frg and e0 stay exact and e1frg is unbiased.
    """
    import jax

    ham, sys_, ham_data = ufrag0["ham"], ufrag0["sys"], ufrag0["ham_data"]
    td = make_upt2ccsd_trial_data(ufrag0["tin"].data, sys_)
    ctx = ufrag0["ops"].build_meas_ctx(ham_data, td)
    full = lmu.make_upt2ccsd_meas_ops(sys_, measure_type="sto_chol", nchol_chunk=3, n_chol_head="full")
    sto = lmu.make_upt2ccsd_meas_ops(sys_, measure_type="sto_chol", nchol_chunk=3, n_chol_head=8, n_chol_samples=8)
    assert not full.needs_rng("energy") and sto.needs_rng("energy")
    cf, cs = full.build_meas_ctx(ham_data, td), sto.build_meas_ctx(ham_data, td)
    w = next(_random_walkers(ham.norb, ham.nelec, 1, seed=5))
    ref = np.asarray(lmu.energy_kernel_uw_uh_bar(w, ham_data, ctx, td))
    assert np.abs(np.asarray(lmu.energy_kernel_uw_uh_sto(w, ham_data, cf, td)) - ref).max() < 1e-10
    with pytest.raises(ValueError):
        lmu.energy_kernel_uw_uh_sto(w, ham_data, cs, td)  # a sampled tail needs a key

    keys = jax.random.split(jax.random.PRNGKey(1), 2000)
    outs = np.asarray(jax.vmap(lambda k: lmu.energy_kernel_uw_uh_sto(w, ham_data, cs, td, k))(keys))
    assert np.abs(outs[:, [0, 1, 3]] - ref[[0, 1, 3]]).max() < 1e-10
    e1 = outs[:, 2]
    assert e1.real.std() > 0  # it does sample
    assert abs(e1.real.mean() - ref[2].real) < 5 * e1.real.std() / np.sqrt(len(keys))
    assert abs(e1.imag.mean() - ref[2].imag) < 5 * e1.imag.std() / np.sqrt(len(keys))


def test_u_sto_chol_full_head_run_reproduces_bar_run(o2t, tmp_path):
    """From one fragment file and seed, upt2ccsd_sto_chol with a full head is the upt2ccsd run."""
    qmc = dict(n_walkers=20, n_eql_blocks=2, n_blocks=20, dt=0.005, n_prop_steps=10, seed=3)
    with contextlib.redirect_stdout(io.StringIO()):
        path = LnoFragMixed(o2t["mf"], o2t["frags"][0], chol_cut=CHOL_CUT).save(tmp_path / "frag1.h5")
        e_bar = LnoFragMixed.from_frag_data(path, trial="upt2ccsd", **qmc).kernel()
        e_full = LnoFragMixed.from_frag_data(
            path, trial="upt2ccsd_sto_chol", trial_kwargs={"n_chol_head": "full"}, **qmc
        ).kernel()
        e_sto = LnoFragMixed.from_frag_data(
            path, trial="upt2ccsd_sto_chol", trial_kwargs={"chol_cost_ratio": 0.5}, **qmc
        ).kernel()
    assert abs(e_full[0] - e_bar[0]) < 1e-12 and abs(e_full[1] - e_bar[1]) < 1e-12
    assert np.isfinite(e_sto[0]) and e_sto[0] != e_bar[0]


# ----------------------------------------------------------------------------- recipes and files


def test_u_recipes():
    assert ("uhf", "upt2ccsd") in available_mixed_recipes()
    assert ("uhf", "upt2ccsd_sto_chol") in available_mixed_recipes()
    rec = get_mixed_recipe("upt2ccsd")
    assert rec.guide == "uhf" and rec.ham_basis == "uchol" and rec.walker_kind == "unrestricted"
    assert rec.needs_amplitudes and rec.components == lmu.TRIAL_COMPONENTS
    with pytest.raises(ValueError):
        get_mixed_recipe("upt2ccsd", guide="rhf")


def test_u_trial_must_match_mean_field(o2t):
    with pytest.raises(ValueError):
        LnoAfqmcMixed(o2t["mf"], o2t["lo_coeff"], o2t["frag_list"], nfrozen=o2t["nfrozen"], trial="pt2ccsd")


def test_u_frag_file_round_trip(o2t, tmp_path):
    mf, frag = o2t["mf"], o2t["frags"][0]
    with contextlib.redirect_stdout(io.StringIO()):
        fm = LnoFragMixed(mf, frag, chol_cut=CHOL_CUT)  # trial defaults to upt2ccsd
        path = fm.save(tmp_path / "frag1.h5")
        ham = fm.stage().ham
        frag2, staged, attrs = lst.load_frag(path)
    assert fm.trial == "upt2ccsd" and fm.guide == "uhf"
    assert frag2.unrestricted and frag2.nact == frag.nact and frag2.frag_name == frag.frag_name
    for x, y in zip(frag2.t2, frag.t2):
        assert np.allclose(x, y)
    assert staged.ham.basis == "uchol" and staged.ham.norb == (8, 8) and staged.trial.kind == "uhf"
    assert np.array_equal(staged.ham.chol_a, ham.chol_a) and np.array_equal(staged.ham.h1_b, ham.h1_b)
    assert abs(attrs["emf"] - mf.e_tot) < 1e-12
    with contextlib.redirect_stdout(io.StringIO()):
        fm2 = LnoFragMixed.from_frag_data(path)
        job = fm2.build_job()
    assert fm2.trial == "upt2ccsd" and job.mix_trial_data.nocc == (7, 5)


# ----------------------------------------------------------------------------- end to end


@pytest.mark.slow
def test_ulno_afqmc_o2_triplet_end_to_end(o2t, tmp_path, request):
    """
    Both triplet O2 fragments: at tau = 0 the fragment energies add up to the (LNO-)UCCSD
    energy exactly; after sampling they add up to the full-space AFQMC/upt2CCSD (bar)
    correlation energy within error bars; and a fragment re-run from its file with the
    same seed reproduces the loop.
    """
    if not request.config.getoption("--run-slow"):
        pytest.skip("need --run-slow option to run")
    from trot.afqmc import AfqmcMixed

    mf, nfrozen = o2t["mf"], o2t["nfrozen"]
    qmc = dict(n_walkers=100, n_eql_blocks=10, n_blocks=60, dt=0.005, n_prop_steps=50)
    out = tmp_path
    with contextlib.redirect_stdout(io.StringIO()):
        lno = LnoAfqmcMixed(
            mf, o2t["lo_coeff"], o2t["frag_list"], frag_name=o2t["frag_name"], lno_thresh=1e-12,
            nfrozen=nfrozen, seed=17, chol_cut=CHOL_CUT,
            frag_output=str(out / "fragment.out"), lno_output=str(out / "lno_result.out"),
            save_frag_data=str(out / "frag_data"), keep_qmc_results=True, debug_memory=True, **qmc,
        )
        e_qmc, e_qmc_err = lno.kernel()
    assert lno.trial == "upt2ccsd" and lno.guide == "uhf"
    init_sum = sum(r.frag_init_energy for r in lno.frag_qmc_results)
    assert abs(init_sum - lno.e_cc) < 1e-8
    assert abs(init_sum - o2t["e_ccsd"]) < 1e-6
    assert (out / "fragment.out1").exists() and (out / "lno_result.out").exists()
    assert lno.lno_size == [(8, 8), (8, 8)] and lno.n_done == 2

    mycc = cc.UCCSD(mf, frozen=nfrozen)
    mycc.kernel()
    with contextlib.redirect_stdout(io.StringIO()):
        af = AfqmcMixed(mycc, trial="upt2ccsd_bar", norb_frozen_core=nfrozen, seed=17, chol_cut=CHOL_CUT, **qmc)
        e_ref, err_ref = af.kernel()
    assert abs(e_qmc - (e_ref - mf.e_tot)) < 3 * np.hypot(e_qmc_err, err_ref)

    with contextlib.redirect_stdout(io.StringIO()):
        fr = LnoFragMixed.from_frag_data(out / "frag_data" / "frag1.h5", seed=int(lno._seeds[0]), **qmc)
        e1, err1 = fr.kernel()
    assert abs(e1 - lno.lno_eqmc[0]) < 1e-10 and abs(err1 - lno.lno_eqmc_err[0]) < 1e-10
