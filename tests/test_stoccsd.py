"""
The stochastically sampled CCSD guide (trial/stoccsd.py, meas/stoccsd.py,
staging_stoccsd.py): the T2 factorization, the sampler, the multi-determinant kernels
against the per-determinant RHF kernels, the sampled overlap against the exact CCSD
state built in Fock space, and the stager on a pyscf CCSD object.
"""

from trot import config

config.configure_once()

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, scf
from pyscf.fci import addons, cistring

from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.meas import rhf as meas_rhf
from trot.meas.stoccsd import (
    StoccsdMeasCtx,
    build_meas_ctx,
    energy_kernel_rw_rh,
    force_bias_kernel_rw_rh,
    get_stoccsd_meas_cfg,
    make_stoccsd_meas_ops,
)
from trot.staging_stoccsd import STOCCSD_KIND, stage_stoccsd_guide
from trot.testing import make_random_ham_chol, rand_orthonormal_cols
from trot.trial.cisd import CisdTrial
from trot.trial.cisd import overlap_r as cisd_overlap_r
from trot.trial.pt2ccsd import Pt2ccsdTrial
from trot.trial.pt2ccsd import overlap_r as pt2_overlap_r
from trot.trial.rhf import RhfTrial
from trot.trial.stoccsd import (
    StoccsdGuide,
    decompose_t2,
    det_overlaps_r,
    get_rdm1,
    make_stoccsd_trial_data,
    make_stoccsd_trial_ops,
    overlap_r,
    resample,
    sample_dets,
    thouless_r,
)

jax.config.update("jax_enable_x64", True)

_NORB, _NOCC, _NCHOL = 7, 3, 9


def _random_t2(key, nocc, nvir, scale=0.05):
    t2 = scale * jax.random.normal(key, (nocc, nvir, nocc, nvir))
    return 0.5 * (t2 + t2.transpose(2, 3, 0, 1))  # t_iajb = t_jbia


def _thouless_np(nocc, norb, t1):
    exp_t1 = np.eye(norb)
    exp_t1[:nocc, nocc:] = t1
    return exp_t1.T @ np.eye(norb)[:, :nocc]


@pytest.fixture(scope="module")
def random_case():
    key = jax.random.PRNGKey(11)
    k_ham, k_t1, k_t2, k_w, k_s = jax.random.split(key, 5)
    ham = make_random_ham_chol(k_ham, norb=_NORB, n_chol=_NCHOL)
    nvir = _NORB - _NOCC
    t1 = 0.05 * np.asarray(jax.random.normal(k_t1, (_NOCC, nvir)))
    t2 = np.asarray(_random_t2(k_t2, _NOCC, nvir))
    mo_t = _thouless_np(_NOCC, _NORB, t1)
    tau = decompose_t2(t2, thresh=1e-12)
    sys = System(norb=_NORB, nelec=(_NOCC, _NOCC), walker_kind="restricted")
    guide = make_stoccsd_trial_data({"mo_t": mo_t, "tau": tau}, sys, n_samples=6, seed=3)
    walker = rand_orthonormal_cols(k_w, _NORB, _NOCC)
    # a walker near the reference, so that the sampled overlaps are not all tiny
    walker = walker + 0.5 * jnp.eye(_NORB, _NOCC)
    walker, _ = jnp.linalg.qr(walker)
    return dict(ham=ham, sys=sys, guide=guide, walker=walker, t1=t1, t2=t2, mo_t=mo_t, tau=tau)


# ---------------------------------------------------------------------------
# factorization and sampler
# ---------------------------------------------------------------------------


def test_decompose_t2_reconstructs_t2():
    nocc, nvir = 3, 4
    t2 = np.asarray(_random_t2(jax.random.PRNGKey(0), nocc, nvir, scale=0.1))
    tau = decompose_t2(t2, thresh=1e-12)
    assert tau.shape == (nocc * nvir, nocc, nvir) and np.iscomplexobj(tau)
    rebuilt = np.einsum("gia,gjb->iajb", tau, tau)
    assert np.abs(rebuilt.imag).max() < 1e-12
    np.testing.assert_allclose(rebuilt.real, t2, atol=1e-12)
    # negative eigenvalues are where tau is imaginary, so a threshold drops modes
    thresh = 0.05
    tau_cut = decompose_t2(t2, thresh=thresh)
    assert tau_cut.shape[0] < tau.shape[0]
    err = np.abs(np.einsum("gia,gjb->iajb", tau_cut, tau_cut).real - t2).max()
    assert err <= thresh
    with pytest.raises(ValueError, match="symmetric"):
        decompose_t2(np.asarray(jax.random.normal(jax.random.PRNGKey(1), (nocc, nvir, nocc, nvir))))


def test_sample_dets_shapes_and_no_modes(random_case):
    guide = random_case["guide"]
    key = jax.random.PRNGKey(5)
    dets = sample_dets(guide.mo_t, guide.tau, key, 4)
    assert dets.shape == (4, _NORB, _NOCC) and dets.dtype == jnp.complex128
    # the occupied rows are the reference's, the virtual rows are shifted by (y.tau)^T
    np.testing.assert_allclose(
        dets[:, :_NOCC], np.broadcast_to(guide.mo_t[:_NOCC], (4, _NOCC, _NOCC))
    )
    # different keys, different samples; the same key, the same sample
    assert not np.allclose(dets, sample_dets(guide.mo_t, guide.tau, jax.random.PRNGKey(6), 4))
    np.testing.assert_array_equal(dets, sample_dets(guide.mo_t, guide.tau, key, 4))
    # no modes: every sample is the Thouless reference
    tau0 = jnp.zeros((0, _NOCC, _NORB - _NOCC), dtype=jnp.complex128)
    dets0 = sample_dets(guide.mo_t, tau0, key, 3)
    np.testing.assert_allclose(dets0, np.broadcast_to(guide.mo_t, (3, _NORB, _NOCC)))
    # resample keeps the size and replaces the sample
    g2 = resample(guide, jax.random.PRNGKey(9))
    assert g2.n_samples == guide.n_samples and not np.allclose(g2.dets, guide.dets)


def test_thouless_matches_matrix_form(random_case):
    mo_t = jnp.asarray(random_case["mo_t"], dtype=jnp.complex128)
    t = 0.1 * (jax.random.normal(jax.random.PRNGKey(2), (_NOCC, _NORB - _NOCC)) + 0.3j)
    exp_t = jnp.eye(_NORB, dtype=jnp.complex128).at[:_NOCC, _NOCC:].set(t)
    np.testing.assert_allclose(thouless_r(mo_t, t), exp_t.T @ mo_t, atol=1e-14)


def test_trial_data_and_ops(random_case):
    sys, guide = random_case["sys"], random_case["guide"]
    assert isinstance(guide, StoccsdGuide)
    assert (guide.norb, guide.nocc, guide.nvir) == (_NORB, _NOCC, _NORB - _NOCC)
    assert guide.n_modes == _NOCC * (_NORB - _NOCC) and guide.n_samples == 6
    ops = make_stoccsd_trial_ops(sys)
    assert ops.overlap is overlap_r and ops.get_rdm1 is get_rdm1
    dm = get_rdm1(guide)
    assert dm.shape == (2, _NORB, _NORB)
    np.testing.assert_allclose(jnp.trace(dm[0]), _NOCC, atol=1e-12)
    np.testing.assert_allclose(dm[0] @ dm[0], dm[0], atol=1e-12)  # a projector
    np.testing.assert_array_equal(dm[0], np.diag(np.arange(_NORB) < _NOCC))  # the HF density
    # the pytree round trip keeps the arrays
    leaves, treedef = jax.tree_util.tree_flatten(guide)
    assert len(leaves) == 3
    g2 = jax.tree_util.tree_unflatten(treedef, leaves)
    np.testing.assert_array_equal(g2.dets, guide.dets)
    with pytest.raises(ValueError, match="n_samples"):
        make_stoccsd_trial_data(
            {"mo_t": random_case["mo_t"], "tau": random_case["tau"]}, sys, n_samples=0
        )
    with pytest.raises(ValueError, match="walker_kind"):
        make_stoccsd_trial_ops(System(norb=_NORB, nelec=(_NOCC, _NOCC), walker_kind="unrestricted"))


# ---------------------------------------------------------------------------
# kernels against the per-determinant RHF kernels
# ---------------------------------------------------------------------------


def _rhf_per_det(ham, walker, d):
    """(overlap, force bias, energy) of one sampled determinant with meas/rhf's kernels."""
    trial = RhfTrial(mo_coeff=d)
    ctx = meas_rhf.build_meas_ctx(ham, trial)
    olp = meas_rhf.overlap_r(walker, trial)
    fb = meas_rhf.force_bias_kernel_rw_rh(walker, ham, ctx, trial)
    e = meas_rhf.energy_kernel_rw_rh(walker, ham, ctx, trial)
    return olp, fb, e


def test_overlap_is_the_mean_of_the_sampled_determinants(random_case):
    ham, guide, walker = random_case["ham"], random_case["guide"], random_case["walker"]
    olps = np.array([_rhf_per_det(ham, walker, d)[0] for d in guide.dets])
    np.testing.assert_allclose(det_overlaps_r(walker, guide.dets), olps, rtol=1e-12)
    np.testing.assert_allclose(overlap_r(walker, guide), olps.mean(), rtol=1e-12)
    # one sample, no modes: the pt2CCSD (Thouless reference) overlap
    tau0 = jnp.zeros((0, _NOCC, _NORB - _NOCC), dtype=jnp.complex128)
    g1 = make_stoccsd_trial_data(
        {"mo_t": random_case["mo_t"], "tau": tau0}, random_case["sys"], n_samples=1
    )
    ref = pt2_overlap_r(
        walker,
        Pt2ccsdTrial(mo_t=jnp.asarray(random_case["mo_t"]), t2=jnp.asarray(random_case["t2"])),
    )
    np.testing.assert_allclose(overlap_r(walker, g1), ref, rtol=1e-12)


def test_force_bias_and_energy_are_overlap_weighted_rhf_kernels(random_case):
    ham, guide, walker = random_case["ham"], random_case["guide"], random_case["walker"]
    ctx = build_meas_ctx(ham, guide)
    assert isinstance(ctx, StoccsdMeasCtx) and ctx.chol_fb_flat.dtype == jnp.float64
    per_det = [_rhf_per_det(ham, walker, d) for d in guide.dets]
    olps = np.array([o for o, _, _ in per_det])
    fbs = np.array([f for _, f, _ in per_det])
    es = np.array([e for _, _, e in per_det])
    fb_ref = (olps[:, None] * fbs).sum(0) / olps.sum()
    e_ref = (olps * es).sum() / olps.sum()
    fb = force_bias_kernel_rw_rh(walker, ham, ctx, guide)
    e = energy_kernel_rw_rh(walker, ham, ctx, guide)
    assert fb.shape == (_NCHOL,)
    np.testing.assert_allclose(fb, fb_ref, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(e, e_ref, rtol=1e-11)
    # under jit and vmap over walkers as the propagator calls them
    walkers = jnp.stack([walker, jnp.linalg.qr(walker + 0.1j * jnp.eye(_NORB, _NOCC))[0]])
    fb_v = jax.jit(jax.vmap(force_bias_kernel_rw_rh, in_axes=(0, None, None, None)))(
        walkers, ham, ctx, guide
    )
    np.testing.assert_allclose(fb_v[0], fb_ref, rtol=1e-11, atol=1e-13)
    e_v = jax.jit(jax.vmap(energy_kernel_rw_rh, in_axes=(0, None, None, None)))(
        walkers, ham, ctx, guide
    )
    np.testing.assert_allclose(e_v[0], e_ref, rtol=1e-11)


def test_meas_ops_and_mixed_precision(random_case):
    ham, sys, guide, walker = (
        random_case["ham"],
        random_case["sys"],
        random_case["guide"],
        random_case["walker"],
    )
    ops = make_stoccsd_meas_ops(sys)
    assert ops.overlap is overlap_r
    assert ops.has_kernel(k_force_bias) and ops.has_kernel(k_energy)
    assert get_stoccsd_meas_cfg(ops).mixed_precision is False
    ctx = ops.build_meas_ctx(ham, guide)
    fb = ops.require_kernel(k_force_bias)(walker, ham, ctx, guide)
    e = ops.require_kernel(k_energy)(walker, ham, ctx, guide)

    ops_mp = make_stoccsd_meas_ops(sys, mixed_precision=True)
    assert get_stoccsd_meas_cfg(ops_mp).mixed_precision is True
    ctx_mp = ops_mp.build_meas_ctx(ham, guide)
    assert ctx_mp.chol_fb_flat.dtype == jnp.float32 and ctx_mp.chol.dtype == jnp.float64
    fb_mp = ops_mp.require_kernel(k_force_bias)(walker, ham, ctx_mp, guide)
    e_mp = ops_mp.require_kernel(k_energy)(walker, ham, ctx_mp, guide)
    assert fb_mp.dtype == jnp.complex128
    # single precision cholesky products (TF32 on GPU), the energy untouched
    np.testing.assert_allclose(fb_mp, fb, rtol=5e-3, atol=5e-3 * float(jnp.abs(fb).max()))
    np.testing.assert_allclose(e_mp, e, rtol=1e-12)
    with pytest.raises(ValueError, match="walker_kind"):
        make_stoccsd_meas_ops(System(norb=_NORB, nelec=(_NOCC, _NOCC), walker_kind="generalized"))


# ---------------------------------------------------------------------------
# the sampled overlap against the exact CCSD state in Fock space
# ---------------------------------------------------------------------------


def _apply_e(ci, p, q, norb, nelec):
    """E_pq |ci> = sum_sigma a+_{p sigma} a_{q sigma} |ci>."""
    na, nb = nelec
    out = addons.cre_a(addons.des_a(ci, norb, (na, nb), q), norb, (na - 1, nb), p)
    out = out + addons.cre_b(addons.des_b(ci, norb, (na, nb), q), norb, (na, nb - 1), p)
    return out


def _apply_t(ci, t1, t2, norb, nelec):
    """(T1 + T2)|ci> with T1 = sum t_ia E_ai and T2 = 1/2 sum t_iajb E_ai E_bj."""
    nocc, nvir = t1.shape
    out = np.zeros_like(ci)
    for i in range(nocc):
        for a in range(nvir):
            out += t1[i, a] * _apply_e(ci, nocc + a, i, norb, nelec)
    for j in range(nocc):
        for b in range(nvir):
            ebj = _apply_e(ci, nocc + b, j, norb, nelec)
            for i in range(nocc):
                for a in range(nvir):
                    out += 0.5 * t2[i, a, j, b] * _apply_e(ebj, nocc + a, i, norb, nelec)
    return out


def _exp_t_hf(t1, t2, norb, nelec, order):
    """sum_{k<=order} T^k/k! |HF> as an FCI vector."""
    na, nb = nelec
    ci = np.zeros((cistring.num_strings(norb, na), cistring.num_strings(norb, nb)))
    ci[0, 0] = 1.0
    out, term = ci.copy(), ci.copy()
    for k in range(1, order + 1):
        term = _apply_t(term, t1, t2, norb, nelec) / k
        out += term
    return out


def _fci_overlap(ci, walker, norb, nocc):
    """<ci|walker> for a restricted walker: sum_IJ ci_IJ^* det(w[I]) det(w[J])."""
    strings = cistring.make_strings(range(norb), nocc)
    occs = [[p for p in range(norb) if (s >> p) & 1] for s in strings]
    dets = np.array([np.linalg.det(np.asarray(walker)[occ, :]) for occ in occs])
    return np.einsum("ij,i,j->", np.conj(ci), dets, dets)


def test_fock_space_reference_agrees_with_the_cisd_overlap():
    """Pins the sign and factor conventions of the Fock space construction."""
    norb, nocc = 5, 2
    nvir = norb - nocc
    k1, k2, kw = jax.random.split(jax.random.PRNGKey(21), 3)
    t1 = 0.1 * np.asarray(jax.random.normal(k1, (nocc, nvir)))
    t2 = np.asarray(_random_t2(k2, nocc, nvir, scale=0.2))
    walker = rand_orthonormal_cols(kw, norb, nocc)
    ci = _exp_t_hf(t1, t2, norb, (nocc, nocc), order=1)  # (1 + T1 + T2)|HF>
    ref = _fci_overlap(ci, walker, norb, nocc)
    cisd = cisd_overlap_r(walker, CisdTrial(ci1=jnp.asarray(t1), ci2=jnp.asarray(t2)))
    np.testing.assert_allclose(cisd, ref, rtol=1e-10)


def test_sampled_overlap_converges_to_the_ccsd_overlap():
    """
    E_y <exp(y.tau) psi~|phi> = <exp(T1 + T2) HF|phi>, checked on a 2 pair, 5 orbital
    case where exp(T2) has a quadruples term the CISD overlap lacks.
    """
    norb, nocc = 5, 2
    nvir = norb - nocc
    k1, k2, kw, ks = jax.random.split(jax.random.PRNGKey(22), 4)
    t1 = 0.1 * np.asarray(jax.random.normal(k1, (nocc, nvir)))
    t2 = np.asarray(_random_t2(k2, nocc, nvir, scale=0.3))
    walker = rand_orthonormal_cols(kw, norb, nocc)
    walker, _ = jnp.linalg.qr(walker + 1.0 * jnp.eye(norb, nocc))

    exact = _fci_overlap(_exp_t_hf(t1, t2, norb, (nocc, nocc), order=2 * nocc), walker, norb, nocc)
    cisd = _fci_overlap(_exp_t_hf(t1, t2, norb, (nocc, nocc), order=1), walker, norb, nocc)
    assert abs(exact - cisd) > 1e-3  # the quadruples matter in this case

    mo_t = jnp.asarray(_thouless_np(nocc, norb, t1))
    tau = jnp.asarray(decompose_t2(t2, thresh=1e-12))
    n_batch, batch = 400, 2000
    sample_mean = jax.jit(
        lambda key: jnp.mean(det_overlaps_r(walker, sample_dets(mo_t, tau, key, batch)))
    )
    means = np.array([sample_mean(k) for k in jax.random.split(ks, n_batch)])
    est = means.mean()
    err = means.std(ddof=1) / np.sqrt(n_batch)
    assert abs(est - exact) < 4.0 * err + 1e-6, (est, exact, err)
    assert abs(est - exact) < abs(est - cisd)


# ---------------------------------------------------------------------------
# staging
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h4_ccsd():
    mol = gto.M(atom="H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6", basis="sto-3g", verbose=0)
    mf = scf.RHF(mol).run()
    mycc = cc.CCSD(mf).run()
    return mycc


def test_stage_stoccsd_guide(h4_ccsd):
    mycc = h4_ccsd
    nocc, nvir = mycc.t1.shape
    norb = nocc + nvir
    tr = stage_stoccsd_guide(mycc, t2_thresh=1e-10)
    assert tr.kind == STOCCSD_KIND and tr.source_kind == "cc"
    assert set(tr.data) == {"mo_t", "tau", "t2_thresh"}
    assert tr.data["mo_t"].shape == (norb, nocc)
    assert tr.data["tau"].shape[1:] == (nocc, nvir) and np.iscomplexobj(tr.data["tau"])
    t2_iajb = np.asarray(mycc.t2).transpose(0, 2, 1, 3)
    np.testing.assert_allclose(
        np.einsum("gia,gjb->iajb", tr.data["tau"], tr.data["tau"]).real, t2_iajb, atol=1e-9
    )
    # the same reference as the pt2CCSD trial
    from trot.mixed import stage_pt2ccsd_trial

    np.testing.assert_allclose(tr.data["mo_t"], stage_pt2ccsd_trial(mycc).data["mo_t"])
    # a threshold just under the largest |eigenvalue| keeps that one mode only
    evals = np.abs(np.linalg.eigvalsh(t2_iajb.reshape(nocc * nvir, nocc * nvir)))
    tr_cut = stage_stoccsd_guide(mycc, t2_thresh=0.999 * evals.max())
    assert tr_cut.data["tau"].shape[0] == 1 < tr.data["tau"].shape[0]
    # and the staged data build a guide
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")
    guide = make_stoccsd_trial_data(tr.data, sys, n_samples=4, seed=1)
    assert guide.dets.shape == (4, norb, nocc)
