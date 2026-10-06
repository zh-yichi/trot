"""
Unrestricted (uchol) hamiltonian, phase two: the UCISD trial on it.

Checks:
  - on a random hamiltonian duplicated across spins, the uchol UCISD overlap, force bias
    and energy kernels reproduce meas.ucisd's restricted-hamiltonian kernels with the
    beta rotation set to the identity;
  - with every CI coefficient zero they reduce to the uchol UHF kernels;
  - the overlap and the force bias, which sum their terms before the large products,
    reproduce the term by term formulas they replace on unstructured random input;
  - a restricted CCSD converted with convert_to_uccsd gives the same run as Afqmc with
    unrestricted walkers (same seed), since the beta basis equals the alpha basis;
  - the energy at tau = 0 is the UCCSD energy, with and without a frozen core;
  - a genuine UCCSD (NH2) agrees with the alpha-basis run within statistical error;
  - staged inputs round trip through the uchol h5 layout with the UCISD trial.
"""

from trot import config

config.configure_once()

import contextlib
import io
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, gto, scf

from trot.afqmc import Afqmc, AfqmcUh
from trot.core.system import System, System_uh
from trot.ham.chol_u import HamCholU, from_ham_chol
from trot.meas.ucisd import (
    energy_kernel_uw_rh,
    force_bias_kernel_uw_rh,
    make_ucisd_meas_ops,
)
from trot.meas.ucisd_uh import (
    build_meas_ctx_uh,
    energy_kernel_uw_uh,
    get_ucisd_uh_meas_cfg,
    force_bias_kernel_uw_uh,
    make_ucisd_meas_ops_uh,
    overlap_uw_uh,
)
from trot.meas.pt2ccsd_chunking import Pt2ccsdChunkMeasCfg
from trot.meas.uhf_uh import build_meas_ctx_uh as uhf_build_meas_ctx_uh
from trot.meas.uhf_uh import energy_kernel_uw_uh as uhf_energy_kernel_uw_uh
from trot.meas.uhf_uh import force_bias_kernel_uw_uh as uhf_force_bias_kernel_uw_uh
from trot.testing import make_random_ham_chol
from trot.trial.ucisd import UcisdTrial, overlap_u
from trot.trial.ucisd_uh import get_rdm1_uh, make_ucisd_trial_data_uh
from trot.trial.uhf import UhfTrial, overlap_u as uhf_overlap_u

jax.config.update("jax_enable_x64", True)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------------------
# random hamiltonian: reduction to the restricted-hamiltonian kernels
# ---------------------------------------------------------------------------

_NORB, _NUP, _NDN, _NCHOL = 6, 3, 2, 7


def _random_ucisd_trial(key, norb, nup, ndn, scale1=0.05, scale2=0.02) -> UcisdTrial:
    """Random coefficients with the same-spin doubles antisymmetric in both index pairs."""
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    nva, nvb = norb - nup, norb - ndn

    def antisym(x):  # (i, j, a, b) -> antisymmetric in (i, j) and (a, b), then (i, a, j, b)
        x = x - x.transpose(1, 0, 2, 3)
        x = x - x.transpose(0, 1, 3, 2)
        return x.transpose(0, 2, 1, 3)

    return UcisdTrial(
        mo_coeff_a=jnp.eye(norb),
        mo_coeff_b=jnp.eye(norb),
        c1a=scale1 * jax.random.normal(k1, (nup, nva)),
        c1b=scale1 * jax.random.normal(k2, (ndn, nvb)),
        c2aa=scale2 * antisym(jax.random.normal(k3, (nup, nup, nva, nva))),
        c2ab=scale2 * jax.random.normal(k4, (nup, nva, ndn, nvb)),
        c2bb=scale2 * antisym(jax.random.normal(k5, (ndn, ndn, nvb, nvb))),
    )


@pytest.fixture(scope="module")
def random_pair():
    key = jax.random.PRNGKey(3)
    k_ham, k_trial, k_wa, k_wb = jax.random.split(key, 4)
    ham = make_random_ham_chol(k_ham, norb=_NORB, n_chol=_NCHOL)
    ham_u = from_ham_chol(ham)
    trial = _random_ucisd_trial(k_trial, _NORB, _NUP, _NDN)
    n_walkers = 4
    wa = jnp.eye(_NORB)[:, :_NUP] + 0.3 * (
        jax.random.normal(k_wa, (n_walkers, _NORB, _NUP))
        + 0.5j * jax.random.normal(k_wb, (n_walkers, _NORB, _NUP))
    )
    wb = jnp.eye(_NORB)[:, :_NDN] + 0.3 * (
        jax.random.normal(k_wb, (n_walkers, _NORB, _NDN))
        + 0.5j * jax.random.normal(k_wa, (n_walkers, _NORB, _NDN))
    )
    sys_r = System(norb=_NORB, nelec=(_NUP, _NDN), walker_kind="unrestricted")
    sys_u = System_uh(norb=(_NORB, _NORB), nelec=(_NUP, _NDN))
    return dict(ham=ham, ham_u=ham_u, trial=trial, walkers=(wa, wb), sys_r=sys_r, sys_u=sys_u)


def test_kernels_reduce_to_restricted(random_pair):
    ham, ham_u, trial = random_pair["ham"], random_pair["ham_u"], random_pair["trial"]
    wa, wb = random_pair["walkers"]
    # testing=True keeps every contraction of his kernels in double precision
    meas_r = make_ucisd_meas_ops(random_pair["sys_r"], mixed_precision=False, testing=True)
    ctx_r = meas_r.build_meas_ctx(ham, trial)
    ctx_u = build_meas_ctx_uh(ham_u, trial)
    for i in range(wa.shape[0]):
        w = (wa[i], wb[i])
        o_r, o_u = complex(overlap_u(w, trial)), complex(overlap_uw_uh(w, trial))
        assert o_u == pytest.approx(o_r, rel=1e-11)
        fb_r = force_bias_kernel_uw_rh(w, ham, ctx_r, trial)
        fb_u = force_bias_kernel_uw_uh(w, ham_u, ctx_u, trial)
        np.testing.assert_allclose(np.asarray(fb_u), np.asarray(fb_r), rtol=1e-10, atol=1e-12)
        e_r = complex(energy_kernel_uw_rh(w, ham, ctx_r, trial))
        e_u = complex(energy_kernel_uw_uh(w, ham_u, ctx_u, trial))
        assert e_u == pytest.approx(e_r, rel=1e-10)


def test_energy_is_independent_of_the_cholesky_chunk(random_pair):
    ham_u, trial = random_pair["ham_u"], random_pair["trial"]
    wa, wb = random_pair["walkers"]

    def ctx(nchol_chunk):
        return build_meas_ctx_uh(ham_u, trial, Pt2ccsdChunkMeasCfg(nchol_chunk=nchol_chunk))

    assert ctx(None).nchol_chunk == _NCHOL and ctx(3).nchol_chunk == 3
    for i in range(wa.shape[0]):
        w = (wa[i], wb[i])
        # one chunk holding every vector
        e_ref = complex(energy_kernel_uw_uh(w, ham_u, ctx(_NCHOL), trial))
        # 1 and 2 divide evenly, 3 and 5 pad the last chunk with zero vectors
        for nchol_chunk in (1, 2, 3, 5):
            e_c = complex(energy_kernel_uw_uh(w, ham_u, ctx(nchol_chunk), trial))
            assert e_c == pytest.approx(e_ref, rel=1e-12)


def test_mixed_precision_energy(random_pair):
    """Single precision <C1 h2> and <C2 h2> contractions, chosen through the meas ctx."""
    ham_u, trial, sys_u = random_pair["ham_u"], random_pair["trial"], random_pair["sys_u"]
    wa, wb = random_pair["walkers"]
    dp = make_ucisd_meas_ops_uh(sys_u, mixed_precision=False, nchol_chunk=3)
    mp = make_ucisd_meas_ops_uh(sys_u, mixed_precision=True, nchol_chunk=3)
    ctx_dp, ctx_mp = dp.build_meas_ctx(ham_u, trial), mp.build_meas_ctx(ham_u, trial)
    assert ctx_dp.cfg.mixed_complex_dtype == jnp.complex128
    assert ctx_mp.cfg.mixed_complex_dtype == jnp.complex64
    assert get_ucisd_uh_meas_cfg(mp).nchol_chunk == 3 and ctx_mp.nchol_chunk == 3
    for i in range(wa.shape[0]):
        w = (wa[i], wb[i])
        e_dp = energy_kernel_uw_uh(w, ham_u, ctx_dp, trial)
        e_mp = energy_kernel_uw_uh(w, ham_u, ctx_mp, trial)
        assert e_mp.dtype == e_dp.dtype
        # loose: float32 matmuls on a GPU (TF32) are only good to ~1e-4
        assert complex(e_mp) == pytest.approx(complex(e_dp), rel=1e-2)
        assert complex(e_mp) != complex(e_dp)


def _overlap_term_by_term(walker, td):
    """afqmc's _calc_overlap as ported first: one einsum per term, all in double."""
    wa, wb = walker
    nocc_a, nocc_b = td.nocc
    green_a = (wa @ jnp.linalg.inv(wa[:nocc_a, :])).T[:, nocc_a:]
    green_b = (wb @ jnp.linalg.inv(wb[:nocc_b, :])).T[:, nocc_b:]
    o0 = jnp.linalg.det(wa[:nocc_a, :]) * jnp.linalg.det(wb[:nocc_b, :])
    o1 = jnp.einsum("ia,ia", td.c1a, green_a) + jnp.einsum("ia,ia", td.c1b, green_b)
    o2 = (
        0.5 * jnp.einsum("iajb,ia,jb", td.c2aa, green_a, green_a)
        + 0.5 * jnp.einsum("iajb,ia,jb", td.c2bb, green_b, green_b)
        + jnp.einsum("iajb,ia,jb", td.c2ab, green_a, green_b)
    )
    return (1.0 + o1 + o2) * o0


def _force_bias_term_by_term(walker, ham, td):
    """afqmc's _calc_force_bias as ported first: chol contracted once per term."""
    wa, wb = walker
    nocc_a, nocc_b = td.nocc
    norb_a, norb_b = wa.shape[0], wb.shape[0]
    green_a = (wa @ jnp.linalg.inv(wa[:nocc_a, :])).T
    green_b = (wb @ jnp.linalg.inv(wb[:nocc_b, :])).T
    green_occ_a, green_occ_b = green_a[:, nocc_a:], green_b[:, nocc_b:]
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(norb_a - nocc_a)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(norb_b - nocc_b)))
    chol_a, chol_b = ham.chol_a, ham.chol_b
    lg = jnp.einsum("gpj,pj->g", chol_a[:, :nocc_a, :], green_a) + jnp.einsum(
        "gpj,pj->g", chol_b[:, :nocc_b, :], green_b
    )

    ci1g = jnp.einsum("pt,pt->", td.c1a, green_occ_a) + jnp.einsum("pt,pt->", td.c1b, green_occ_b)
    ci1gp_a = jnp.einsum("pt,it->pi", td.c1a, greenp_a)
    ci1gp_b = jnp.einsum("pt,it->pi", td.c1b, greenp_b)
    gci1gp_a = jnp.einsum("pj,pi->ij", green_a, ci1gp_a)
    gci1gp_b = jnp.einsum("pj,pi->ij", green_b, ci1gp_b)
    fb_1 = (
        ci1g * lg
        - jnp.einsum("gij,ij->g", chol_a, gci1gp_a)
        - jnp.einsum("gij,ij->g", chol_b, gci1gp_b)
    )

    ci2g_a = jnp.einsum("ptqu,pt->qu", td.c2aa, green_occ_a)
    ci2g_b = jnp.einsum("ptqu,pt->qu", td.c2bb, green_occ_b)
    ci2g_ab_a = jnp.einsum("ptqu,qu->pt", td.c2ab, green_occ_b)
    ci2g_ab_b = jnp.einsum("ptqu,pt->qu", td.c2ab, green_occ_a)
    gci2g = (
        0.5 * jnp.einsum("qu,qu->", ci2g_a, green_occ_a)
        + 0.5 * jnp.einsum("qu,qu->", ci2g_b, green_occ_b)
        + jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a)
    )
    ci2_green_a = (greenp_a @ (ci2g_a + ci2g_ab_a).T) @ green_a
    ci2_green_b = (greenp_b @ (ci2g_b + ci2g_ab_b).T) @ green_b
    fb_2 = (
        lg * gci2g
        - jnp.einsum("gij,ij->g", chol_a, ci2_green_a)
        - jnp.einsum("gij,ij->g", chol_b, ci2_green_b)
    )
    return (lg + fb_1 + fb_2) / (1.0 + ci1g + gci2g)


@pytest.mark.parametrize(
    "norb, nocc, nchol",
    [((6, 6), (3, 2), 7), ((7, 5), (2, 4), 5), ((5, 8), (1, 3), 9), ((4, 6), (3, 1), 1)],
)
@pytest.mark.parametrize("walker_kind", ["near_reference", "orthonormal", "gaussian"])
def test_overlap_and_force_bias_match_term_by_term(norb, nocc, nchol, walker_kind):
    """
    Nothing in the input is structured: norb and nocc differ between the spins, the
    doubles carry no index symmetry and the cholesky vectors are not symmetric, so a
    transposed or mispaired index in the summed form shows up.
    """
    (norb_a, norb_b), (nocc_a, nocc_b) = norb, nocc
    nvir_a, nvir_b = norb_a - nocc_a, norb_b - nocc_b
    rng = np.random.default_rng(norb_a + 10 * nocc_b + 100 * nchol)
    r = lambda *shape: jnp.asarray(rng.normal(size=shape))
    ham = HamCholU(
        h0=jnp.asarray(0.0),
        h1_a=r(norb_a, norb_a),
        h1_b=r(norb_b, norb_b),
        chol_a=r(nchol, norb_a, norb_a),
        chol_b=r(nchol, norb_b, norb_b),
    )
    td = UcisdTrial(
        mo_coeff_a=jnp.eye(norb_a),
        mo_coeff_b=jnp.eye(norb_b),
        c1a=0.3 * r(nocc_a, nvir_a),
        c1b=0.3 * r(nocc_b, nvir_b),
        c2aa=0.3 * r(nocc_a, nvir_a, nocc_a, nvir_a),
        c2ab=0.3 * r(nocc_a, nvir_a, nocc_b, nvir_b),
        c2bb=0.3 * r(nocc_b, nvir_b, nocc_b, nvir_b),
    )
    n_walkers = 16

    def walkers(n, o):
        x = rng.normal(size=(n_walkers, n, o)) + 1j * rng.normal(size=(n_walkers, n, o))
        if walker_kind == "orthonormal":
            x = np.linalg.qr(x)[0]
        elif walker_kind == "near_reference":
            x = 0.2 * x
            x[:, :o] += np.eye(o)
        return jnp.asarray(x)

    w = (walkers(norb_a, nocc_a), walkers(norb_b, nocc_b))
    ctx = build_meas_ctx_uh(ham, td)
    o_new = jax.vmap(overlap_uw_uh, (0, None))(w, td)
    o_ref = jax.vmap(_overlap_term_by_term, (0, None))(w, td)
    np.testing.assert_allclose(np.asarray(o_new), np.asarray(o_ref), rtol=1e-10)
    fb_new = jax.vmap(force_bias_kernel_uw_uh, (0, None, None, None))(w, ham, ctx, td)
    fb_ref = jax.vmap(_force_bias_term_by_term, (0, None, None))(w, ham, td)
    scale = np.max(np.abs(np.asarray(fb_ref)), axis=1, keepdims=True)
    np.testing.assert_allclose(np.asarray(fb_new) / scale, np.asarray(fb_ref) / scale, atol=1e-10)


def test_mixed_precision_force_bias_and_overlap(random_pair):
    """
    mixed_precision moves the force bias's large products to single precision; the
    overlap follows only with overlap_mixed_precision.
    """
    ham_u, trial, sys_u = random_pair["ham_u"], random_pair["trial"], random_pair["sys_u"]
    wa, wb = random_pair["walkers"]
    dp = make_ucisd_meas_ops_uh(sys_u, mixed_precision=False)
    mp = make_ucisd_meas_ops_uh(sys_u, mixed_precision=True)
    mp_ovlp = make_ucisd_meas_ops_uh(sys_u, mixed_precision=True, overlap_mixed_precision=True)
    assert dp.overlap is overlap_uw_uh and mp.overlap is overlap_uw_uh
    # the overlap flag alone does nothing: it rides on mixed_precision
    assert (
        make_ucisd_meas_ops_uh(sys_u, mixed_precision=False, overlap_mixed_precision=True).overlap
        is overlap_uw_uh
    )
    ctx_dp, ctx_mp = dp.build_meas_ctx(ham_u, trial), mp.build_meas_ctx(ham_u, trial)
    for i in range(wa.shape[0]):
        w = (wa[i], wb[i])
        fb_dp = force_bias_kernel_uw_uh(w, ham_u, ctx_dp, trial)
        fb_mp = force_bias_kernel_uw_uh(w, ham_u, ctx_mp, trial)
        assert fb_mp.dtype == fb_dp.dtype
        # loose: float32 matmuls on a GPU (TF32) are only good to ~1e-4
        np.testing.assert_allclose(np.asarray(fb_mp), np.asarray(fb_dp), rtol=1e-2, atol=1e-3)
        assert not np.array_equal(np.asarray(fb_mp), np.asarray(fb_dp))

        o_dp, o_mp = dp.overlap(w, trial), mp_ovlp.overlap(w, trial)
        assert o_mp.dtype == o_dp.dtype
        assert complex(o_mp) == pytest.approx(complex(o_dp), rel=1e-3)
        assert complex(o_mp) != complex(o_dp)


def test_zero_coefficients_reduce_to_uhf(random_pair):
    ham_u, trial = random_pair["ham_u"], random_pair["trial"]
    wa, wb = random_pair["walkers"]
    td0 = UcisdTrial(
        mo_coeff_a=trial.mo_coeff_a,
        mo_coeff_b=trial.mo_coeff_b,
        c1a=jnp.zeros_like(trial.c1a),
        c1b=jnp.zeros_like(trial.c1b),
        c2aa=jnp.zeros_like(trial.c2aa),
        c2ab=jnp.zeros_like(trial.c2ab),
        c2bb=jnp.zeros_like(trial.c2bb),
    )
    ctx0 = build_meas_ctx_uh(ham_u, td0)
    uhf = UhfTrial(trial.mo_coeff_a[:, :_NUP], trial.mo_coeff_b[:, :_NDN])
    uctx = uhf_build_meas_ctx_uh(ham_u, uhf)
    for i in range(wa.shape[0]):
        w = (wa[i], wb[i])
        assert complex(overlap_uw_uh(w, td0)) == pytest.approx(
            complex(uhf_overlap_u(w, uhf)), rel=1e-12
        )
        np.testing.assert_allclose(
            np.asarray(force_bias_kernel_uw_uh(w, ham_u, ctx0, td0)),
            np.asarray(uhf_force_bias_kernel_uw_uh(w, ham_u, uctx, uhf)),
            rtol=1e-11,
            atol=1e-13,
        )
        assert complex(energy_kernel_uw_uh(w, ham_u, ctx0, td0)) == pytest.approx(
            complex(uhf_energy_kernel_uw_uh(w, ham_u, uctx, uhf)), rel=1e-11
        )


def test_trial_data_and_rdm1(random_pair):
    trial, sys_u = random_pair["trial"], random_pair["sys_u"]
    data = {
        "ci1a": trial.c1a,
        "ci1b": trial.c1b,
        "ci2aa": trial.c2aa,
        "ci2ab": trial.c2ab,
        "ci2bb": trial.c2bb,
    }
    td = make_ucisd_trial_data_uh(data, sys_u)
    assert td.nocc == (_NUP, _NDN) and td.mo_coeff_a.shape == (_NORB, _NORB)
    dm_a, dm_b = get_rdm1_uh(td)
    np.testing.assert_allclose(np.asarray(dm_a), np.diag(np.arange(_NORB) < _NUP))
    np.testing.assert_allclose(np.asarray(dm_b), np.diag(np.arange(_NORB) < _NDN))
    with pytest.raises(ValueError, match="amplitudes span"):
        make_ucisd_trial_data_uh(data, System_uh(norb=(_NORB + 1, _NORB), nelec=(_NUP, _NDN)))
    meas = make_ucisd_meas_ops_uh(sys_u)
    assert meas.overlap is overlap_uw_uh


# ---------------------------------------------------------------------------
# molecules
# ---------------------------------------------------------------------------


def _h2o_ccsd():
    mol = gto.M(
        atom="""
        O        0.0000000000      0.0000000000      0.0000000000
        H        0.9562300000      0.0000000000      0.0000000000
        H       -0.2353791634      0.9268076728      0.0000000000
        """,
        basis="sto-6g",
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    return mycc


def _nh2_uccsd(frozen: int | None = None):
    mol = gto.M(
        atom="""
        N        0.0000000000      0.0000000000      0.0000000000
        H        1.0225900000      0.0000000000      0.0000000000
        H       -0.2281193615      0.9968208791      0.0000000000
        """,
        basis="sto-6g",
        spin=1,
        verbose=0,
    )
    mf = scf.UHF(mol).newton()
    mf.conv_tol = 1e-12
    mf.kernel()
    mycc = cc.UCCSD(mf, frozen=frozen)
    mycc.conv_tol = 1e-10
    mycc.kernel()
    return mycc


@pytest.fixture(scope="module")
def h2o_ucc():
    return cc.addons.convert_to_uccsd(_h2o_ccsd())


@pytest.fixture(scope="module")
def nh2_ucc():
    return _nh2_uccsd()


@pytest.fixture(scope="module")
def nh2_ucc_frozen():
    return _nh2_uccsd(frozen=1)


_PARAMS: dict[str, Any] = dict(
    n_eql_blocks=4, n_blocks=20, seed=1234, n_walkers=5, error_method="blocking"
)


def _tau0_energy(af) -> float:
    job = _quiet(af.build_job)
    state, _ = _quiet(job.prepare_runtime)
    return float(np.real(state.e_estimate))


def test_closed_shell_uccsd_reproduces_alpha_basis_run(h2o_ucc):
    """
    A restricted CCSD converted to UCCSD: both spins share the basis, so the uchol run
    with the UCISD trial must match Afqmc with unrestricted walkers, same seed.
    """
    ref = Afqmc(h2o_ucc, chol_cut=1e-6, **_PARAMS)
    ref.walker_kind = "unrestricted"
    ref.mixed_precision = False
    e_ref, err_ref = _quiet(ref.kernel)

    af = AfqmcUh(h2o_ucc, chol_cut=1e-6, **_PARAMS)
    af.mixed_precision = False
    e_u, err_u = _quiet(af.kernel)

    job = af.job
    assert job.staged.trial.kind == "ucisd"
    assert isinstance(job.trial_data, UcisdTrial) and isinstance(job.ham_data, HamCholU)
    assert e_u == pytest.approx(e_ref, abs=1e-8)
    assert err_u == pytest.approx(err_ref, abs=1e-8)


@pytest.mark.parametrize("fixture", ["nh2_ucc", "nh2_ucc_frozen"])
def test_tau0_energy_is_the_uccsd_energy(request, fixture):
    """<UCISD|H|HF>/<UCISD|HF> with the CC-derived coefficients is the UCCSD energy."""
    mycc = request.getfixturevalue(fixture)
    af = AfqmcUh(mycc, chol_cut=1e-8, **_PARAMS)
    assert _tau0_energy(af) == pytest.approx(mycc.e_tot, abs=1e-6)
    staged = af.stage()
    n = int(mycc.frozen or 0)
    assert staged.ham.frozen == (n, n) and af.norb_frozen_core == (n, n)
    assert staged.ham.norb == (mycc.mol.nao - n, mycc.mol.nao - n)


def test_frozen_core_must_match_cc(nh2_ucc_frozen):
    with pytest.raises(ValueError, match="contradicts cc.frozen"):
        AfqmcUh(nh2_ucc_frozen, norb_frozen_core=2, chol_cut=1e-6).stage()
    # repeating cc.frozen is fine
    AfqmcUh(nh2_ucc_frozen, norb_frozen_core=1, chol_cut=1e-6).stage()


def test_nh2_agrees_with_alpha_basis_run(nh2_ucc):
    """
    A genuine UCCSD: the bases differ, the walk is a different representation of the
    same operators. The alpha-basis regression value is from tests/test_ucisd.py.
    """
    e_alpha, err_alpha = -55.41533781603285, 0.0001071700818560977
    af = AfqmcUh(nh2_ucc, chol_cut=1e-6, **_PARAMS)
    af.mixed_precision = False
    e_u, err_u = _quiet(af.kernel)
    assert np.isfinite(e_u) and np.isfinite(err_u)
    assert abs(e_u - e_alpha) < 3.0 * np.hypot(err_u, err_alpha)


def test_staged_io_round_trip(nh2_ucc_frozen, tmp_path):
    path = tmp_path / "nh2_ucisd_uchol.h5"
    af = AfqmcUh(nh2_ucc_frozen, chol_cut=1e-6, **_PARAMS)
    af.mixed_precision = False
    _quiet(af.save_staged, path)
    e_direct, err_direct = _quiet(af.kernel)

    af2 = AfqmcUh.from_staged(path, **_PARAMS)
    af2.mixed_precision = False
    staged = af2.stage()
    assert staged.trial.kind == "ucisd" and staged.ham.frozen == (1, 1)
    e_loaded, err_loaded = _quiet(af2.kernel)
    assert e_loaded == pytest.approx(e_direct, abs=1e-10)
    assert err_loaded == pytest.approx(err_direct, abs=1e-10)


if __name__ == "__main__":
    pytest.main([__file__])
