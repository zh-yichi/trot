"""
meas/cisd_rh.py: the restricted CISD guide's kernels with the local energy scanned over
chunks of cholesky vectors, after meas/ucisd_uh.py.

Checks:
  - overlap, force bias and energy reproduce meas/cisd.py's kernels (every vector at
    once, all in double) on unstructured random walkers, for trial layouts with a trial
    core and outer virtuals, and chunks that divide the cholesky index or zero pad it;
  - mixed_precision moves the force bias's and the energy's large products to single
    precision; the overlap follows only with overlap_mixed_precision;
  - the mixed recipe's cisd guide gets these kernels (tests/test_mixed_pt2ccsd.py checks
    the chunk).
"""

from trot import config

config.configure_once()

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trot.core.ops import k_energy, k_force_bias
from trot.core.system import System
from trot.meas import cisd as cisd_meas
from trot.meas.cisd_rh import (
    build_meas_ctx_rh,
    energy_kernel_rw_rh,
    force_bias_kernel_rw_rh,
    get_cisd_rh_meas_cfg,
    make_cisd_meas_ops_rh,
    overlap_rw_rh,
)
from trot.meas.pt2ccsd_chunking import Pt2ccsdChunkMeasCfg
from trot.testing import make_random_ham_chol
from trot.trial.cisd import CisdTrial, overlap_r

jax.config.update("jax_enable_x64", True)

_ARGS = (0, None, None, None)


def _case(seed, nocc_act, nvir_act, ncore, nouter, nchol, walker_kind, n_walkers=8, scale=0.3):
    nocc = ncore + nocc_act
    norb = nocc + nvir_act + nouter
    rng = np.random.default_rng(seed)
    ham = make_random_ham_chol(jax.random.PRNGKey(seed), norb=norb, n_chol=nchol)
    trial = CisdTrial(
        ci1=scale * jnp.asarray(rng.normal(size=(nocc_act, nvir_act))),
        ci2=scale * jnp.asarray(rng.normal(size=(nocc_act, nvir_act, nocc_act, nvir_act))),
        nocc_t_core=ncore,
        nvir_t_outer=nouter,
    )
    w = rng.normal(size=(n_walkers, norb, nocc)) + 1j * rng.normal(size=(n_walkers, norb, nocc))
    if walker_kind == "orthonormal":
        w = np.linalg.qr(w)[0]
    elif walker_kind == "near_reference":
        w = 0.2 * w
        w[:, :nocc] += np.eye(nocc)
    sys = System(norb=norb, nelec=(nocc, nocc), walker_kind="restricted")
    return sys, ham, trial, jnp.asarray(w)


@pytest.mark.parametrize(
    "nocc_act, nvir_act, ncore, nouter, nchol",
    [
        (3, 3, 0, 0, 7),
        (2, 4, 1, 0, 5),
        (3, 2, 0, 2, 9),
        (2, 3, 2, 1, 4),
        (1, 1, 1, 1, 1),
        (4, 5, 1, 2, 11),
    ],
)
@pytest.mark.parametrize("walker_kind", ["near_reference", "orthonormal", "gaussian"])
def test_kernels_match_meas_cisd(nocc_act, nvir_act, ncore, nouter, nchol, walker_kind):
    seed = 1000 * nocc_act + 100 * nvir_act + 10 * ncore + nouter + nchol
    sys, ham, trial, w = _case(seed, nocc_act, nvir_act, ncore, nouter, nchol, walker_kind)
    # testing=True keeps every contraction of meas.cisd's kernels in double precision
    ref = cisd_meas.make_cisd_meas_ops(sys, memory_mode="high", mixed_precision=False, testing=True)
    ctx_ref = ref.build_meas_ctx(ham, trial)
    o_ref = jax.vmap(overlap_r, (0, None))(w, trial)
    fb_ref = jax.vmap(ref.require_kernel(k_force_bias), _ARGS)(w, ham, ctx_ref, trial)
    e_ref = jax.vmap(ref.require_kernel(k_energy), _ARGS)(w, ham, ctx_ref, trial)

    o_new = jax.vmap(overlap_rw_rh, (0, None))(w, trial)
    np.testing.assert_allclose(np.asarray(o_new), np.asarray(o_ref), rtol=1e-11)
    # chunks that divide nchol, ones that pad the last chunk, and the whole index at once
    for nchol_chunk in sorted({1, 2, max(1, nchol - 1), nchol, None}, key=lambda c: (c is None, c)):
        ctx = build_meas_ctx_rh(ham, trial, Pt2ccsdChunkMeasCfg(nchol_chunk=nchol_chunk))
        fb_new = jax.vmap(force_bias_kernel_rw_rh, _ARGS)(w, ham, ctx, trial)
        scale = np.max(np.abs(np.asarray(fb_ref)), axis=1, keepdims=True)
        np.testing.assert_allclose(
            np.asarray(fb_new) / scale, np.asarray(fb_ref) / scale, atol=1e-11
        )
        e_new = jax.vmap(energy_kernel_rw_rh, _ARGS)(w, ham, ctx, trial)
        np.testing.assert_allclose(np.asarray(e_new), np.asarray(e_ref), rtol=1e-11)


def test_mixed_precision_wiring():
    sys, ham, trial, w = _case(3, 3, 3, 1, 1, 8, "near_reference")
    dp = make_cisd_meas_ops_rh(sys, mixed_precision=False, nchol_chunk=4)
    mp = make_cisd_meas_ops_rh(sys, mixed_precision=True, nchol_chunk=4)
    mp_ovlp = make_cisd_meas_ops_rh(sys, mixed_precision=True, overlap_mixed_precision=True)
    assert get_cisd_rh_meas_cfg(dp).mixed_complex_dtype == jnp.complex128
    assert get_cisd_rh_meas_cfg(mp).mixed_complex_dtype == jnp.complex64
    assert dp.overlap is overlap_rw_rh and mp.overlap is overlap_rw_rh
    ctx_dp, ctx_mp = dp.build_meas_ctx(ham, trial), mp.build_meas_ctx(ham, trial)
    assert ctx_dp.nchol_chunk == 4 and ctx_mp.ci2_eff.dtype == jnp.float32
    for name in (k_force_bias, k_energy):
        a = jax.vmap(dp.require_kernel(name), _ARGS)(w, ham, ctx_dp, trial)
        b = jax.vmap(mp.require_kernel(name), _ARGS)(w, ham, ctx_mp, trial)
        assert a.dtype == b.dtype
        # loose: float32 matmuls on a GPU (TF32) are only good to ~1e-4
        np.testing.assert_allclose(np.asarray(b), np.asarray(a), rtol=1e-2, atol=1e-3)
        assert not np.array_equal(np.asarray(a), np.asarray(b))
    o_dp = jax.vmap(dp.overlap, (0, None))(w, trial)
    o_mp = jax.vmap(mp_ovlp.overlap, (0, None))(w, trial)
    np.testing.assert_allclose(np.asarray(o_mp), np.asarray(o_dp), rtol=1e-3)
    assert not np.array_equal(np.asarray(o_mp), np.asarray(o_dp))
    with pytest.raises(ValueError, match="restricted walkers"):
        make_cisd_meas_ops_rh(System(norb=6, nelec=(3, 3), walker_kind="unrestricted"))
