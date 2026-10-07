"""
Restricted CISD measurement kernels on the restricted hamiltonian with the local energy
scanned over chunks of cholesky vectors (kernel suffix _rw_rh: restricted walker,
restricted hamiltonian, as meas/cisd.py names them).

meas/cisd.py sums the two body term over every cholesky vector at once (memory_mode
"high") or one vector at a time ("low"). These kernels follow meas/ucisd_uh.py instead:
the vectors come in chunks of meas_ctx.nchol_chunk (pad_reshape_chol, zero padded), the
energy is a lax.scan of plain einsums over them, and the large products are kept few and
real. Same trial (trial.cisd.CisdTrial, with its trial-core / active / outer layout),
same hamiltonian, same numbers as meas/cisd.py's kernels (tests/test_cisd_rh.py).

Conventions (nocc = nocc_full, the occupied orbitals of the correlation space; the CI
coefficients span the active blocks occ_act / vir_act):
    green      (nocc, norb)        [phi (phi_occ)^-1]^T, the half green's function, [I | G]
    green_act  (nocc_act, norb)    its active occupied rows
    x          (nocc_act, nvir_act) green[occ_act, vir_act]
    greenp     (norb, nvir_act)    [green[:, vir_act]; -1 on the vir_act rows; 0 outer]
    ci2x       ci2[p,t,q,u] -> ci2[p,u,q,t], the exchange ordering, so that the direct
               and exchange doubles contractions are one: ci2g = (2 ci2 - ci2x) . x
    gl         (k, nocc, norb)     green . L_g for a chunk of k vectors

With the doubles folded, the overlap is (1 + 2 ci1 . x + x . ci2g) det(phi_occ)^2, the
force bias 2 L . ([green; 0] - greenp (ci1 + ci2g)^T green_act / overlap), and the energy
the one of meas/cisd.py with the singles and doubles traces against chol merged into
one. chol is symmetric in its orbital indices, so one gl serves every term. The CI
coefficients and the cholesky vectors are real.

Precision: the force bias's products with the doubles and the cholesky tensor, and the
energy's <C1 h2> / <C2 h2> terms, run in the mixed dtypes of the measurement context;
gl, <h2> and every sum stay in the working precision (meas/cisd.py keeps its L c2 L term
in single precision unless testing=True). The overlap follows only when asked
(DEFAULT_OVERLAP_MIXED_PRECISION).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias, o_rdm1
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.cisd import CisdTrial
from .cisd import _greens_restricted, rdm1_kernel_rw
from .pt2ccsd_chunking import (
    Pt2ccsdChunkMeasCfg,
    make_chunk_meas_cfg,
    pad_reshape_chol,
    resolve_nchol_chunk,
)
from .ucisd_uh import _rdot, _split

_CISD_RH_MEAS_CFG_ATTR = "_cisd_rh_meas_cfg"

# whether the overlap's <C2> product follows mixed_precision when make_cisd_meas_ops_rh is
# not told. Off: the overlap sets the walker weights, so it stays in the working precision
DEFAULT_OVERLAP_MIXED_PRECISION: bool = False


def _ci2_eff(ci2: jax.Array) -> jax.Array:
    """2 ci2 - ci2x as a (nocc_act nvir_act, nocc_act nvir_act) matrix."""
    nov = ci2.shape[0] * ci2.shape[1]
    return (2.0 * ci2 - jnp.transpose(ci2, (0, 3, 2, 1))).reshape(nov, nov)


def _ci2_green_occ(
    ci2_eff: jax.Array, x: jax.Array, rtype: Any = None, ctype: Any = None
) -> jax.Array:
    """ci2g_qu = sum_pt (2 ci2 - ci2x)_ptqu x_pt, (nocc_act, nvir_act), back in x's dtype."""
    shape = x.shape
    xv = x.reshape(shape[0] * shape[1])
    if ctype is not None:
        xv = xv.astype(ctype)
    if rtype is not None:
        ci2_eff = ci2_eff.astype(rtype)
    return (jnp.real(xv) @ ci2_eff + 1.0j * (jnp.imag(xv) @ ci2_eff)).astype(x.dtype).reshape(shape)


def _active_blocks(green: jax.Array, trial_data: CisdTrial):
    """green_act, x and greenp of meas/cisd.py's _active_green_blocks."""
    occ_act, vir_act = trial_data.occ_act_slice, trial_data.vir_act_slice
    green_act = green[occ_act, :]
    x = green[occ_act, vir_act]
    greenp = jnp.zeros((trial_data.norb, trial_data.nvir), dtype=green.dtype)
    greenp = greenp.at[: trial_data.nocc_full, :].set(green[:, vir_act])
    greenp = greenp.at[vir_act, :].set(-jnp.eye(trial_data.nvir, dtype=green.dtype))
    return green_act, x, greenp


def overlap_rw_rh(
    walker: jax.Array, trial_data: CisdTrial, *, rtype: Any = None, ctype: Any = None
) -> jax.Array:
    """
    <T|phi>. rtype / ctype are the dtypes of the <C2> product (None: the working
    precision); the determinant, <C1> and the sums are always in the working precision.
    """
    nocc_full = trial_data.nocc_full
    wocc = walker[:nocc_full, :]
    green = _greens_restricted(walker, nocc_full)
    x = green[trial_data.occ_act_slice, trial_data.vir_act_slice]
    det0 = jnp.linalg.det(wocc)
    o1 = jnp.einsum("ia,ia->", trial_data.ci1, x)
    ci2g = _ci2_green_occ(_ci2_eff(trial_data.ci2), x, rtype, ctype)
    o2 = jnp.einsum("qu,qu->", ci2g, x)
    return (1.0 + 2.0 * o1 + o2) * det0 * det0


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CisdMeasCtxRh:
    """
    ci2_eff:      2 ci2 - ci2x as a matrix, in the mixed real dtype
    cfg:          static, the mixed dtypes
    nchol_chunk:  static, the even division of the cholesky index actually scanned
    """

    ci2_eff: jax.Array
    cfg: Pt2ccsdChunkMeasCfg
    nchol_chunk: int

    def tree_flatten(self):
        return (self.ci2_eff,), (self.cfg, self.nchol_chunk)

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, nchol_chunk = aux
        (ci2_eff,) = children
        return cls(ci2_eff=ci2_eff, cfg=cfg, nchol_chunk=nchol_chunk)


def build_meas_ctx_rh(
    ham_data: HamChol,
    trial_data: CisdTrial,
    cfg: Pt2ccsdChunkMeasCfg = Pt2ccsdChunkMeasCfg(),
) -> CisdMeasCtxRh:
    if ham_data.basis != "restricted":
        raise ValueError("CISD restricted MeasOps assumes HamChol.basis == 'restricted'.")
    nchol = int(ham_data.nchol) if ham_data.nchol is not None else int(ham_data.chol.shape[0])
    nchol_chunk = resolve_nchol_chunk(nchol, cfg.nchol_chunk)
    ci2_eff = _ci2_eff(trial_data.ci2).astype(cfg.mixed_real_dtype)
    return CisdMeasCtxRh(ci2_eff=ci2_eff, cfg=cfg, nchol_chunk=nchol_chunk)


def _common(walker: jax.Array, meas_ctx: CisdMeasCtxRh, trial_data: CisdTrial):
    """green, its active blocks, ci1 . x, ci2g, x . ci2g and the excitation overlap."""
    cfg = meas_ctx.cfg
    green = _greens_restricted(walker, trial_data.nocc_full)
    green_act, x, greenp = _active_blocks(green, trial_data)
    ci1g = jnp.einsum("pt,pt->", trial_data.ci1, x)
    ci2g = _ci2_green_occ(meas_ctx.ci2_eff, x, cfg.mixed_real_dtype, cfg.mixed_complex_dtype)
    gci2g = jnp.einsum("qu,qu->", ci2g, x)
    overlap = 1.0 + 2.0 * ci1g + gci2g
    return green, green_act, x, greenp, ci1g, ci2g, gci2g, overlap


def force_bias_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtxRh, trial_data: CisdTrial
) -> jax.Array:
    """
    <T| L_g |phi> / <T|phi> for every cholesky vector g, as 2 L . Y with
    Y = [green; 0] - greenp (ci1 + ci2g)^T green_act / overlap: every term is linear in
    the cholesky vector, so chol is contracted once, in the mixed dtypes of meas_ctx.cfg
    (the force bias only shifts the sampled fields, so its precision does not bias the
    walk).
    """
    cfg = meas_ctx.cfg
    rtype, ctype = cfg.mixed_real_dtype, cfg.mixed_complex_dtype
    green, green_act, x, greenp, ci1g, ci2g, gci2g, overlap = _common(walker, meas_ctx, trial_data)
    nocc, norb = green.shape
    lin = -((greenp @ (trial_data.ci1 + ci2g).T) @ green_act) / overlap
    lin = lin.at[:nocc, :].add(green)
    chol_flat = ham_data.chol.reshape(ham_data.chol.shape[0], norb * norb).astype(rtype)
    return 2.0 * _rdot(chol_flat, lin.reshape(norb * norb).astype(ctype)).astype(green.dtype)


def _chunk_terms(
    chol_c: jax.Array,
    green: jax.Array,
    x: jax.Array,
    ci12_green_c: jax.Array,
    ci2_green_c: jax.Array,
    ci1g1_c: jax.Array,
    ci1_r: jax.Array,
    ci2_eff: jax.Array,
    occ_act: slice,
    vir_act: slice,
    rtype: Any,
    ctype: Any,
) -> tuple[jax.Array, ...]:
    """
    The two body terms of one chunk of k cholesky vectors, summed over the chunk, in
    meas/cisd.py's grouping:

        e2_0      2 (tr gl)^2 - tr(gl gl)             working precision, exact
        e2_12     -4 tr(L (ci1_green + ci2_green)) tr(gl)
        e2_1_3    2 [ gl . gl . ci1 G  -  gl ci1 . gl ]
        e2_2_2_2  2 gl . L ci2_green
        e2_2_3    glgp . (2 ci2 - ci2x) . glgp          (L c2 L)

    All but e2_0 are in ctype; ci12_green_c, ci2_green_c and ci1g1_c come in ctype, ci1_r
    and ci2_eff in rtype.
    """
    nocc, norb = green.shape
    k = chol_c.shape[0]

    # green = [I | G]: gl = L[:, :nocc, :] + G . L[:, nocc:, :], two real products
    gl = chol_c[:, :nocc, :] + _split(
        lambda g: jnp.einsum("pa,gai->gpi", g, chol_c[:, nocc:, :], optimize="optimal"),
        green[:, nocc:],
    )
    glo = gl[:, :, :nocc]
    tr_gl = jnp.einsum("gpp->g", glo, optimize="optimal")
    e2_0 = 2.0 * jnp.sum(tr_gl * tr_gl) - jnp.einsum("gpq,gqp->", glo, glo, optimize="optimal")

    tr_gl_c = tr_gl.astype(ctype)
    chol_r = chol_c.astype(rtype)
    gl_c = gl.astype(ctype)
    glo_c, glv_c = gl_c[:, :, :nocc], gl_c[:, :, vir_act]

    # singles and doubles against tr(gl): one trace of chol
    lci12g = _rdot(chol_r.reshape(k, norb * norb), ci12_green_c.reshape(norb * norb))
    e2_12 = -4.0 * (lci12g @ tr_gl_c)

    # singles: lg1_gpq = glo_gqp (chol symmetric)
    e2_1_3_1 = jnp.einsum("gqp,gaq,ap->", glo_c, glo_c[:, occ_act, :], ci1g1_c, optimize="optimal")
    glci1 = _split(lambda y: jnp.einsum("gqt,at->gaq", y, ci1_r, optimize="optimal"), glv_c)
    e2_1_3_2 = -jnp.einsum("gaq,gaq->", glci1, glo_c[:, occ_act, :], optimize="optimal")
    e2_1_3 = 2.0 * (e2_1_3_1 + e2_1_3_2)

    # doubles: the occupied rows of chol against ci2_green meet gl
    lci2_green = _split(
        lambda y: jnp.einsum("gpi,ji->gpj", chol_r[:, :nocc, :], y, optimize="optimal"),
        ci2_green_c,
    )
    e2_2_2_2 = 2.0 * jnp.einsum("gpi,gpi->", gl_c, lci2_green, optimize="optimal")

    # gl . greenp on the active occupied rows: glo . green[:, vir_act] - gl[:, :, vir_act]
    glgp = (
        jnp.einsum(
            "gpj,ja->gpa", glo_c[:, occ_act, :], green[:, vir_act].astype(ctype), optimize="optimal"
        )
        - glv_c[:, occ_act, :]
    )
    nov = glgp.shape[1] * glgp.shape[2]
    glgp = glgp.reshape(k, nov)
    lc2 = _split(lambda y: y @ ci2_eff, glgp)
    e2_2_3 = jnp.einsum("gm,gm->", lc2, glgp, optimize="optimal")

    return e2_0, e2_12, e2_1_3, e2_2_2_2, e2_2_3


def energy_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: CisdMeasCtxRh, trial_data: CisdTrial
) -> jax.Array:
    """
    <T| H |phi> / <T|phi>, h0 included. The two body sum is scanned over chunks of
    meas_ctx.nchol_chunk cholesky vectors. The <C1 h2> and <C2 h2> contractions run in
    the mixed dtypes of meas_ctx.cfg; the one body terms, <h2> and every partial sum stay
    in the working precision.
    """
    cfg = meas_ctx.cfg
    rtype, ctype = cfg.mixed_real_dtype, cfg.mixed_complex_dtype
    occ_act, vir_act = trial_data.occ_act_slice, trial_data.vir_act_slice

    green, green_act, x, greenp, ci1g, ci2g, gci2g, overlap = _common(walker, meas_ctx, trial_data)
    nocc = green.shape[0]
    h1 = ham_data.h1
    e0 = ham_data.h0

    # ---- one body: 2 hg overlap - 2 h1 . greenp (ci1 + ci2g)^T green_act
    hg = jnp.einsum("pj,pj->", h1[:nocc, :], green, optimize="optimal")
    ci2_green = (greenp @ ci2g.T) @ green_act
    ci12_green = (greenp @ trial_data.ci1.T) @ green_act + ci2_green
    e1 = 2.0 * hg * overlap - 2.0 * jnp.einsum("ij,ij->", h1, ci12_green, optimize="optimal")

    # ---- two body, chunked over the cholesky index; a zero padded vector contributes to
    # no term
    chol, _, _, _ = pad_reshape_chol(ham_data.chol, meas_ctx.nchol_chunk)
    ci1g1_c = (trial_data.ci1 @ green[:, vir_act].T).astype(ctype)  # (nocc_act, nocc)
    zero = 0.0 + 0j * hg

    def scanned_fun(carry, chol_c):
        terms = _chunk_terms(
            chol_c,
            green,
            x,
            ci12_green.astype(ctype),
            ci2_green.astype(ctype),
            ci1g1_c,
            trial_data.ci1.astype(rtype),
            meas_ctx.ci2_eff,
            occ_act,
            vir_act,
            rtype,
            ctype,
        )
        return tuple(c + t.astype(zero.dtype) for c, t in zip(carry, terms)), None

    (e2_0, e2_12, e2_1_3, e2_2_2_2, e2_2_3), _ = lax.scan(scanned_fun, (zero,) * 5, chol)
    e2 = e2_0 * overlap + e2_12 + e2_1_3 + e2_2_2_2 + e2_2_3

    return (e1 + e2) / overlap + e0


def make_cisd_meas_ops_rh(
    sys: System,
    mixed_precision: bool = True,
    *,
    testing: bool = False,
    nchol_chunk: int | None = None,
    overlap_mixed_precision: bool | None = None,
) -> MeasOps:
    """
    MeasOps of the restricted CISD trial on the restricted hamiltonian, restricted
    walkers only, as meas.cisd.make_cisd_meas_ops. mixed_precision runs in single
    precision the local energy's <C1 h2> and <C2 h2> contractions and the force bias's
    products with the doubles and the cholesky tensor; the overlap follows only with
    overlap_mixed_precision (None: DEFAULT_OVERLAP_MIXED_PRECISION). nchol_chunk caps the
    cholesky vectors per step of the local energy's scan (None: DEFAULT_NCHOL_CHUNK).
    The rdm1 observable is meas.cisd's.
    """
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"CISD MeasOps currently supports only restricted walkers, got: {sys.walker_kind}"
        )
    cfg = make_chunk_meas_cfg(
        mixed_precision=mixed_precision, testing=testing, nchol_chunk=nchol_chunk
    )
    if overlap_mixed_precision is None:
        overlap_mixed_precision = DEFAULT_OVERLAP_MIXED_PRECISION
    overlap = overlap_rw_rh
    if mixed_precision and overlap_mixed_precision:
        overlap = partial(overlap_rw_rh, rtype=cfg.mixed_real_dtype, ctype=cfg.mixed_complex_dtype)
    meas_ops = MeasOps(
        overlap=overlap,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx_rh(ham_data, trial_data, cfg),
        kernels={k_force_bias: force_bias_kernel_rw_rh, k_energy: energy_kernel_rw_rh},
        observables={o_rdm1: rdm1_kernel_rw},
    )
    object.__setattr__(meas_ops, _CISD_RH_MEAS_CFG_ATTR, cfg)
    return meas_ops


def get_cisd_rh_meas_cfg(meas_ops: MeasOps) -> Pt2ccsdChunkMeasCfg | None:
    cfg = getattr(meas_ops, _CISD_RH_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, Pt2ccsdChunkMeasCfg) else None
