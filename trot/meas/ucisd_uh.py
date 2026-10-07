"""
UCISD measurement kernels on the unrestricted (uchol) hamiltonian.

Overlap, force bias and local energy of an unrestricted walker (wa, wb), each spin in
its own orbital basis, against

    <T| = <HF| (1 + C1 + C2),   C1 = sum c1_ia i+ a,   C2 = 1/2 sum c2_iajb ... (same spin)
                                                       + sum c2_iajb ...     (alpha-beta)

with the reference of each spin in the leading nocc orbitals of that spin's basis
(kernel suffix _uw_uh: unrestricted walker, unrestricted hamiltonian; meas/ucisd.py's
_uw_rh kernels are the same trial on the restricted hamiltonian, where beta is rotated
into the alpha basis). Ported from afqmc's wavefunctions_unrestricted.ucisd
(_calc_overlap, _calc_force_bias, _calc_energy, _build_measurement_intermediates), whose
ham_data {"h1": [h1a, h1b], "chol": [La, Lb]} is HamCholU and whose wave_data
{ci1A, ci2AA, ...} is UcisdTrial.

The local energy scans its two body sum over chunks of nchol_chunk cholesky vectors, as
meas/upt2ccsd_bar_uh.py does: every three index intermediate is built inside the scan,
(k, nocc, norb) per spin for a chunk of k vectors, so the kernel's peak memory is set by
the chunk rather than by the number of vectors. The chunk size and the mixed dtypes of
the <C1 h2> and <C2 h2> contractions are the measurement context's (UcisdMeasCtxUh).

In every kernel the large products are kept few and real: the doubles meet the green's
function as (nocc nvir, nocc nvir) matrix products, the force bias contracts the cholesky
tensor once per spin (every term is linear in it), the energy's gl and gl . greenp use
the identity blocks of green and greenp, and a real matrix times a complex one is two
real products (_rdot, _split) rather than a complex promoted one. The force bias runs its
products in the mixed dtypes of the measurement context like the energy's <C1 h2> and
<C2 h2> terms; the overlap does only when asked (DEFAULT_OVERLAP_MIXED_PRECISION). The CI
coefficients and the cholesky vectors are real.

Conventions, per spin (dropping the spin label):
    green      (nocc, norb)   [phi (phi_occ)^-1]^T,   the half green's function
    green_occ  (nocc, nvir)   its virtual columns
    greenp     (norb, nvir)   [green_occ; -1]
    gl         (k, nocc, norb) green contracted with a chunk of chol, G_ir L_g,qr

With every coefficient zero the kernels reduce to meas.uhf_uh's, and with identical
alpha and beta bases to meas.ucisd's _uw_rh kernels (tests/test_uchol_ucisd.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..ham.chol_u import HamCholU
from ..trial.ucisd import UcisdTrial
from .pt2ccsd_chunking import (
    Pt2ccsdChunkMeasCfg,
    make_chunk_meas_cfg,
    pad_reshape_chol,
    resolve_nchol_chunk,
)
from .upt2ccsd_uh import nchol_of

_UCISD_UH_MEAS_CFG_ATTR = "_ucisd_uh_meas_cfg"


def _greens(wa: jax.Array, wb: jax.Array, nocc_a: int, nocc_b: int):
    green_a = (wa @ jnp.linalg.inv(wa[:nocc_a, :])).T
    green_b = (wb @ jnp.linalg.inv(wb[:nocc_b, :])).T
    return green_a, green_b


# whether the overlap's <C2> products follow mixed_precision when make_ucisd_meas_ops_uh is
# not told. Off: the overlap sets the walker weights, so it stays in the working precision
DEFAULT_OVERLAP_MIXED_PRECISION: bool = False


def _rdot(mat: jax.Array, vec: jax.Array) -> jax.Array:
    """
    mat @ vec for a real mat and a complex vec, as two real products: mat is not promoted
    to complex, which would copy it and double the multiplications.
    """
    return lax.complex(mat @ jnp.real(vec), mat @ jnp.imag(vec))


def _ldot(vec: jax.Array, mat: jax.Array) -> jax.Array:
    """vec @ mat for a complex vec and a real mat, as _rdot."""
    return lax.complex(jnp.real(vec) @ mat, jnp.imag(vec) @ mat)


def _ci2_green_occ(
    trial_data: UcisdTrial,
    green_occ_a: jax.Array,
    green_occ_b: jax.Array,
    rtype: Any = None,
    ctype: Any = None,
    *,
    with_ab_b: bool = True,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array | None]:
    """
    The doubles contracted with the virtual columns of the green's functions,

        ci2g_a     c2aa_ptqu G^a_pt   (nocc_a, nvir_a)
        ci2g_b     c2bb_ptqu G^b_pt   (nocc_b, nvir_b)
        ci2g_ab_a  c2ab_ptqu G^b_qu   (nocc_a, nvir_a)
        ci2g_ab_b  c2ab_ptqu G^a_pt   (nocc_b, nvir_b), None unless with_ab_b

    each one (nocc nvir, nocc nvir) matrix product. rtype / ctype are the dtypes of the
    products (None: those of the operands); the results come back in green's dtype.
    """
    wtype = green_occ_a.dtype
    shape_a, shape_b = green_occ_a.shape, green_occ_b.shape
    nov_a, nov_b = shape_a[0] * shape_a[1], shape_b[0] * shape_b[1]
    g_a, g_b = green_occ_a.reshape(nov_a), green_occ_b.reshape(nov_b)
    c2aa = trial_data.c2aa.reshape(nov_a, nov_a)
    c2ab = trial_data.c2ab.reshape(nov_a, nov_b)
    c2bb = trial_data.c2bb.reshape(nov_b, nov_b)
    if ctype is not None:
        g_a, g_b = g_a.astype(ctype), g_b.astype(ctype)
    if rtype is not None:
        c2aa, c2ab, c2bb = c2aa.astype(rtype), c2ab.astype(rtype), c2bb.astype(rtype)

    ci2g_a = _ldot(g_a, c2aa).astype(wtype).reshape(shape_a)
    ci2g_b = _ldot(g_b, c2bb).astype(wtype).reshape(shape_b)
    ci2g_ab_a = _rdot(c2ab, g_b).astype(wtype).reshape(shape_a)
    ci2g_ab_b = _ldot(g_a, c2ab).astype(wtype).reshape(shape_b) if with_ab_b else None
    return ci2g_a, ci2g_b, ci2g_ab_a, ci2g_ab_b


def overlap_uw_uh(
    walker: tuple[jax.Array, jax.Array],
    trial_data: UcisdTrial,
    *,
    rtype: Any = None,
    ctype: Any = None,
) -> jax.Array:
    """
    <T|phi>. rtype / ctype are the dtypes of the <C2> products (None: the working
    precision); the determinants, <C1> and the sums are always in the working precision.
    """
    wa, wb = walker
    nocc_a, nocc_b = trial_data.nocc
    c1a, c1b = trial_data.c1a, trial_data.c1b
    green_a, green_b = _greens(wa, wb, nocc_a, nocc_b)
    green_occ_a, green_occ_b = green_a[:, nocc_a:], green_b[:, nocc_b:]
    o0 = jnp.linalg.det(wa[:nocc_a, :]) * jnp.linalg.det(wb[:nocc_b, :])
    o1 = jnp.einsum("ia,ia", c1a, green_occ_a) + jnp.einsum("ia,ia", c1b, green_occ_b)
    ci2g_a, ci2g_b, ci2g_ab_a, _ = _ci2_green_occ(
        trial_data, green_occ_a, green_occ_b, rtype, ctype, with_ab_b=False
    )
    o2 = (
        0.5 * jnp.einsum("qu,qu->", ci2g_a, green_occ_a)
        + 0.5 * jnp.einsum("qu,qu->", ci2g_b, green_occ_b)
        + jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a)
    )
    return (1.0 + o1 + o2) * o0


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class UcisdMeasCtxUh:
    """
    Static settings of the energy kernel; the kernels read the hamiltonian directly, and
    afqmc's lci1 intermediates, a (n_chol, norb, nocc) tensor per spin, are rebuilt chunk
    by chunk in the energy scan.
    """

    cfg: Pt2ccsdChunkMeasCfg  # static
    nchol_chunk: int  # static, the even division of the cholesky index actually scanned

    def tree_flatten(self):
        return (), (self.cfg, self.nchol_chunk)

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, nchol_chunk = aux
        return cls(cfg=cfg, nchol_chunk=nchol_chunk)


def build_meas_ctx_uh(
    ham_data: HamCholU,
    trial_data: UcisdTrial,
    cfg: Pt2ccsdChunkMeasCfg = Pt2ccsdChunkMeasCfg(),
) -> UcisdMeasCtxUh:
    if ham_data.basis != "uchol":
        raise ValueError("UCISD unrestricted MeasOps assumes HamCholU.basis == 'uchol'.")
    nchol_chunk = resolve_nchol_chunk(nchol_of(ham_data), cfg.nchol_chunk)
    return UcisdMeasCtxUh(cfg=cfg, nchol_chunk=nchol_chunk)


def _chol_dot_ci_green(
    chol: jax.Array,
    green: jax.Array,
    ci_green_occ: jax.Array,
    overlap: jax.Array,
    rtype: Any,
    ctype: Any,
) -> jax.Array:
    """
    One spin of <T| L_g |phi> / <T|phi>. The reference, single and double excitation
    terms are all linear in the cholesky vector, so they are summed first and chol is
    contracted once:

        fb_g = sum_ij L_g,ij [ (G; 0) - greenp ci_green_occ^T G / overlap ]_ij

    with ci_green_occ = c1 + c2 . green_occ, (nocc, nvir), and (G; 0) the green's function
    padded with zero rows to (norb, norb). The (n_chol, norb^2) product is in rtype / ctype.
    """
    nocc, norb = green.shape
    greenp = jnp.vstack((green[:, nocc:], -jnp.eye(norb - nocc)))
    lin = -((greenp @ ci_green_occ.T) @ green) / overlap
    lin = lin.at[:nocc, :].add(green)
    chol_flat = chol.reshape(chol.shape[0], norb * norb).astype(rtype)
    return _rdot(chol_flat, lin.reshape(norb * norb).astype(ctype)).astype(green.dtype)


def force_bias_kernel_uw_uh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: UcisdMeasCtxUh,
    trial_data: UcisdTrial,
) -> jax.Array:
    """
    <T| L_g |phi> / <T|phi> for every cholesky vector g. The products with the doubles
    and with the cholesky tensor run in the mixed dtypes of meas_ctx.cfg: the force bias
    only shifts the sampled fields, so its precision does not bias the walk.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype

    wa, wb = walker
    nocc_a, nocc_b = trial_data.nocc
    c1a, c1b = trial_data.c1a, trial_data.c1b
    green_a, green_b = _greens(wa, wb, nocc_a, nocc_b)
    green_occ_a = green_a[:, nocc_a:]
    green_occ_b = green_b[:, nocc_b:]

    ci1g = jnp.einsum("pt,pt->", c1a, green_occ_a) + jnp.einsum("pt,pt->", c1b, green_occ_b)
    ci2g_a, ci2g_b, ci2g_ab_a, ci2g_ab_b = _ci2_green_occ(
        trial_data, green_occ_a, green_occ_b, rtype, ctype
    )
    gci2g = (
        0.5 * jnp.einsum("qu,qu->", ci2g_a, green_occ_a)
        + 0.5 * jnp.einsum("qu,qu->", ci2g_b, green_occ_b)
        + jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a)
    )
    overlap = 1.0 + ci1g + gci2g

    fb_a = _chol_dot_ci_green(
        ham_data.chol_a, green_a, c1a + ci2g_a + ci2g_ab_a, overlap, rtype, ctype
    )
    fb_b = _chol_dot_ci_green(
        ham_data.chol_b, green_b, c1b + ci2g_b + ci2g_ab_b, overlap, rtype, ctype
    )
    return fb_a + fb_b


def _split(fn: Any, x: jax.Array) -> jax.Array:
    """A linear fn with real operands applied to a complex x as two real calls."""
    return lax.complex(fn(jnp.real(x)), fn(jnp.imag(x)))


def _chunk_terms(
    chol_c: tuple[jax.Array, jax.Array],
    green_occ: tuple[jax.Array, jax.Array],
    ci12_green: tuple[jax.Array, jax.Array],
    ci1g1: tuple[jax.Array, jax.Array],
    ci2_green: tuple[jax.Array, jax.Array],
    c1_r: tuple[jax.Array, jax.Array],
    c2_r: tuple[jax.Array, jax.Array, jax.Array],
    rtype: Any,
    ctype: Any,
) -> tuple[jax.Array, ...]:
    """
    The two body terms of one chunk of k cholesky vectors, summed over the chunk:

        e2_0      <h2>                                     working precision, exact
        e2_12     -tr(L (ci1_green + ci2_green/2)) tr(G L)
        e2_1_3_1  G L . G L . c1 G
        e2_1_3_2  -G L c1 . G L
        e2_2_2_2  G L . L ci2_green / 2
        e2_2_3    L c2 L

    All but e2_0 are in ctype. ci12_green, ci1g1 and ci2_green come in ctype, c1_r and c2_r
    in rtype. ci2_green is 8 * (same spin) + 2 * (opposite spin) per spin. chol is
    symmetric in its orbital indices, so one gl = G_ir L_g,qr serves every term.

    The large products are kept real and small: green = [I | G_ov] makes gl the slice
    L[:, :, :nocc]^T plus G_ov . L[:, :, nocc:] (two real products in the working
    precision, L never promoted to complex), greenp = [G_ov; -I] makes gl . greenp the
    (nocc, nocc) product glo . G_ov - glv, the two traces against chol are one, and every
    real times complex contraction is two real ones (_split).
    """
    nocc = (green_occ[0].shape[0], green_occ[1].shape[0])

    # reference: 1/2 [ (tr L_g G)^2 - tr(L_g G L_g G) ], coulomb over both spins
    gl, tr_gl, ex_gl = [], 0.0, 0.0
    for s in range(2):
        chol_s, no = chol_c[s], nocc[s]
        gl_s = jnp.transpose(chol_s[:, :, :no], (0, 2, 1)) + _split(
            lambda x, chol_v=chol_s[:, :, no:]: jnp.einsum(
                "ia,gqa->giq", x, chol_v, optimize="optimal"
            ),
            green_occ[s],
        )
        glo_s = gl_s[:, :, :no]
        tr_gl = tr_gl + jnp.einsum("gpp->g", glo_s, optimize="optimal")
        ex_gl = ex_gl + jnp.einsum("gpq,gqp->g", glo_s, glo_s, optimize="optimal")
        gl.append(gl_s)
    e2_0 = jnp.sum((tr_gl * tr_gl - ex_gl) / 2.0)
    tr_gl = tr_gl.astype(ctype)

    lci12g = 0.0
    e2_1_3_1 = e2_1_3_2 = e2_2_2_2 = 0.0
    glgp = []
    for s in range(2):
        no = nocc[s]
        k, norb = chol_c[s].shape[0], chol_c[s].shape[1]
        chol_r = chol_c[s].astype(rtype)
        gl_c = gl[s].astype(ctype)
        glo, glv = gl_c[:, :, :no], gl_c[:, :, no:]

        # single and double excitations against tr(G L): one trace
        lci12g = lci12g + _rdot(chol_r.reshape(k, norb * norb), ci12_green[s].reshape(norb * norb))
        # single excitations
        e2_1_3_1 = e2_1_3_1 + jnp.einsum("gqp,grq,rp->", glo, glo, ci1g1[s], optimize="optimal")
        # afqmc's lci1: the virtual columns of gl contracted with c1
        glci1 = _split(
            lambda x, c1=c1_r[s]: jnp.einsum("gqt,pt->gpq", x, c1, optimize="optimal"), glv
        )
        e2_1_3_2 = e2_1_3_2 - jnp.einsum("gpq,gpq->", glci1, glo, optimize="optimal")
        # double excitations: only the occupied rows of chol . ci2_green meet a nonzero row
        # of gl
        lci2_green = _split(
            lambda x, chol_o=chol_r[:, :no, :]: jnp.einsum(
                "gir,qr->giq", chol_o, x, optimize="optimal"
            ),
            ci2_green[s],
        )
        e2_2_2_2 = e2_2_2_2 + 0.5 * jnp.einsum("giq,giq->", gl_c, lci2_green, optimize="optimal")
        # gl . greenp
        glgp.append(
            jnp.einsum("gij,ja->gia", glo, green_occ[s].astype(ctype), optimize="optimal") - glv
        )
    e2_12 = -(lci12g @ tr_gl)

    # L c2 L: 1/2 L c2aa L + 1/2 L c2bb L + L c2ab L
    c2aa_r, c2ab_r, c2bb_r = c2_r
    lc2_aa = _split(lambda x: jnp.einsum("gia,iajb->gjb", x, c2aa_r, optimize="optimal"), glgp[0])
    lc2_bb = _split(lambda x: jnp.einsum("gia,iajb->gjb", x, c2bb_r, optimize="optimal"), glgp[1])
    lc2_ab = _split(lambda x: jnp.einsum("gia,iajb->gjb", x, c2ab_r, optimize="optimal"), glgp[0])
    e2_2_3 = (
        0.5 * jnp.einsum("gjb,gjb->", lc2_aa, glgp[0], optimize="optimal")
        + 0.5 * jnp.einsum("gjb,gjb->", lc2_bb, glgp[1], optimize="optimal")
        + jnp.einsum("gjb,gjb->", lc2_ab, glgp[1], optimize="optimal")
    )

    return e2_0, e2_12, e2_1_3_1, e2_1_3_2, e2_2_2_2, e2_2_3


def energy_kernel_uw_uh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: UcisdMeasCtxUh,
    trial_data: UcisdTrial,
) -> jax.Array:
    """
    <T| H |phi> / <T|phi>, h0 included. The two body sum is scanned over chunks of
    meas_ctx.nchol_chunk cholesky vectors. The <C1 h2> and <C2 h2> contractions run in
    the mixed dtypes of meas_ctx.cfg; the one body terms, <h2> and every partial sum stay
    in the working precision.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype

    wa, wb = walker
    nocc_a, nocc_b = trial_data.nocc
    norb_a, norb_b = wa.shape[0], wb.shape[0]
    c1a, c1b = trial_data.c1a, trial_data.c1b
    c2aa, c2ab, c2bb = trial_data.c2aa, trial_data.c2ab, trial_data.c2bb
    green_a, green_b = _greens(wa, wb, nocc_a, nocc_b)
    green_occ_a = green_a[:, nocc_a:]
    green_occ_b = green_b[:, nocc_b:]
    greenp_a = jnp.vstack((green_occ_a, -jnp.eye(norb_a - nocc_a)))
    greenp_b = jnp.vstack((green_occ_b, -jnp.eye(norb_b - nocc_b)))

    h1_a, h1_b = ham_data.h1_a, ham_data.h1_b
    hg = jnp.einsum("pj,pj->", h1_a[:nocc_a, :], green_a) + jnp.einsum(
        "pj,pj->", h1_b[:nocc_b, :], green_b
    )

    e0 = ham_data.h0

    # ---- one body
    e1_0 = hg

    ci1g = jnp.einsum("pt,pt->", c1a, green_occ_a) + jnp.einsum("pt,pt->", c1b, green_occ_b)
    e1_1_1 = ci1g * hg
    ci1_green_a = (greenp_a @ c1a.T) @ green_a
    ci1_green_b = (greenp_b @ c1b.T) @ green_b
    e1_1_2 = -(jnp.einsum("ij,ij->", h1_a, ci1_green_a) + jnp.einsum("ij,ij->", h1_b, ci1_green_b))
    e1_1 = e1_1_1 + e1_1_2

    ci2g_a, ci2g_b, ci2g_ab_a, ci2g_ab_b = _ci2_green_occ(trial_data, green_occ_a, green_occ_b)
    ci2g_a, ci2g_b = ci2g_a / 4, ci2g_b / 4
    gci2g_a = jnp.einsum("qu,qu->", ci2g_a, green_occ_a)
    gci2g_b = jnp.einsum("qu,qu->", ci2g_b, green_occ_b)
    gci2g_ab = jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a)
    gci2g = 2 * (gci2g_a + gci2g_b) + gci2g_ab
    e1_2_1 = hg * gci2g
    ci2_green_a = (greenp_a @ ci2g_a.T) @ green_a
    ci2_green_ab_a = (greenp_a @ ci2g_ab_a.T) @ green_a
    ci2_green_b = (greenp_b @ ci2g_b.T) @ green_b
    ci2_green_ab_b = (greenp_b @ ci2g_ab_b.T) @ green_b
    e1_2_2 = -jnp.einsum("ij,ij->", h1_a, 4 * ci2_green_a + ci2_green_ab_a) - jnp.einsum(
        "ij,ij->", h1_b, 4 * ci2_green_b + ci2_green_ab_b
    )
    e1_2 = e1_2_1 + e1_2_2

    e1 = e1_0 + e1_1 + e1_2

    # ---- two body, chunked over the shared cholesky index. both spins are padded the
    # same way, and a zero vector contributes to no term
    chol_a, _, _, _ = pad_reshape_chol(ham_data.chol_a, meas_ctx.nchol_chunk)
    chol_b, _, _, _ = pad_reshape_chol(ham_data.chol_b, meas_ctx.nchol_chunk)

    ci2_green_a = 8 * ci2_green_a + 2 * ci2_green_ab_a
    ci2_green_b = 8 * ci2_green_b + 2 * ci2_green_ab_b
    ci12_green_c = (
        (ci1_green_a + 0.5 * ci2_green_a).astype(ctype),
        (ci1_green_b + 0.5 * ci2_green_b).astype(ctype),
    )
    ci1g1_c = ((c1a @ green_occ_a.T).astype(ctype), (c1b @ green_occ_b.T).astype(ctype))
    ci2_green_c = (ci2_green_a.astype(ctype), ci2_green_b.astype(ctype))
    c1_r = (c1a.astype(rtype), c1b.astype(rtype))
    c2_r = (c2aa.astype(rtype), c2ab.astype(rtype), c2bb.astype(rtype))

    zero = 0.0 + 0j * hg

    def scanned_fun(carry, x):
        terms = _chunk_terms(
            x,
            (green_occ_a, green_occ_b),
            ci12_green_c,
            ci1g1_c,
            ci2_green_c,
            c1_r,
            c2_r,
            rtype,
            ctype,
        )
        return tuple(c + t.astype(zero.dtype) for c, t in zip(carry, terms)), None

    (e2_0, e2_12, e2_1_3_1, e2_1_3_2, e2_2_2_2, e2_2_3), _ = lax.scan(
        scanned_fun, (zero,) * 6, (chol_a, chol_b)
    )

    # single and double excitations: <h2> times the excitation overlap, then the rest
    e2 = e2_0 * (1.0 + ci1g + gci2g) + e2_12 + e2_1_3_1 + e2_1_3_2 + e2_2_2_2 + e2_2_3

    overlap = 1.0 + ci1g + gci2g
    return (e1 + e2) / overlap + e0


def make_ucisd_meas_ops_uh(
    sys: Any,
    mixed_precision: bool = False,
    *,
    testing: bool = False,
    nchol_chunk: int | None = None,
    overlap_mixed_precision: bool | None = None,
) -> MeasOps:
    """
    MeasOps of the UCISD trial on the unrestricted hamiltonian. Only unrestricted walkers
    are meaningful, as for make_uhf_meas_ops_uh. mixed_precision runs in single precision
    the local energy's <C1 h2> and <C2 h2> contractions and the force bias's products with
    the doubles and the cholesky tensor. The overlap stays in the working precision unless
    overlap_mixed_precision (None: DEFAULT_OVERLAP_MIXED_PRECISION) is set as well, which
    runs its <C2> products in single precision too. nchol_chunk caps the cholesky vectors
    per step of the local energy's scan (None: DEFAULT_NCHOL_CHUNK).
    """
    wk = sys.walker_kind.lower()
    if wk != "unrestricted":
        raise ValueError(
            f"the unrestricted hamiltonian path requires walker_kind='unrestricted', got {wk!r}"
        )
    cfg = make_chunk_meas_cfg(
        mixed_precision=mixed_precision, testing=testing, nchol_chunk=nchol_chunk
    )
    if overlap_mixed_precision is None:
        overlap_mixed_precision = DEFAULT_OVERLAP_MIXED_PRECISION
    overlap = overlap_uw_uh
    if mixed_precision and overlap_mixed_precision:
        overlap = partial(overlap_uw_uh, rtype=cfg.mixed_real_dtype, ctype=cfg.mixed_complex_dtype)
    meas_ops = MeasOps(
        overlap=overlap,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx_uh(ham_data, trial_data, cfg),
        kernels={k_force_bias: force_bias_kernel_uw_uh, k_energy: energy_kernel_uw_uh},
        observables={},
    )
    object.__setattr__(meas_ops, _UCISD_UH_MEAS_CFG_ATTR, cfg)
    return meas_ops


def get_ucisd_uh_meas_cfg(meas_ops: MeasOps) -> Pt2ccsdChunkMeasCfg | None:
    cfg = getattr(meas_ops, _UCISD_UH_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, Pt2ccsdChunkMeasCfg) else None
