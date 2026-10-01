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
from .upt2ccsd_uh import e2_0_g, l2t2_g, nchol_of

_UCISD_UH_MEAS_CFG_ATTR = "_ucisd_uh_meas_cfg"


def _greens(wa: jax.Array, wb: jax.Array, nocc_a: int, nocc_b: int):
    green_a = (wa @ jnp.linalg.inv(wa[:nocc_a, :])).T
    green_b = (wb @ jnp.linalg.inv(wb[:nocc_b, :])).T
    return green_a, green_b


def overlap_uw_uh(walker: tuple[jax.Array, jax.Array], trial_data: UcisdTrial) -> jax.Array:
    wa, wb = walker
    nocc_a, nocc_b = trial_data.nocc
    c1a, c1b = trial_data.c1a, trial_data.c1b
    c2aa, c2ab, c2bb = trial_data.c2aa, trial_data.c2ab, trial_data.c2bb
    green_a, green_b = _greens(wa, wb, nocc_a, nocc_b)
    green_a, green_b = green_a[:, nocc_a:], green_b[:, nocc_b:]
    o0 = jnp.linalg.det(wa[:nocc_a, :]) * jnp.linalg.det(wb[:nocc_b, :])
    o1 = jnp.einsum("ia,ia", c1a, green_a) + jnp.einsum("ia,ia", c1b, green_b)
    o2 = (
        0.5 * jnp.einsum("iajb,ia,jb", c2aa, green_a, green_a, optimize="optimal")
        + 0.5 * jnp.einsum("iajb,ia,jb", c2bb, green_b, green_b, optimize="optimal")
        + jnp.einsum("iajb,ia,jb", c2ab, green_a, green_b, optimize="optimal")
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


def force_bias_kernel_uw_uh(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: UcisdMeasCtxUh,
    trial_data: UcisdTrial,
) -> jax.Array:
    """<T| L_g |phi> / <T|phi> for every cholesky vector g."""
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

    chol_a, chol_b = ham_data.chol_a, ham_data.chol_b
    rot_chol_a = chol_a[:, :nocc_a, :]
    rot_chol_b = chol_b[:, :nocc_b, :]
    lg_a = jnp.einsum("gpj,pj->g", rot_chol_a, green_a)
    lg_b = jnp.einsum("gpj,pj->g", rot_chol_b, green_b)
    lg = lg_a + lg_b

    # reference
    fb_0 = lg

    # single excitations
    ci1g = jnp.einsum("pt,pt->", c1a, green_occ_a) + jnp.einsum("pt,pt->", c1b, green_occ_b)
    fb_1_1 = ci1g * lg
    ci1gp_a = jnp.einsum("pt,it->pi", c1a, greenp_a)
    ci1gp_b = jnp.einsum("pt,it->pi", c1b, greenp_b)
    gci1gp_a = jnp.einsum("pj,pi->ij", green_a, ci1gp_a)
    gci1gp_b = jnp.einsum("pj,pi->ij", green_b, ci1gp_b)
    fb_1_2 = -jnp.einsum("gij,ij->g", chol_a, gci1gp_a) - jnp.einsum("gij,ij->g", chol_b, gci1gp_b)
    fb_1 = fb_1_1 + fb_1_2

    # double excitations
    ci2g_a = jnp.einsum("ptqu,pt->qu", c2aa, green_occ_a)
    ci2g_b = jnp.einsum("ptqu,pt->qu", c2bb, green_occ_b)
    ci2g_ab_a = jnp.einsum("ptqu,qu->pt", c2ab, green_occ_b)
    ci2g_ab_b = jnp.einsum("ptqu,pt->qu", c2ab, green_occ_a)
    gci2g = (
        0.5 * jnp.einsum("qu,qu->", ci2g_a, green_occ_a)
        + 0.5 * jnp.einsum("qu,qu->", ci2g_b, green_occ_b)
        + jnp.einsum("pt,pt->", ci2g_ab_a, green_occ_a)
    )
    fb_2_1 = lg * gci2g
    ci2_green_a = (greenp_a @ (ci2g_a + ci2g_ab_a).T) @ green_a
    ci2_green_b = (greenp_b @ (ci2g_b + ci2g_ab_b).T) @ green_b
    fb_2_2 = -jnp.einsum("gij,ij->g", chol_a, ci2_green_a) - jnp.einsum(
        "gij,ij->g", chol_b, ci2_green_b
    )
    fb_2 = fb_2_1 + fb_2_2

    overlap = 1.0 + ci1g + gci2g
    return (fb_0 + fb_1 + fb_2) / overlap


def _chunk_terms(
    chol_a_c: jax.Array,
    chol_b_c: jax.Array,
    green: tuple[jax.Array, jax.Array],
    greenp: tuple[jax.Array, jax.Array],
    ci1_green: tuple[jax.Array, jax.Array],
    ci1g1: tuple[jax.Array, jax.Array],
    ci2_green: tuple[jax.Array, jax.Array],
    c1_r: tuple[jax.Array, jax.Array],
    c2_r: tuple[jax.Array, jax.Array, jax.Array],
    nocc: tuple[int, int],
    rtype: Any,
    ctype: Any,
) -> tuple[jax.Array, ...]:
    """
    The two body terms of one chunk of k cholesky vectors, summed over the chunk:

        e2_0      <h2>                          working precision, exact
        e2_1_2    -tr(L ci1_green) tr(G L)
        e2_1_3_1  G L . G L . c1 G
        e2_1_3_2  -G L c1 . G L
        e2_2_2_1  -tr(L ci2_green) tr(G L) / 2
        e2_2_2_2  G L . L ci2_green / 2
        e2_2_3    L c2 L

    All but e2_0 are in ctype. ci1_green, ci1g1 and ci2_green come in ctype, c1_r and c2_r
    in rtype. ci2_green is 8 * (same spin) + 2 * (opposite spin) per spin. chol is
    symmetric in its orbital indices, so one gl = G_ir L_g,qr serves every term.
    """
    nocc_a, nocc_b = nocc

    gl_a = jnp.einsum("ir,gqr->giq", green[0], chol_a_c, optimize="optimal")  # (k, nocc_a, norb_a)
    gl_b = jnp.einsum("ir,gqr->giq", green[1], chol_b_c, optimize="optimal")  # (k, nocc_b, norb_b)

    # reference: 1/2 [ (tr L_g G)^2 - tr(L_g G L_g G) ], coulomb over both spins
    e2_0_c, tr_gl = e2_0_g(gl_a[:, :, :nocc_a], gl_b[:, :, :nocc_b])

    chol_a_r, chol_b_r = chol_a_c.astype(rtype), chol_b_c.astype(rtype)
    gl_a, gl_b = gl_a.astype(ctype), gl_b.astype(ctype)
    glo_a, glo_b = gl_a[:, :, :nocc_a], gl_b[:, :, :nocc_b]
    tr_gl = tr_gl.astype(ctype)

    # single excitations
    lci1g = jnp.einsum("gij,ij->g", chol_a_r, ci1_green[0], optimize="optimal") + jnp.einsum(
        "gij,ij->g", chol_b_r, ci1_green[1], optimize="optimal"
    )
    e2_1_2 = -(lci1g @ tr_gl)
    e2_1_3_1 = jnp.einsum("gqp,grq,rp->", glo_a, glo_a, ci1g1[0], optimize="optimal") + jnp.einsum(
        "gqp,grq,rp->", glo_b, glo_b, ci1g1[1], optimize="optimal"
    )
    # afqmc's lci1: the virtual columns of gl contracted with c1
    glci1_a = jnp.einsum("gqt,pt->gpq", gl_a[:, :, nocc_a:], c1_r[0], optimize="optimal")
    glci1_b = jnp.einsum("gqt,pt->gpq", gl_b[:, :, nocc_b:], c1_r[1], optimize="optimal")
    e2_1_3_2 = -jnp.einsum("gpq,gpq->", glci1_a, glo_a, optimize="optimal") - jnp.einsum(
        "gpq,gpq->", glci1_b, glo_b, optimize="optimal"
    )

    # double excitations
    lci2g = jnp.einsum("gij,ij->g", chol_a_r, ci2_green[0], optimize="optimal") + jnp.einsum(
        "gij,ij->g", chol_b_r, ci2_green[1], optimize="optimal"
    )
    e2_2_2_1 = -(lci2g @ tr_gl) / 2.0
    # only the occupied rows of chol . ci2_green meet a nonzero row of gl
    lci2_green_a = jnp.einsum(
        "gir,qr->giq", chol_a_r[:, :nocc_a, :], ci2_green[0], optimize="optimal"
    )
    lci2_green_b = jnp.einsum(
        "gir,qr->giq", chol_b_r[:, :nocc_b, :], ci2_green[1], optimize="optimal"
    )
    e2_2_2_2 = 0.5 * (
        jnp.einsum("giq,giq->", gl_a, lci2_green_a, optimize="optimal")
        + jnp.einsum("giq,giq->", gl_b, lci2_green_b, optimize="optimal")
    )
    glgp_a = jnp.einsum("giq,qa->gia", gl_a, greenp[0], optimize="optimal")
    glgp_b = jnp.einsum("giq,qa->gia", gl_b, greenp[1], optimize="optimal")
    e2_2_3 = jnp.sum(l2t2_g(glgp_a, glgp_b, c2_r))

    return jnp.sum(e2_0_c), e2_1_2, e2_1_3_1, e2_1_3_2, e2_2_2_1, e2_2_2_2, e2_2_3


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

    ci2g_a = jnp.einsum("ptqu,pt->qu", c2aa, green_occ_a) / 4
    ci2g_b = jnp.einsum("ptqu,pt->qu", c2bb, green_occ_b) / 4
    ci2g_ab_a = jnp.einsum("ptqu,qu->pt", c2ab, green_occ_b)
    ci2g_ab_b = jnp.einsum("ptqu,pt->qu", c2ab, green_occ_a)
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

    greenp_c = (greenp_a.astype(ctype), greenp_b.astype(ctype))
    ci1_green_c = (ci1_green_a.astype(ctype), ci1_green_b.astype(ctype))
    ci1g1_c = ((c1a @ green_occ_a.T).astype(ctype), (c1b @ green_occ_b.T).astype(ctype))
    ci2_green_c = (
        (8 * ci2_green_a + 2 * ci2_green_ab_a).astype(ctype),
        (8 * ci2_green_b + 2 * ci2_green_ab_b).astype(ctype),
    )
    c1_r = (c1a.astype(rtype), c1b.astype(rtype))
    c2_r = (c2aa.astype(rtype), c2ab.astype(rtype), c2bb.astype(rtype))

    zero = 0.0 + 0j * hg

    def scanned_fun(carry, x):
        terms = _chunk_terms(
            x[0],
            x[1],
            (green_a, green_b),
            greenp_c,
            ci1_green_c,
            ci1g1_c,
            ci2_green_c,
            c1_r,
            c2_r,
            (nocc_a, nocc_b),
            rtype,
            ctype,
        )
        return tuple(c + t.astype(zero.dtype) for c, t in zip(carry, terms)), None

    (e2_0, e2_1_2, e2_1_3_1, e2_1_3_2, e2_2_2_1, e2_2_2_2, e2_2_3), _ = lax.scan(
        scanned_fun, (zero,) * 7, (chol_a, chol_b)
    )

    # single excitations
    e2_1_1 = e2_0 * ci1g
    e2_1 = e2_1_1 + e2_1_2 + e2_1_3_1 + e2_1_3_2

    # double excitations
    e2_2_1 = e2_0 * gci2g
    e2_2 = e2_2_1 + e2_2_2_1 + e2_2_2_2 + e2_2_3

    e2 = e2_0 + e2_1 + e2_2

    overlap = 1.0 + ci1g + gci2g
    return (e1 + e2) / overlap + e0


def make_ucisd_meas_ops_uh(
    sys: Any,
    mixed_precision: bool = False,
    *,
    testing: bool = False,
    nchol_chunk: int | None = None,
) -> MeasOps:
    """
    MeasOps of the UCISD trial on the unrestricted hamiltonian. Only unrestricted walkers
    are meaningful, as for make_uhf_meas_ops_uh. mixed_precision runs the local energy's
    <C1 h2> and <C2 h2> contractions in single precision; the overlap and the force bias
    stay in the working precision. nchol_chunk caps the cholesky vectors per step of the
    local energy's scan (None: DEFAULT_NCHOL_CHUNK).
    """
    wk = sys.walker_kind.lower()
    if wk != "unrestricted":
        raise ValueError(
            f"the unrestricted hamiltonian path requires walker_kind='unrestricted', got {wk!r}"
        )
    cfg = make_chunk_meas_cfg(
        mixed_precision=mixed_precision, testing=testing, nchol_chunk=nchol_chunk
    )
    meas_ops = MeasOps(
        overlap=overlap_uw_uh,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx_uh(ham_data, trial_data, cfg),
        kernels={k_force_bias: force_bias_kernel_uw_uh, k_energy: energy_kernel_uw_uh},
        observables={},
    )
    object.__setattr__(meas_ops, _UCISD_UH_MEAS_CFG_ATTR, cfg)
    return meas_ops


def get_ucisd_uh_meas_cfg(meas_ops: MeasOps) -> Pt2ccsdChunkMeasCfg | None:
    cfg = getattr(meas_ops, _UCISD_UH_MEAS_CFG_ATTR, None)
    return cfg if isinstance(cfg, Pt2ccsdChunkMeasCfg) else None
