from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ...cholesky import equal_chunks
from ...core.ops import MeasOps, k_energy
from ...ham.chol_u import HamCholU
from ...meas.pt2ccsd import (
    _MEMORY_MODES,
    _PT2CCSD_MEAS_CFG_ATTR,
    DEFAULT_NCHOL_CHUNK,
    ChunkPlan,
    Pt2ccsdMeasCfg,
    chol_sampling_proposal,
    get_pt2ccsd_meas_cfg,
    resolve_chol_budget,
)
from ...meas.upt2ccsd import (
    _e2_0_g,
    _nchol,
    _pad_reshape,
    build_bar_intermediates_u,
)
from ...meas.upt2ccsd import plan_chunking_for_run_u as _trot_plan_chunking_for_run_u
from ..trial.upt2ccsd import Upt2ccsdTrial, overlap_u
from .pt2ccsd import TRIAL_COMPONENTS

__all__ = [
    "TRIAL_COMPONENTS",
    "Pt2ccsdMeasCfg",
    "Upt2ccsdMeasCtx",
    "build_meas_ctx",
    "energy_kernel_uw_uh_bar",
    "energy_kernel_uw_uh_sto",
    "make_upt2ccsd_meas_ops",
    "get_pt2ccsd_meas_cfg",
    "plan_chunking_for_run_u",
]

# The unrestricted LNO fragment estimator, mirroring trot/meas/upt2ccsd.py and ported
# from afqmc's lno_afqmc/wavefunctions_unrestricted.py (class upt2ccsd: _calc_e0bar_frag,
# _t2eorb_tc, _build_measurement_intermediates).
#
# As the restricted one (lnoafqmc/meas/pt2ccsd.py) it is the bar estimator and only that:
# each spin's exp(T1) sits on that spin's hamiltonian and walker, the reference is the
# bare determinant, and the kernel returns TRIAL_COMPONENTS = [t2frg, e0frg, e1frg, e0]
# per walker, with E_F = <e0frg> + <e1frg> - <t2frg><e0> formed over the blocks
# (lnoafqmc.stat_utils). It measures on the uchol hamiltonian, alpha and beta each in
# their own LNO basis over one shared cholesky index.
#
# Differences from trot's energy_kernel_uw_uh_bar, which this otherwise follows line by
# line: the doubles are projected on their first occupied index, so the same-spin blocks
# are no longer antisymmetric under i <-> j (the exchange half of every T2-contracted
# one-body intermediate is formed instead of doubling the direct one) and t2ba is
# independent of t2ab; and e0frg needs the transformed fock matrices and the constant
# e0t1orb in the ctx.
#
# measure_type "sto_chol" samples the T2-contracted two-body sum of e1frg over the shared
# cholesky index (afqmc's upt2ccsd_sto_chol, trot's energy_kernel_uw_uh_sto), with the
# proposal scored by the per-vector fragment two-body energy of e0frg; see the restricted
# module for the reasoning.
#
# The kernel is a trial-side quantity only: the guide energy and the trial/guide overlap
# ratio come from the guide's own ops in blocks.block_frag.

_MEASURE_TYPES_U = ("bar", "sto_chol")


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Upt2ccsdMeasCtx:
    cfg: Pt2ccsdMeasCfg  # static
    nchol_chunk: int = 1  # static; sets the shape the chol tensors are reshaped to

    exp_t1_a: jax.Array | None = None  # (norb_a, norb_a)
    exp_t1_b: jax.Array | None = None  # (norb_b, norb_b)
    h1_bar_a: jax.Array | None = None  # (norb_a, norb_a)
    h1_bar_b: jax.Array | None = None  # (norb_b, norb_b)
    chol_bar_a: jax.Array | None = None  # (nchol, norb_a, norb_a)
    chol_bar_b: jax.Array | None = None  # (nchol, norb_b, norb_b)
    fock_bar_a: jax.Array | None = None  # (norb_a, norb_a), fock of H_bar at the reference
    fock_bar_b: jax.Array | None = None  # (norb_b, norb_b)
    e0t1orb: jax.Array | None = None  # scalar, <exp(T1)HF|H|HF> projected on the fragment

    def tree_flatten(self):
        children = (
            self.exp_t1_a,
            self.exp_t1_b,
            self.h1_bar_a,
            self.h1_bar_b,
            self.chol_bar_a,
            self.chol_bar_b,
            self.fock_bar_a,
            self.fock_bar_b,
            self.e0t1orb,
        )
        aux = (self.cfg, self.nchol_chunk)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, nchol_chunk = aux
        (
            exp_t1_a,
            exp_t1_b,
            h1_bar_a,
            h1_bar_b,
            chol_bar_a,
            chol_bar_b,
            fock_bar_a,
            fock_bar_b,
            e0t1orb,
        ) = children
        return cls(
            cfg=cfg,
            nchol_chunk=nchol_chunk,
            exp_t1_a=exp_t1_a,
            exp_t1_b=exp_t1_b,
            h1_bar_a=h1_bar_a,
            h1_bar_b=h1_bar_b,
            chol_bar_a=chol_bar_a,
            chol_bar_b=chol_bar_b,
            fock_bar_a=fock_bar_a,
            fock_bar_b=fock_bar_b,
            e0t1orb=e0t1orb,
        )


def ufock_from_chol(
    nocc: tuple[int, int], h1: tuple[jax.Array, jax.Array], chol: tuple[jax.Array, jax.Array]
) -> tuple[jax.Array, jax.Array]:
    """
    Unrestricted fock matrices of (h1_s, chol_s) at the reference occupying each spin's
    first nocc_s orbitals: h1_s + J[a + b] - K[s]. Written for non-symmetric chol (the
    transformed tensors), as afqmc's integral.get_ufock.
    """
    nocc_a, nocc_b = nocc
    chol_a, chol_b = chol
    tr_l = jnp.einsum("gii->g", chol_a[:, :nocc_a, :nocc_a], optimize="optimal") + jnp.einsum(
        "gii->g", chol_b[:, :nocc_b, :nocc_b], optimize="optimal"
    )
    out = []
    for h1_s, chol_s, nocc_s in ((h1[0], chol_a, nocc_a), (h1[1], chol_b, nocc_b)):
        jeff = jnp.einsum("gpq,g->pq", chol_s, tr_l, optimize="optimal")
        keff = jnp.einsum(
            "gpj,gjq->pq", chol_s[:, :, :nocc_s], chol_s[:, :nocc_s, :], optimize="optimal"
        )
        out.append(h1_s + jeff - keff)
    return out[0], out[1]


def _frag_two_body(
    lg_a: jax.Array, lg_b: jax.Array, prjlo_a: jax.Array, prjlo_b: jax.Array
) -> jax.Array:
    """
    1/2 sum_g [ (L_a P_a)(tr L_a + tr L_b) - (L_a L_a P_a) + the same for beta ], with
    lg_s a (k, nocc_s, nocc_s) contraction of the ov cholesky block with an ov matrix.
    The projected two-body form shared by e0t1orb (with t1) and e0frg (with the green).
    """
    tr = jnp.einsum("gjj->g", lg_a, optimize="optimal") + jnp.einsum("gjj->g", lg_b, optimize="optimal")
    coul = jnp.einsum("gik,ik,g->", lg_a, prjlo_a, tr, optimize="optimal") + jnp.einsum(
        "gik,ik,g->", lg_b, prjlo_b, tr, optimize="optimal"
    )
    exch = jnp.einsum("gij,gjk,ik->", lg_a, lg_a, prjlo_a, optimize="optimal") + jnp.einsum(
        "gij,gjk,ik->", lg_b, lg_b, prjlo_b, optimize="optimal"
    )
    return 0.5 * (coul - exch)


def e0t1orb_from_chol_u(ham_data: HamCholU, trial_data: Upt2ccsdTrial) -> jax.Array:
    """
    <exp(T1)HF| H |HF> restricted to the fragment, from the untransformed cholesky
    vectors. afqmc's ham_data['e0t1orb'] (aa + ab + ba + bb).
    """
    nocc_a, nocc_b = trial_data.nocc
    lt1_a = jnp.einsum(
        "ia,gja->gij", trial_data.t1a, ham_data.chol_a[:, :nocc_a, nocc_a:], optimize="optimal"
    )
    lt1_b = jnp.einsum(
        "ia,gja->gij", trial_data.t1b, ham_data.chol_b[:, :nocc_b, nocc_b:], optimize="optimal"
    )
    return _frag_two_body(lt1_a, lt1_b, trial_data.prjlo_a, trial_data.prjlo_b)


def build_meas_ctx(
    ham_data: HamCholU,
    trial_data: Upt2ccsdTrial,
    cfg: Pt2ccsdMeasCfg = Pt2ccsdMeasCfg(measure_type="bar"),
) -> Upt2ccsdMeasCtx:
    if ham_data.basis != "uchol":
        raise ValueError(
            "the unrestricted fragment pt2CCSD MeasOps assume HamCholU.basis == 'uchol'; "
            f"got {ham_data.basis!r}."
        )
    if cfg.measure_type not in _MEASURE_TYPES_U:
        raise ValueError(
            f"unknown measure_type {cfg.measure_type!r}; the LNO estimator has only {_MEASURE_TYPES_U}"
        )
    if cfg.memory_mode not in _MEMORY_MODES:
        raise ValueError(f"unknown memory_mode {cfg.memory_mode!r}; expected one of {_MEMORY_MODES}")

    nchol = _nchol(ham_data)
    requested = DEFAULT_NCHOL_CHUNK if cfg.nchol_chunk is None else int(cfg.nchol_chunk)
    if requested < 1:
        raise ValueError(f"nchol_chunk must be >= 1, got {cfg.nchol_chunk}")
    cap = min(requested, nchol) if nchol > 0 else requested
    _, nchol_chunk, _ = equal_chunks(nchol, cap)

    bar = build_bar_intermediates_u(ham_data, trial_data)
    fock_bar_a, fock_bar_b = ufock_from_chol(
        trial_data.nocc, (bar["h1_bar_a"], bar["h1_bar_b"]), (bar["chol_bar_a"], bar["chol_bar_b"])
    )
    e0t1orb = e0t1orb_from_chol_u(ham_data, trial_data)

    return Upt2ccsdMeasCtx(
        cfg=cfg,
        nchol_chunk=nchol_chunk,
        exp_t1_a=bar["exp_t1_a"],
        exp_t1_b=bar["exp_t1_b"],
        h1_bar_a=bar["h1_bar_a"],
        h1_bar_b=bar["h1_bar_b"],
        chol_bar_a=bar["chol_bar_a"],
        chol_bar_b=bar["chol_bar_b"],
        fock_bar_a=fock_bar_a,
        fock_bar_b=fock_bar_b,
        e0t1orb=e0t1orb,
    )


# ---------------------------------------------------------------------------------------
# pieces of the kernel
# ---------------------------------------------------------------------------------------


class _BarWalker(NamedTuple):
    """Chunk independent, per walker intermediates of the fragment bar kernel."""

    green_a: jax.Array  # (nocc_a, norb_a), the half green against the bare reference
    green_b: jax.Array  # (nocc_b, norb_b)
    greenp_a: jax.Array  # (norb_a, nvir_a)
    greenp_b: jax.Array  # (norb_b, nvir_b)
    t2_green_a: jax.Array  # (norb_a, norb_a)
    t2_green_b: jax.Array  # (norb_b, norb_b)
    gt2g: jax.Array  # <P T2>
    e1_0: jax.Array  # <h1>
    e1_2: jax.Array  # <P T2 h1>


def _t2_one_body(
    trial_data: Upt2ccsdTrial,
    greenov: tuple[jax.Array, jax.Array],
    greenrow: tuple[jax.Array, jax.Array],
    greenp: tuple[jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    The projected-T2 contractions that need no cholesky vector: gt2g = <P T2> and the
    (norb_s, norb_s) matrices t2_green_s = Gp_pb t_iajb G_ia G_jq connected within each
    spin. As trot's _t2_one_body, with the exchange halves of the same-spin blocks and
    the t2ba block kept, since the projection breaks i <-> j. afqmc's _t2eorb_tc.
    """
    t2aa, t2ab, t2ba, t2bb = trial_data.t2aa, trial_data.t2ab, trial_data.t2ba, trial_data.t2bb
    gov_a, gov_b = greenov

    t2g_aa_c = jnp.einsum("iajb,ia->jb", t2aa, gov_a, optimize="optimal") / 4
    t2g_aa_e = jnp.einsum("iajb,ja->ib", t2aa, gov_a, optimize="optimal") / 4
    t2g_bb_c = jnp.einsum("iajb,ia->jb", t2bb, gov_b, optimize="optimal") / 4
    t2g_bb_e = jnp.einsum("iajb,ja->ib", t2bb, gov_b, optimize="optimal") / 4
    t2g_ab_a = jnp.einsum("iajb,ia->jb", t2ab, gov_a, optimize="optimal") / 2  # beta ov
    t2g_ab_b = jnp.einsum("iajb,jb->ia", t2ab, gov_b, optimize="optimal") / 2  # alpha ov
    t2g_ba_a = jnp.einsum("iajb,jb->ia", t2ba, gov_a, optimize="optimal") / 2  # beta ov
    t2g_ba_b = jnp.einsum("iajb,ia->jb", t2ba, gov_b, optimize="optimal") / 2  # alpha ov

    gt2g_aa = jnp.einsum("jb,jb->", t2g_aa_c, gov_a, optimize="optimal")
    gt2g_bb = jnp.einsum("jb,jb->", t2g_bb_c, gov_b, optimize="optimal")
    gt2g_ab = jnp.einsum("jb,jb->", t2g_ab_a, gov_b, optimize="optimal")
    gt2g_ba = jnp.einsum("jb,jb->", t2g_ba_b, gov_a, optimize="optimal")
    gt2g = 2 * (gt2g_aa + gt2g_bb) + (gt2g_ab + gt2g_ba)

    # 2 * (same spin direct - exchange) + (both opposite spin blocks, contracted onto this spin)
    t2_green_a = greenp[0] @ (2 * (t2g_aa_c - t2g_aa_e) + t2g_ab_b + t2g_ba_b).T @ greenrow[0]
    t2_green_b = greenp[1] @ (2 * (t2g_bb_c - t2g_bb_e) + t2g_ba_a + t2g_ab_a).T @ greenrow[1]
    return gt2g, t2_green_a, t2_green_b


def _l2t2_g(
    glgp_a: jax.Array, glgp_b: jax.Array, t2_r: tuple[jax.Array, jax.Array, jax.Array, jax.Array]
) -> jax.Array:
    """
    e2_2_3 per cholesky vector: 1/2 (L t2aa L + L t2ab L + L t2ba L + L t2bb L), glgp_s the
    (k, nocc_s, nvir_s) contraction of gl with greenp.
    """
    t2aa_r, t2ab_r, t2ba_r, t2bb_r = t2_r
    lt2_aa = jnp.einsum("gia,iajb->gjb", glgp_a, t2aa_r, optimize="optimal")
    lt2_ab = jnp.einsum("gia,iajb->gjb", glgp_a, t2ab_r, optimize="optimal")
    lt2_ba = jnp.einsum("gia,iajb->gjb", glgp_b, t2ba_r, optimize="optimal")
    lt2_bb = jnp.einsum("gia,iajb->gjb", glgp_b, t2bb_r, optimize="optimal")
    l2t2_aa = jnp.einsum("gjb,gjb->g", lt2_aa, glgp_a, optimize="optimal")
    l2t2_ab = jnp.einsum("gjb,gjb->g", lt2_ab, glgp_b, optimize="optimal")
    l2t2_ba = jnp.einsum("gjb,gjb->g", lt2_ba, glgp_a, optimize="optimal")
    l2t2_bb = jnp.einsum("gjb,gjb->g", lt2_bb, glgp_b, optimize="optimal")
    return 0.5 * (l2t2_aa + l2t2_ab + l2t2_ba + l2t2_bb)


def _bar_walker(
    walker: tuple[jax.Array, jax.Array],
    meas_ctx: Upt2ccsdMeasCtx,
    trial_data: Upt2ccsdTrial,
    name: str,
) -> _BarWalker:
    """
    Transform the walker with exp(T1) and build everything that does not touch a cholesky
    vector. As trot's _bar_walker, with the fragment _t2_one_body.
    """
    h1_a, h1_b = meas_ctx.h1_bar_a, meas_ctx.h1_bar_b
    if (
        h1_a is None
        or h1_b is None
        or meas_ctx.exp_t1_a is None
        or meas_ctx.exp_t1_b is None
        or meas_ctx.chol_bar_a is None
        or meas_ctx.chol_bar_b is None
        or meas_ctx.fock_bar_a is None
        or meas_ctx.fock_bar_b is None
    ):
        raise ValueError(f"{name} needs the bar intermediates; build the ctx first.")

    nocc_a, nocc_b = trial_data.nocc
    nvir_a, nvir_b = trial_data.nvir
    wu, wd = walker
    walker_bar_a = meas_ctx.exp_t1_a @ wu
    walker_bar_b = meas_ctx.exp_t1_b @ wd

    green_a = (walker_bar_a @ jnp.linalg.inv(walker_bar_a[:nocc_a, :])).T
    green_b = (walker_bar_b @ jnp.linalg.inv(walker_bar_b[:nocc_b, :])).T
    greenp_a = jnp.vstack((green_a[:, nocc_a:], -jnp.eye(nvir_a, dtype=green_a.dtype)))
    greenp_b = jnp.vstack((green_b[:, nocc_b:], -jnp.eye(nvir_b, dtype=green_b.dtype)))

    # one body energy. only the occupied rows of h1 meet a nonzero row of the green
    e1_0 = jnp.einsum("pq,pq->", h1_a[:nocc_a, :], green_a, optimize="optimal") + jnp.einsum(
        "pq,pq->", h1_b[:nocc_b, :], green_b, optimize="optimal"
    )

    gt2g, t2_green_a, t2_green_b = _t2_one_body(
        trial_data,
        greenov=(green_a[:, nocc_a:], green_b[:, nocc_b:]),
        greenrow=(green_a, green_b),
        greenp=(greenp_a, greenp_b),
    )
    e1_2_1 = e1_0 * gt2g
    e1_2_2 = -jnp.einsum("pq,pq->", h1_a, t2_green_a, optimize="optimal") - jnp.einsum(
        "pq,pq->", h1_b, t2_green_b, optimize="optimal"
    )

    return _BarWalker(
        green_a=green_a,
        green_b=green_b,
        greenp_a=greenp_a,
        greenp_b=greenp_b,
        t2_green_a=t2_green_a,
        t2_green_b=t2_green_b,
        gt2g=gt2g,
        e1_0=e1_0,
        e1_2=e1_2_1 + e1_2_2,
    )


def _bar_chunk_terms(
    chol_a_c: jax.Array,
    chol_b_c: jax.Array,
    bw: _BarWalker,
    trial_data: Upt2ccsdTrial,
    t2_r: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    rtype: Any,
    ctype: Any,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    The two-body terms of one chunk of k cholesky vectors, per vector, as trot's
    _bar_chunk_terms (e2_0 exact in complex128, the three T2-contracted ones in ctype),
    with the fragment _l2t2_g.
    """
    nocc_a, nocc_b = trial_data.nocc

    gl_a = jnp.einsum("ir,gqr->giq", bw.green_a, chol_a_c, optimize="optimal")  # (k, nocc_a, norb_a)
    gl_b = jnp.einsum("ir,gqr->giq", bw.green_b, chol_b_c, optimize="optimal")  # (k, nocc_b, norb_b)
    e2_0_g, tr_gl = _e2_0_g(gl_a[:, :, :nocc_a], gl_b[:, :, :nocc_b])

    # e2_2_2_1: only the trace of chol . t2_green is needed
    lt2g = jnp.einsum(
        "gpr,pr->g", chol_a_c.astype(rtype), bw.t2_green_a.astype(ctype), optimize="optimal"
    ) + jnp.einsum(
        "gpr,pr->g", chol_b_c.astype(rtype), bw.t2_green_b.astype(ctype), optimize="optimal"
    )
    e2_2_2_1_g = -lt2g.astype(ctype) * tr_gl.astype(ctype)

    # e2_2_2_2: only the occupied rows of chol . t2_green meet a nonzero row of gl
    lt2_green_a = jnp.einsum(
        "gir,qr->giq",
        chol_a_c[:, :nocc_a, :].astype(rtype),
        bw.t2_green_a.astype(ctype),
        optimize="optimal",
    )
    lt2_green_b = jnp.einsum(
        "gir,qr->giq",
        chol_b_c[:, :nocc_b, :].astype(rtype),
        bw.t2_green_b.astype(ctype),
        optimize="optimal",
    )
    e2_2_2_2_g = jnp.einsum(
        "giq,giq->g", gl_a.astype(ctype), lt2_green_a.astype(ctype), optimize="optimal"
    ) + jnp.einsum("giq,giq->g", gl_b.astype(ctype), lt2_green_b.astype(ctype), optimize="optimal")

    # e2_2_3
    glgp_a = jnp.einsum("giq,qa->gia", gl_a.astype(ctype), bw.greenp_a.astype(ctype), optimize="optimal")
    glgp_b = jnp.einsum("giq,qa->gia", gl_b.astype(ctype), bw.greenp_b.astype(ctype), optimize="optimal")
    e2_2_3_g = _l2t2_g(glgp_a, glgp_b, t2_r)

    return e2_0_g.astype(jnp.complex128), e2_2_2_1_g, e2_2_2_2_g, e2_2_3_g


def _frag_two_body_g(
    lg_a: jax.Array, lg_b: jax.Array, prjlo_a: jax.Array, prjlo_b: jax.Array
) -> jax.Array:
    """_frag_two_body per cholesky vector, (k,): the summand, not the sum."""
    tr = jnp.einsum("gjj->g", lg_a, optimize="optimal") + jnp.einsum("gjj->g", lg_b, optimize="optimal")
    p_g = jnp.einsum("gik,ik->g", lg_a, prjlo_a, optimize="optimal") + jnp.einsum(
        "gik,ik->g", lg_b, prjlo_b, optimize="optimal"
    )
    x_g = jnp.einsum("gij,gjk,ik->g", lg_a, lg_a, prjlo_a, optimize="optimal") + jnp.einsum(
        "gij,gjk,ik->g", lg_b, lg_b, prjlo_b, optimize="optimal"
    )
    return 0.5 * (p_g * tr - x_g)


def _e0bar_frag_scored(
    bw: _BarWalker, meas_ctx: Upt2ccsdMeasCtx, trial_data: Upt2ccsdTrial
) -> tuple[jax.Array, jax.Array]:
    """
    _e0bar_frag, also returning the two-body term per cholesky vector, (nchol,): the
    scores the semistochastic kernel builds its proposal from. afqmc's
    _calc_e0bar_frag_scored.
    """
    nocc_a, nocc_b = trial_data.nocc
    prjlo_a, prjlo_b = trial_data.prjlo_a, trial_data.prjlo_b
    gov_a = bw.green_a[:, nocc_a:]
    gov_b = bw.green_b[:, nocc_b:]
    assert meas_ctx.fock_bar_a is not None and meas_ctx.fock_bar_b is not None
    assert meas_ctx.chol_bar_a is not None and meas_ctx.chol_bar_b is not None

    e1 = jnp.einsum(
        "ia,ik,ka->", gov_a, prjlo_a, meas_ctx.fock_bar_a[:nocc_a, nocc_a:], optimize="optimal"
    ) + jnp.einsum(
        "ia,ik,ka->", gov_b, prjlo_b, meas_ctx.fock_bar_b[:nocc_b, nocc_b:], optimize="optimal"
    )

    chol_ov_a = meas_ctx.chol_bar_a[:, :nocc_a, nocc_a:]
    chol_ov_b = meas_ctx.chol_bar_b[:, :nocc_b, nocc_b:]
    nchol = chol_ov_a.shape[0]
    nchunks, k, pad = equal_chunks(nchol, meas_ctx.nchol_chunk)
    chol_ov_a = _pad_reshape(chol_ov_a, nchunks, k, pad)
    chol_ov_b = _pad_reshape(chol_ov_b, nchunks, k, pad)

    def scanned_fun(carry, x):
        lg_a = jnp.einsum("gia,ka->gik", x[0], gov_a, optimize="optimal")
        lg_b = jnp.einsum("gia,ka->gik", x[1], gov_b, optimize="optimal")
        e2_g = _frag_two_body_g(lg_a, lg_b, prjlo_a, prjlo_b)
        return carry + jnp.sum(e2_g), e2_g

    e2, e2_chunks = lax.scan(scanned_fun, jnp.zeros((), dtype=gov_a.dtype), (chol_ov_a, chol_ov_b))
    return meas_ctx.e0t1orb + e1 + e2, e2_chunks.reshape(-1)[:nchol]


def _e0bar_frag(bw: _BarWalker, meas_ctx: Upt2ccsdMeasCtx, trial_data: Upt2ccsdTrial) -> jax.Array:
    """
    The projected correlation part of <HF| H_bar |walker_bar> / <HF|walker_bar>:
    e0t1orb + the projected fock ov term + the projected ov-ov two-body term. afqmc's
    _calc_e0bar_frag.
    """
    nocc_a, nocc_b = trial_data.nocc
    prjlo_a, prjlo_b = trial_data.prjlo_a, trial_data.prjlo_b
    gov_a = bw.green_a[:, nocc_a:]
    gov_b = bw.green_b[:, nocc_b:]
    assert meas_ctx.fock_bar_a is not None and meas_ctx.fock_bar_b is not None
    assert meas_ctx.chol_bar_a is not None and meas_ctx.chol_bar_b is not None

    e1 = jnp.einsum(
        "ia,ik,ka->", gov_a, prjlo_a, meas_ctx.fock_bar_a[:nocc_a, nocc_a:], optimize="optimal"
    ) + jnp.einsum(
        "ia,ik,ka->", gov_b, prjlo_b, meas_ctx.fock_bar_b[:nocc_b, nocc_b:], optimize="optimal"
    )

    chol_ov_a = meas_ctx.chol_bar_a[:, :nocc_a, nocc_a:]
    chol_ov_b = meas_ctx.chol_bar_b[:, :nocc_b, nocc_b:]
    nchunks, k, pad = equal_chunks(chol_ov_a.shape[0], meas_ctx.nchol_chunk)
    chol_ov_a = _pad_reshape(chol_ov_a, nchunks, k, pad)
    chol_ov_b = _pad_reshape(chol_ov_b, nchunks, k, pad)

    def scanned_fun(carry, x):
        lg_a = jnp.einsum("gia,ka->gik", x[0], gov_a, optimize="optimal")
        lg_b = jnp.einsum("gia,ka->gik", x[1], gov_b, optimize="optimal")
        return carry + _frag_two_body(lg_a, lg_b, prjlo_a, prjlo_b), None

    e2, _ = lax.scan(scanned_fun, jnp.zeros((), dtype=gov_a.dtype), (chol_ov_a, chol_ov_b))
    return meas_ctx.e0t1orb + e1 + e2


# ---------------------------------------------------------------------------------------
# kernel
# ---------------------------------------------------------------------------------------


def energy_kernel_uw_uh_bar(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: Upt2ccsdMeasCtx,
    trial_data: Upt2ccsdTrial,
) -> jax.Array:
    """
    The fragment pt2CCSD estimator for one unrestricted walker, [t2frg, e0frg, e1frg, e0].

    Follows trot's energy_kernel_uw_uh_bar: each spin's similarity transformed hamiltonian
    in meas_ctx, the walker exp_t1_s @ walker_s, and the (nocc_s, norb_s) half greens
    against the bare reference. ham_data is unused; the tensors are in meas_ctx.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    bw = _bar_walker(walker, meas_ctx, trial_data, "energy_kernel_uw_uh_bar")
    e0frg = _e0bar_frag(bw, meas_ctx, trial_data)

    chol_a, chol_b = meas_ctx.chol_bar_a, meas_ctx.chol_bar_b
    assert chol_a is not None and chol_b is not None
    nchunks, nchol_chunk, pad = equal_chunks(chol_a.shape[0], meas_ctx.nchol_chunk)
    chol_a = _pad_reshape(chol_a, nchunks, nchol_chunk, pad)
    chol_b = _pad_reshape(chol_b, nchunks, nchol_chunk, pad)

    t2_r = (
        trial_data.t2aa.astype(rtype),
        trial_data.t2ab.astype(rtype),
        trial_data.t2ba.astype(rtype),
        trial_data.t2bb.astype(rtype),
    )

    def scanned_fun(carry, x):
        e2_0_g, e2_2_2_1_g, e2_2_2_2_g, e2_2_3_g = _bar_chunk_terms(
            x[0], x[1], bw, trial_data, t2_r, rtype, ctype
        )
        carry[0] += jnp.sum(e2_0_g)
        carry[1] += jnp.sum(e2_2_2_1_g).astype(c128)
        carry[2] += jnp.sum(e2_2_2_2_g).astype(c128)
        carry[3] += jnp.sum(e2_2_3_g).astype(c128)
        return carry, None

    zero = jnp.zeros((), dtype=c128)
    [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(
        scanned_fun, [zero, zero, zero, zero], (chol_a, chol_b)
    )

    e2_2 = e2_0 * bw.gt2g + e2_2_2_1 + e2_2_2_2 + e2_2_3

    t2frg = bw.gt2g  # <HF| P T2 |walker_bar> / <HF|walker_bar>
    e0 = bw.e1_0 + e2_0  # <HF| H_bar |walker_bar> / <HF|walker_bar>
    e1frg = bw.e1_2 + e2_2  # <HF| P T2 H_bar |walker_bar> / <HF|walker_bar>

    return jnp.stack([t2frg, e0frg, e1frg, e0])


def energy_kernel_uw_uh_sto(
    walker: tuple[jax.Array, jax.Array],
    ham_data: HamCholU,
    meas_ctx: Upt2ccsdMeasCtx,
    trial_data: Upt2ccsdTrial,
    key: jax.Array | None = None,
) -> jax.Array:
    """
    The unrestricted fragment estimator with a semistochastic cholesky sum in e1frg,
    [t2frg, e0frg, e1frg, e0].

    Follows trot's energy_kernel_uw_uh_sto: e2_0, and with it e2_2_1 = e2_0 * gt2g, stays
    exact, and the three T2-contracted accumulators are split into an exactly summed head
    and an importance sampled tail. Both spins share one head/tail split and one set of
    draws, since they share the cholesky index. The proposal comes from the spin summed
    fragment two-body energies of e0frg.

    n_chol_head="full" removes the sampling and reproduces energy_kernel_uw_uh_bar; no key
    is drawn or needed in that limit.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    nocc_a, nocc_b = trial_data.nocc
    bw = _bar_walker(walker, meas_ctx, trial_data, "energy_kernel_uw_uh_sto")
    e0frg, e2frg_g = _e0bar_frag_scored(bw, meas_ctx, trial_data)

    chol_a, chol_b = meas_ctx.chol_bar_a, meas_ctx.chol_bar_b
    assert chol_a is not None and chol_b is not None
    nchol = chol_a.shape[0]
    nchol_chunk = meas_ctx.nchol_chunk
    t2_r = (
        trial_data.t2aa.astype(rtype),
        trial_data.t2ab.astype(rtype),
        trial_data.t2ba.astype(rtype),
        trial_data.t2bb.astype(rtype),
    )

    # ---- pass 1: e2_0, exact, every vector, no T2 anywhere ----
    # fed the half rotated chol_s[:, :nocc_s, :]: e2_0 only touches the occupied blocks
    def scan_e2_0(carry, x):
        gl_occ_a = jnp.einsum("ir,gqr->giq", bw.green_a, x[0], optimize="optimal")
        gl_occ_b = jnp.einsum("ir,gqr->giq", bw.green_b, x[1], optimize="optimal")
        e2_0_g, _ = _e2_0_g(gl_occ_a, gl_occ_b)
        return carry + jnp.sum(e2_0_g.astype(c128)), None

    n_chunk1, chunk1, npad1 = equal_chunks(nchol, nchol_chunk)
    e2_0, _ = lax.scan(
        scan_e2_0,
        jnp.zeros((), dtype=c128),
        (
            _pad_reshape(chol_a[:, :nocc_a, :], n_chunk1, chunk1, npad1),
            _pad_reshape(chol_b[:, :nocc_b, :], n_chunk1, chunk1, npad1),
        ),
    )

    # ---- head / tail split ----
    n_head, n_samples = resolve_chol_budget(
        nchol,
        cfg.n_chol_head,
        cfg.head_chol_ratio,
        cfg.n_chol_samples,
        cfg.chol_cost_ratio,
        cfg.head_sample_ratio,
    )

    head_prefix: int | None = None
    head_idx = None
    if n_head >= nchol:
        # deterministic limit: every vector summed exactly, empty tail, no draw
        head_prefix = nchol
        tail = jnp.zeros((0,), dtype=jnp.int32)
        tail_prob = jnp.zeros((0,))
    else:
        pi_g = chol_sampling_proposal(
            e2frg_g, score_floor=cfg.chol_score_floor, uniform_mix=cfg.chol_uniform_mix
        )
        if cfg.head_from_guide:
            # per-walker ranking, at the cost of a batched gather for the head
            order = jnp.argsort(-pi_g)
            head_idx = jnp.sort(order[:n_head])
            tail = jnp.sort(order[n_head:])
        else:
            # a contiguous prefix is a plain slice, shared across the vmap batch
            head_prefix = n_head
            tail = jnp.arange(n_head, nchol, dtype=jnp.int32)
        tail_prob = pi_g[tail]
        tail_prob = tail_prob / jnp.sum(tail_prob)

    # ---- pass 2: only the accumulators that contract with T2 ----
    def accum(carry, chol_a_c, chol_b_c, w_c):
        _, e2_2_2_1_g, e2_2_2_2_g, e2_2_3_g = _bar_chunk_terms(
            chol_a_c, chol_b_c, bw, trial_data, t2_r, rtype, ctype
        )
        w_c = w_c.astype(ctype)
        carry[0] += jnp.sum(w_c * e2_2_2_1_g).astype(c128)
        carry[1] += jnp.sum(w_c * e2_2_2_2_g).astype(c128)
        carry[2] += jnp.sum(w_c * e2_2_3_g).astype(c128)
        return carry

    zero = jnp.zeros((), dtype=c128)

    def run_slice(chol_a_s, chol_b_s, weights):
        """Scan over the cholesky vectors themselves; only for the contiguous head."""
        n = weights.shape[0]
        if n == 0:
            return zero, zero, zero
        n_ch, chunk, npad = equal_chunks(n, nchol_chunk)
        if npad:
            weights = jnp.pad(weights, (0, npad))
        out, _ = lax.scan(
            lambda carry, x: (accum(carry, x[0], x[1], x[2]), None),
            [zero, zero, zero],
            (
                _pad_reshape(chol_a_s, n_ch, chunk, npad),
                _pad_reshape(chol_b_s, n_ch, chunk, npad),
                weights.reshape(n_ch, chunk),
            ),
        )
        return out[0], out[1], out[2]

    def run_indices(idx, weights):
        """Same sum over cholesky *indices*, gathering both spins inside the scan body."""
        n = weights.shape[0]
        if n == 0:
            return zero, zero, zero
        n_ch, chunk, npad = equal_chunks(n, nchol_chunk)
        if npad:
            # pad with index 0 at zero weight, which contributes nothing
            idx = jnp.pad(idx, (0, npad))
            weights = jnp.pad(weights, (0, npad))
        out, _ = lax.scan(
            lambda carry, x: (accum(carry, chol_a[x[0]], chol_b[x[0]], x[1]), None),
            [zero, zero, zero],
            (idx.reshape(n_ch, chunk), weights.reshape(n_ch, chunk)),
        )
        return out[0], out[1], out[2]

    # head: exact, unit weights
    if head_prefix is not None:
        b_h, c_h, d_h = run_slice(
            chol_a[:head_prefix], chol_b[:head_prefix], jnp.ones(head_prefix, dtype=c128)
        )
    else:
        assert head_idx is not None
        b_h, c_h, d_h = run_indices(head_idx, jnp.ones(head_idx.shape[0], dtype=c128))

    # tail: sampled, so walker dependent and therefore index scanned
    if tail.shape[0] == 0:
        b_t = c_t = d_t = zero
    else:
        if key is None:
            raise ValueError(
                "energy_kernel_uw_uh_sto draws a sampled tail and so needs a PRNG key; "
                "only n_chol_head='full' can run without one."
            )
        sel = jax.random.choice(key, tail.shape[0], shape=(n_samples,), replace=True, p=tail_prob)
        samp_w = (1.0 / (n_samples * tail_prob[sel])).astype(c128)
        b_t, c_t, d_t = run_indices(tail[sel], samp_w)

    # e2_2_1 = e2_0 * gt2g is exact, since e2_0 is
    e2_2 = e2_0 * bw.gt2g + (b_h + b_t) + (c_h + c_t) + (d_h + d_t)

    e0 = bw.e1_0 + e2_0  # fully exact
    e1frg = bw.e1_2 + e2_2

    return jnp.stack([bw.gt2g, e0frg, e1frg, e0])


def plan_chunking_for_run_u(
    sys: Any,
    ham_data: HamCholU,
    trial_data: Upt2ccsdTrial,
    *,
    measure_type: str | None = "bar",
    **kwargs: Any,
) -> ChunkPlan:
    """
    The MixedRecipe memory hook, with trot's unrestricted bar memory model. The fragment
    ctx adds the fock matrices and a constant, and the trial the t2ba block (the size of
    t2ab); both are small against the cholesky copies the model already counts.
    """
    return _trot_plan_chunking_for_run_u(sys, ham_data, trial_data, measure_type="bar", **kwargs)


def make_upt2ccsd_meas_ops(
    sys: Any,
    measure_type: str = "bar",
    memory_mode: str = "low",
    mixed_precision: bool = False,
    testing: bool = False,
    nchol_chunk: int | None = None,
    **cfg_fields: Any,
) -> MeasOps:
    """
    MeasOps of the unrestricted fragment pt2CCSD trial. measure_type "bar" is the (only)
    deterministic fragment estimator; "sto_chol" samples the T2-contracted two-body sum of
    e1frg. The same signature as trot's make_upt2ccsd_meas_ops, so setup_mixed drives it
    unchanged; the semistochastic knobs are Pt2ccsdMeasCfg fields given by name.
    """
    if sys.walker_kind.lower() != "unrestricted":
        raise ValueError(
            "the unrestricted fragment pt2CCSD MeasOps require walker_kind='unrestricted', "
            f"got: {sys.walker_kind}"
        )
    if measure_type not in _MEASURE_TYPES_U:
        raise ValueError(
            f"unknown measure_type {measure_type!r}; the LNO estimator has only {_MEASURE_TYPES_U}"
        )
    if memory_mode not in _MEMORY_MODES:
        raise ValueError(f"unknown memory_mode {memory_mode!r}; expected one of {_MEMORY_MODES}")

    owned = {"measure_type", "memory_mode", "nchol_chunk"} | {
        f.name for f in fields(Pt2ccsdMeasCfg) if f.name.startswith("mixed_")
    }
    valid = {f.name for f in fields(Pt2ccsdMeasCfg)} - owned
    unknown = sorted(set(cfg_fields) - valid)
    if unknown:
        raise ValueError(
            f"unknown or non-overridable Pt2ccsdMeasCfg field(s) {unknown}; settable here: {sorted(valid)}"
        )

    cfg = Pt2ccsdMeasCfg(
        measure_type=measure_type,
        memory_mode=memory_mode,
        mixed_real_dtype=jnp.float32 if mixed_precision else jnp.float64,
        mixed_complex_dtype=jnp.complex64 if mixed_precision else jnp.complex128,
        mixed_real_dtype_testing=jnp.float64 if testing else jnp.float32,
        mixed_complex_dtype_testing=jnp.complex128 if testing else jnp.complex64,
        nchol_chunk=nchol_chunk,
        **cfg_fields,
    )

    # a full head never draws, so the block function must not advance the key stream for it
    samples_tail = measure_type == "sto_chol" and not (
        isinstance(cfg.n_chol_head, str) and cfg.n_chol_head.lower() == "full"
    )
    energy_kernel = {"bar": energy_kernel_uw_uh_bar, "sto_chol": energy_kernel_uw_uh_sto}[measure_type]

    meas_ops = MeasOps(
        overlap=overlap_u,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={k_energy: energy_kernel},
        stochastic_kernels=frozenset({k_energy} if samples_tail else ()),
    )
    # the same attribute as trot's ops, so AfqmcMixed.dump_flags prints the cfg unchanged
    object.__setattr__(meas_ops, _PT2CCSD_MEAS_CFG_ATTR, cfg)
    return meas_ops
