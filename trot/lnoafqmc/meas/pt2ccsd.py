from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ...cholesky import equal_chunks
from ...core.ops import MeasOps, k_energy
from ...core.system import System
from ...ham.chol import HamChol
from ...meas.pt2ccsd import (
    _MEMORY_MODES,
    _PT2CCSD_MEAS_CFG_ATTR,
    DEFAULT_NCHOL_CHUNK,
    ChunkPlan,
    Pt2ccsdMeasCfg,
    build_bar_intermediates,
    chol_sampling_proposal,
    get_pt2ccsd_meas_cfg,
    plan_pt2ccsd_chunking,
    pt2ccsd_memory_model,
    resolve_chol_budget,
)
from ..trial.pt2ccsd import Pt2ccsdTrial, overlap_r

__all__ = [
    "TRIAL_COMPONENTS",
    "Pt2ccsdMeasCfg",
    "Pt2ccsdMeasCtx",
    "build_meas_ctx",
    "energy_kernel_rw_rh_bar",
    "energy_kernel_rw_rh_sto",
    "make_pt2ccsd_meas_ops",
    "get_pt2ccsd_meas_cfg",
    "plan_chunking_for_run",
]

# The LNO fragment estimator, mirroring trot/meas/pt2ccsd.py and ported from afqmc's
# lno_afqmc/wavefunctions_restricted.py (class pt2ccsd: _calc_e0bar_frag, _t2eorb_tc,
# _build_measurement_intermediates).
#
# It is the bar estimator, and only that: the fragment energy is measured with the
# similarity transformed hamiltonian H_bar = exp(T1) H exp(-T1) against the bare
# reference, walker_bar = exp(T1) walker. The kernel returns four numbers per walker,
#
#     t2frg = <HF| P T2 |walker_bar> / <HF|walker_bar>
#     e0frg = correlation part of <HF| H_bar |walker_bar> / <HF|walker_bar>, projected
#     e1frg = <HF| P T2 H_bar |walker_bar> / <HF|walker_bar>
#     e0    = <HF| H_bar |walker_bar> / <HF|walker_bar>, electronic, unprojected
#
# and the fragment energy is E_F = <e0frg> + <e1frg> - <t2frg><e0> over the blocks
# (lnoafqmc.stat_utils). There is no h0.
#
# Differences from trot's energy_kernel_rw_rh_bar, which this otherwise follows line by
# line: the projected t2 is not symmetric under (ia)<->(jb), so both halves of every
# T2-contracted one-body intermediate are formed instead of doubling one; and e0frg
# needs two more intermediates in the ctx, the transformed fock matrix (projected ov
# block) and the constant e0t1orb = <exp(T1)HF| H |HF>_F.
#
# measure_type "sto_chol" is that estimator with the T2-contracted two-body sum sampled
# over the cholesky index (afqmc's pt2ccsd_sto_chol, trot's energy_kernel_rw_rh_sto):
# t2frg, e0frg and e0 stay exact, and the three T2 accumulators of e1frg are split into an
# exactly summed head and an importance sampled tail. Unlike trot's full-space kernel,
# which scores a vector by its share of e2_0, the fragment kernel scores it by its share
# of the fragment two-body energy in e0frg, which is computed on the same walker anyway
# (afqmc's _calc_e0bar_frag_scored). The score only shapes the proposal, so it moves
# variance, never the mean.
#
# The kernel is a trial-side quantity only. afqmc's _calc_ept2_frag also computed the
# guide energy and the trial/guide overlap ratio (t1 = obar/o0) with the guide fixed to
# HF; here those come from the guide's own ops in blocks.block_frag, whatever the guide.

TRIAL_COMPONENTS = ("t2frg", "e0frg", "e1frg", "e0")

_MEASURE_TYPES = ("bar", "sto_chol")


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Pt2ccsdMeasCtx:
    cfg: Pt2ccsdMeasCfg  # static
    nchol_chunk: int = 1  # static; sets the shape the chol tensor is reshaped to

    exp_t1: jax.Array | None = None  # (norb, norb)
    h1_bar: jax.Array | None = None  # (norb, norb)
    chol_bar: jax.Array | None = None  # (nchol, norb, norb)
    fock_bar: jax.Array | None = None  # (norb, norb), fock of H_bar at the reference
    e0t1orb: jax.Array | None = None  # scalar, <exp(T1)HF|H|HF> projected on the fragment

    def tree_flatten(self):
        children = (self.exp_t1, self.h1_bar, self.chol_bar, self.fock_bar, self.e0t1orb)
        aux = (self.cfg, self.nchol_chunk)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        cfg, nchol_chunk = aux
        exp_t1, h1_bar, chol_bar, fock_bar, e0t1orb = children
        return cls(
            cfg=cfg,
            nchol_chunk=nchol_chunk,
            exp_t1=exp_t1,
            h1_bar=h1_bar,
            chol_bar=chol_bar,
            fock_bar=fock_bar,
            e0t1orb=e0t1orb,
        )


def fock_from_chol(nocc: int, h1: jax.Array, chol: jax.Array) -> jax.Array:
    """
    Closed-shell fock matrix of (h1, chol) at the reference occupying the first nocc
    orbitals: h1 + 2 J - K. Written for a non-symmetric chol (the transformed tensor),
    as afqmc's integral.get_rfock.
    """
    jeff = jnp.einsum("gpq,gjj->pq", chol, chol[:, :nocc, :nocc], optimize="optimal")
    keff = jnp.einsum("gpj,gjq->pq", chol[:, :, :nocc], chol[:, :nocc, :], optimize="optimal")
    return h1 + 2 * jeff - keff


def e0t1orb_from_chol(chol: jax.Array, t1: jax.Array, prjlo: jax.Array) -> jax.Array:
    """
    <exp(T1)HF| H |HF> restricted to the fragment: the T1-contracted two-body term,
    2 sum_g (L t1)_ik P_ik tr(L t1)_g - sum_g (L t1)_ij (L t1)_jk P_ik, from the
    untransformed cholesky vectors. afqmc's ham_data['e0t1orb'].
    """
    nocc = t1.shape[0]
    lt1 = jnp.einsum("ia,gja->gij", t1, chol[:, :nocc, nocc:], optimize="optimal")
    coul = 2 * jnp.einsum("gik,ik,gjj->", lt1, prjlo, lt1, optimize="optimal")
    exch = jnp.einsum("gij,gjk,ik->", lt1, lt1, prjlo, optimize="optimal")
    return coul - exch


def build_meas_ctx(
    ham_data: HamChol, trial_data: Pt2ccsdTrial, cfg: Pt2ccsdMeasCfg = Pt2ccsdMeasCfg(measure_type="bar")
) -> Pt2ccsdMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("the fragment pt2CCSD MeasOps assume HamChol.basis == 'restricted'.")
    if cfg.measure_type not in _MEASURE_TYPES:
        raise ValueError(
            f"unknown measure_type {cfg.measure_type!r}; the LNO estimator has only {_MEASURE_TYPES}"
        )
    if cfg.memory_mode not in _MEMORY_MODES:
        raise ValueError(f"unknown memory_mode {cfg.memory_mode!r}; expected one of {_MEMORY_MODES}")

    nchol = ham_data.nchol
    nchol = int(nchol) if nchol is not None else int(ham_data.chol.shape[0])
    requested = DEFAULT_NCHOL_CHUNK if cfg.nchol_chunk is None else int(cfg.nchol_chunk)
    if requested < 1:
        raise ValueError(f"nchol_chunk must be >= 1, got {cfg.nchol_chunk}")
    cap = min(requested, nchol) if nchol > 0 else requested
    _, nchol_chunk, _ = equal_chunks(nchol, cap)

    bar = build_bar_intermediates(ham_data, trial_data)
    nocc = trial_data.nocc
    fock_bar = fock_from_chol(nocc, bar["h1_bar"], bar["chol_bar"])
    e0t1orb = e0t1orb_from_chol(ham_data.chol, trial_data.t1, trial_data.prjlo)

    return Pt2ccsdMeasCtx(
        cfg=cfg,
        nchol_chunk=nchol_chunk,
        exp_t1=bar["exp_t1"],
        h1_bar=bar["h1_bar"],
        chol_bar=bar["chol_bar"],
        fock_bar=fock_bar,
        e0t1orb=e0t1orb,
    )


def _e0bar_frag(
    green: jax.Array, prjlo: jax.Array, fock_bar: jax.Array, chol_bar: jax.Array, e0t1orb: jax.Array, nchol_chunk: int
) -> jax.Array:
    """
    The projected correlation part of <HF| H_bar |walker_bar> / <HF|walker_bar>:
    e0t1orb + the projected fock ov term + the projected ov-ov two-body term. afqmc's
    _calc_e0bar_frag. green is the (nocc, norb) half green of walker_bar.
    """
    nocc = green.shape[0]
    gf_ov = green[:, nocc:]  # (nocc, nvir)
    fock_ov = fock_bar[:nocc, nocc:]
    e1 = 2 * jnp.einsum("ia,ik,ka->", gf_ov, prjlo, fock_ov, optimize="optimal")

    chol_ov = chol_bar[:, :nocc, nocc:]
    nchunks, k, pad = equal_chunks(chol_ov.shape[0], nchol_chunk)
    chol_ov = jnp.pad(chol_ov, ((0, pad), (0, 0), (0, 0)))
    chol_ov = chol_ov.reshape(nchunks, k, nocc, -1)

    def scanned_fun(carry, chol_c):
        lg = jnp.einsum("gia,ka->gik", chol_c, gf_ov, optimize="optimal")
        e2_c = 2 * jnp.einsum("gik,ik,gjj->", lg, prjlo, lg, optimize="optimal")
        e2_e = jnp.einsum("gij,gjk,ik->", lg, lg, prjlo, optimize="optimal")
        return carry + e2_c - e2_e, 0.0

    e2, _ = lax.scan(scanned_fun, jnp.zeros((), dtype=gf_ov.dtype), chol_ov)
    return e0t1orb + e1 + e2


def _e0bar_frag_scored(
    green: jax.Array, prjlo: jax.Array, fock_bar: jax.Array, chol_bar: jax.Array, e0t1orb: jax.Array, nchol_chunk: int
) -> tuple[jax.Array, jax.Array]:
    """
    _e0bar_frag, also returning the two-body term per cholesky vector, (nchol,): the
    scores the semistochastic kernel builds its proposal from. afqmc's
    _calc_e0bar_frag_scored.
    """
    nocc = green.shape[0]
    gf_ov = green[:, nocc:]
    fock_ov = fock_bar[:nocc, nocc:]
    e1 = 2 * jnp.einsum("ia,ik,ka->", gf_ov, prjlo, fock_ov, optimize="optimal")

    chol_ov = chol_bar[:, :nocc, nocc:]
    nchol = chol_ov.shape[0]
    nchunks, k, pad = equal_chunks(nchol, nchol_chunk)
    chol_ov = jnp.pad(chol_ov, ((0, pad), (0, 0), (0, 0)))
    chol_ov = chol_ov.reshape(nchunks, k, nocc, -1)

    def scanned_fun(carry, chol_c):
        lg = jnp.einsum("gia,ka->gik", chol_c, gf_ov, optimize="optimal")
        p_g = jnp.einsum("gik,ik->g", lg, prjlo, optimize="optimal")
        t_g = jnp.einsum("gjj->g", lg, optimize="optimal")
        x_g = jnp.einsum("gij,gjk,ik->g", lg, lg, prjlo, optimize="optimal")
        e2_g = 2 * p_g * t_g - x_g
        return carry + jnp.sum(e2_g), e2_g

    e2, e2_chunks = lax.scan(scanned_fun, jnp.zeros((), dtype=gf_ov.dtype), chol_ov)
    return e0t1orb + e1 + e2, e2_chunks.reshape(-1)[:nchol]


def energy_kernel_rw_rh_bar(
    walker: jax.Array, ham_data: HamChol, meas_ctx: Pt2ccsdMeasCtx, trial_data: Pt2ccsdTrial
) -> jax.Array:
    """
    The fragment pt2CCSD estimator for one restricted walker, [t2frg, e0frg, e1frg, e0].

    Follows trot's energy_kernel_rw_rh_bar: the similarity transformed hamiltonian in
    meas_ctx, the walker exp_t1 @ walker, and the (nocc, norb) half green against the
    bare reference. ham_data is only read for its basis; the tensors are in meas_ctx.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype

    t2 = trial_data.t2
    prjlo = trial_data.prjlo
    nocc, nvir, norb = trial_data.nocc, trial_data.nvir, trial_data.norb

    h1 = meas_ctx.h1_bar
    chol = meas_ctx.chol_bar
    if h1 is None or chol is None or meas_ctx.exp_t1 is None or meas_ctx.fock_bar is None:
        raise ValueError("the fragment energy kernel needs the bar intermediates; build the ctx first.")

    walker_bar = meas_ctx.exp_t1 @ walker  # (norb, nocc)

    # half green, (nocc, norb): the trial is the bare reference, so the full green is
    # this padded with zero rows
    green = (walker_bar @ jnp.linalg.inv(walker_bar[:nocc, :])).T
    green_occ = green[:, nocc:]  # (nocc, nvir)
    # (full green - 1) restricted to the virtual columns, (norb, nvir)
    greenp = jnp.vstack((green_occ, -jnp.eye(nvir, dtype=green.dtype)))
    rot_chol = chol[:, :nocc, :]  # (nchol, nocc, norb)

    # the projected correlation energy at the walker
    e0frg = _e0bar_frag(green, prjlo, meas_ctx.fock_bar, chol, meas_ctx.e0t1orb, meas_ctx.nchol_chunk)

    # one body energy; only the occupied rows of h1 meet a nonzero row of the green
    hg = jnp.einsum("pi,pi->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2 * hg

    # one-body double excitations. t_iajb != t_jbia since the first index is projected,
    # so both halves of each term are formed
    t2g_c_1 = jnp.einsum("iajb,ia->jb", t2, green_occ, optimize="optimal")
    t2g_c_2 = jnp.einsum("iajb,jb->ia", t2, green_occ, optimize="optimal")
    t2g_e_1 = jnp.einsum("iajb,ib->ja", t2, green_occ, optimize="optimal")
    t2g_e_2 = jnp.einsum("iajb,ja->ib", t2, green_occ, optimize="optimal")
    t2_green_c_1 = jnp.einsum("pb,jb,jq->pq", greenp, t2g_c_1, green, optimize="optimal")
    t2_green_c_2 = jnp.einsum("pa,ia,iq->pq", greenp, t2g_c_2, green, optimize="optimal")
    t2_green_e_1 = jnp.einsum("pa,ja,jq->pq", greenp, t2g_e_1, green, optimize="optimal")
    t2_green_e_2 = jnp.einsum("pb,ib,iq->pq", greenp, t2g_e_2, green, optimize="optimal")
    t2_green = (t2_green_c_1 + t2_green_c_2) - 0.5 * (t2_green_e_1 + t2_green_e_2)
    t2g = (t2g_c_1 + t2g_c_2) - 0.5 * (t2g_e_1 + t2g_e_2)
    gt2g = jnp.einsum("ia,ia->", t2g, green_occ, optimize="optimal")
    e1_2_1 = 2 * hg * gt2g
    e1_2_2 = -2 * jnp.einsum("pq,pq->", h1, t2_green, optimize="optimal")
    e1_2 = e1_2_1 + e1_2_2

    # two body energy, chunked over the cholesky index; the zero padding contributes to
    # neither tensor
    nchunks, nchol_chunk, pad = equal_chunks(chol.shape[0], meas_ctx.nchol_chunk)
    chol = jnp.pad(chol, ((0, pad), (0, 0), (0, 0)))
    rot_chol = jnp.pad(rot_chol, ((0, pad), (0, 0), (0, 0)))
    chol = chol.reshape(nchunks, nchol_chunk, norb, norb)
    rot_chol = rot_chol.reshape(nchunks, nchol_chunk, nocc, norb)

    t2_r = t2.astype(rtype)

    def scanned_fun(carry, x):
        chol_c, rot_chol_c = x  # (k, norb, norb), (k, nocc, norb)

        # e2_0
        gl = jnp.einsum("ir,gqr->giq", green, chol_c, optimize="optimal")  # (k, nocc, norb)
        tr_gl = jnp.einsum("gii->g", gl[:, :, :nocc], optimize="optimal")
        e2_0_c = 2 * jnp.einsum("g,g->", tr_gl, tr_gl, optimize="optimal")
        e2_0_e = -jnp.einsum("gij,gji->", gl[:, :, :nocc], gl[:, :, :nocc], optimize="optimal")
        carry[0] += e2_0_c + e2_0_e

        # e2_2_2_1
        lt2g = jnp.einsum("gpr,pr->g", chol_c.astype(rtype), t2_green.astype(ctype), optimize="optimal")
        carry[1] += -jnp.einsum("g,g->", lt2g.astype(ctype), tr_gl.astype(ctype), optimize="optimal").astype(
            jnp.complex128
        )

        # e2_2_2_2
        lt2_green = jnp.einsum(
            "gir,qr->giq", rot_chol_c.astype(rtype), t2_green.astype(ctype), optimize="optimal"
        )
        carry[2] += 0.5 * jnp.einsum(
            "giq,giq->", gl.astype(ctype), lt2_green.astype(ctype), optimize="optimal"
        ).astype(jnp.complex128)

        # e2_2_3
        glgp = jnp.einsum("gir,rb->gib", gl.astype(ctype), greenp.astype(ctype), optimize="optimal")
        lt2_c = jnp.einsum("gia,iajb->gjb", glgp, t2_r, optimize="optimal")
        lt2_e = jnp.einsum("gib,iajb->gja", glgp, t2_r, optimize="optimal")
        l2t2_c = jnp.einsum("gjb,gjb->", lt2_c.astype(ctype), glgp, optimize="optimal").astype(jnp.complex128)
        l2t2_e = jnp.einsum("gja,gja->", lt2_e.astype(ctype), glgp, optimize="optimal").astype(jnp.complex128)
        carry[3] += (2 * l2t2_c - l2t2_e).astype(jnp.complex128)

        return carry, 0.0

    zero = jnp.zeros((), dtype=jnp.complex128)
    [e2_0, e2_2_2_1, e2_2_2_2, e2_2_3], _ = lax.scan(scanned_fun, [zero, zero, zero, zero], (chol, rot_chol))

    e2_2_1 = e2_0 * gt2g
    e2_2_2 = 4 * (e2_2_2_1 + e2_2_2_2)
    e2_2 = e2_2_1 + e2_2_2 + e2_2_3

    t2frg = gt2g  # <HF| P T2 |walker_bar> / <HF|walker_bar>
    e0 = e1_0 + e2_0  # <HF| H_bar |walker_bar> / <HF|walker_bar>
    e1frg = e1_2 + e2_2  # <HF| P T2 H_bar |walker_bar> / <HF|walker_bar>

    return jnp.stack([t2frg, e0frg, e1frg, e0])


def energy_kernel_rw_rh_sto(
    walker: jax.Array,
    ham_data: HamChol,
    meas_ctx: Pt2ccsdMeasCtx,
    trial_data: Pt2ccsdTrial,
    key: jax.Array | None = None,
) -> jax.Array:
    """
    The fragment estimator with a semistochastic cholesky sum in e1frg,
    [t2frg, e0frg, e1frg, e0].

    Follows trot's energy_kernel_rw_rh_sto: e2_0, and with it e2_2_1 = e2_0 * gt2g, is
    summed exactly over every vector (pass 1, no T2), and the accumulators that contract
    with T2 -- e2_2_2_1, e2_2_2_2, e2_2_3 -- are summed exactly over a head and importance
    sampled over the tail (pass 2). The head is a contiguous prefix unless
    cfg.head_from_guide, and the tail is scanned over indices with the gather inside the
    scan body, for the memory reasons given there. The proposal comes from the fragment
    two-body energies of e0frg (see the module comment).

    n_chol_head="full" removes the sampling and reproduces energy_kernel_rw_rh_bar; no key
    is drawn or needed in that limit.
    """
    cfg = meas_ctx.cfg
    rtype = cfg.mixed_real_dtype
    ctype = cfg.mixed_complex_dtype
    c128 = jnp.complex128

    t2 = trial_data.t2
    prjlo = trial_data.prjlo
    nocc, nvir, norb = trial_data.nocc, trial_data.nvir, trial_data.norb

    h1 = meas_ctx.h1_bar
    chol = meas_ctx.chol_bar
    if h1 is None or chol is None or meas_ctx.exp_t1 is None or meas_ctx.fock_bar is None:
        raise ValueError("the fragment energy kernel needs the bar intermediates; build the ctx first.")

    nchol = chol.shape[0]
    nchol_chunk = meas_ctx.nchol_chunk
    walker_bar = meas_ctx.exp_t1 @ walker

    green = (walker_bar @ jnp.linalg.inv(walker_bar[:nocc, :])).T  # (nocc, norb)
    green_occ = green[:, nocc:]
    greenp = jnp.vstack((green_occ, -jnp.eye(nvir, dtype=green.dtype)))

    # the projected correlation energy at the walker, exact, and the per-vector scores
    e0frg, e2frg_g = _e0bar_frag_scored(
        green, prjlo, meas_ctx.fock_bar, chol, meas_ctx.e0t1orb, nchol_chunk
    )

    # one body, exactly as in the bar kernel
    hg = jnp.einsum("pi,pi->", h1[:nocc, :], green, optimize="optimal")
    e1_0 = 2 * hg
    t2g_c_1 = jnp.einsum("iajb,ia->jb", t2, green_occ, optimize="optimal")
    t2g_c_2 = jnp.einsum("iajb,jb->ia", t2, green_occ, optimize="optimal")
    t2g_e_1 = jnp.einsum("iajb,ib->ja", t2, green_occ, optimize="optimal")
    t2g_e_2 = jnp.einsum("iajb,ja->ib", t2, green_occ, optimize="optimal")
    t2g = (t2g_c_1 + t2g_c_2) - 0.5 * (t2g_e_1 + t2g_e_2)
    t2_green = jnp.einsum("pb,jb,jq->pq", greenp, t2g, green, optimize="optimal")
    gt2g = jnp.einsum("ia,ia->", t2g, green_occ, optimize="optimal")
    e1_2 = 2 * hg * gt2g - 2 * jnp.einsum("pq,pq->", h1, t2_green, optimize="optimal")

    # ---- pass 1: e2_0, exact, every vector, no T2 anywhere ----
    def scan_e2_0(carry, rot_c):
        gl_occ = jnp.einsum("ir,gqr->giq", green, rot_c, optimize="optimal")
        tr_gl = jnp.einsum("gii->g", gl_occ, optimize="optimal")
        e2_0_g = 2 * tr_gl * tr_gl - jnp.einsum("gij,gji->g", gl_occ, gl_occ, optimize="optimal")
        return carry + jnp.sum(e2_0_g.astype(c128)), None

    n_chunk1, chunk1, npad1 = equal_chunks(nchol, nchol_chunk)
    # fed the half rotated chol[:, :nocc, :]: e2_0 only touches the occupied block of gl
    rot_all = chol[:, :nocc, :]
    if npad1:
        rot_all = jnp.pad(rot_all, ((0, npad1), (0, 0), (0, 0)))
    e2_0, _ = lax.scan(scan_e2_0, jnp.zeros((), dtype=c128), rot_all.reshape(n_chunk1, chunk1, nocc, norb))

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
    t2_r = t2.astype(rtype)

    def accum(carry, chol_c, w_c):
        """The three T2-contracted accumulators for one chunk, weighted per vector."""
        rot_chol_c = chol_c[:, :nocc, :]
        w_c = w_c.astype(ctype)

        gl = jnp.einsum("ir,gqr->giq", green, chol_c, optimize="optimal")
        tr_gl = jnp.einsum("gii->g", gl[:, :, :nocc], optimize="optimal")

        # e2_2_2_1
        lt2g = jnp.einsum("gpr,pr->g", chol_c.astype(rtype), t2_green.astype(ctype), optimize="optimal")
        carry[0] += jnp.sum(w_c * (-lt2g.astype(ctype) * tr_gl.astype(ctype))).astype(c128)

        # e2_2_2_2
        lt2_green = jnp.einsum(
            "gir,qr->giq", rot_chol_c.astype(rtype), t2_green.astype(ctype), optimize="optimal"
        )
        carry[1] += jnp.sum(
            w_c * 0.5 * jnp.einsum("giq,giq->g", gl.astype(ctype), lt2_green.astype(ctype), optimize="optimal")
        ).astype(c128)

        # e2_2_3
        glgp = jnp.einsum("gir,rb->gib", gl.astype(ctype), greenp.astype(ctype), optimize="optimal")
        lt2_c = jnp.einsum("gia,iajb->gjb", glgp, t2_r, optimize="optimal")
        lt2_e = jnp.einsum("gib,iajb->gja", glgp, t2_r, optimize="optimal")
        l2t2_c = jnp.einsum("gjb,gjb->g", lt2_c.astype(ctype), glgp, optimize="optimal")
        l2t2_e = jnp.einsum("gja,gja->g", lt2_e.astype(ctype), glgp, optimize="optimal")
        carry[2] += jnp.sum(w_c * (2 * l2t2_c - l2t2_e)).astype(c128)
        return carry

    zero = jnp.zeros((), dtype=c128)

    def run_slice(chol_s, weights):
        """Scan over the cholesky vectors themselves; only for the contiguous head."""
        n = weights.shape[0]
        if n == 0:
            return zero, zero, zero
        n_ch, chunk, npad = equal_chunks(n, nchol_chunk)
        if npad:
            chol_s = jnp.pad(chol_s, ((0, npad), (0, 0), (0, 0)))
            weights = jnp.pad(weights, (0, npad))
        out, _ = lax.scan(
            lambda carry, x: (accum(carry, x[0], x[1]), 0.0),
            [zero, zero, zero],
            (chol_s.reshape(n_ch, chunk, norb, norb), weights.reshape(n_ch, chunk)),
        )
        return out[0], out[1], out[2]

    def run_indices(idx, weights):
        """Same sum over cholesky *indices*, gathering inside the scan body."""
        n = weights.shape[0]
        if n == 0:
            return zero, zero, zero
        n_ch, chunk, npad = equal_chunks(n, nchol_chunk)
        if npad:
            # pad with index 0 at zero weight, which contributes nothing
            idx = jnp.pad(idx, (0, npad))
            weights = jnp.pad(weights, (0, npad))
        out, _ = lax.scan(
            lambda carry, x: (accum(carry, chol[x[0]], x[1]), 0.0),
            [zero, zero, zero],
            (idx.reshape(n_ch, chunk), weights.reshape(n_ch, chunk)),
        )
        return out[0], out[1], out[2]

    # head: exact, unit weights
    if head_prefix is not None:
        b_h, c_h, d_h = run_slice(chol[:head_prefix], jnp.ones(head_prefix, dtype=c128))
    else:
        assert head_idx is not None
        b_h, c_h, d_h = run_indices(head_idx, jnp.ones(head_idx.shape[0], dtype=c128))

    # tail: sampled, so walker dependent and therefore index scanned
    if tail.shape[0] == 0:
        b_t = c_t = d_t = zero
    else:
        if key is None:
            raise ValueError(
                "energy_kernel_rw_rh_sto draws a sampled tail and so needs a PRNG key; "
                "only n_chol_head='full' can run without one."
            )
        sel = jax.random.choice(key, tail.shape[0], shape=(n_samples,), replace=True, p=tail_prob)
        samp_w = (1.0 / (n_samples * tail_prob[sel])).astype(c128)
        b_t, c_t, d_t = run_indices(tail[sel], samp_w)

    # e2_2_1 = e2_0 * gt2g is exact, since e2_0 is
    e2_2 = e2_0 * gt2g + 4 * (b_h + c_h + b_t + c_t) + d_h + d_t

    t2frg = gt2g
    e0 = e1_0 + e2_0  # fully exact
    e1frg = e1_2 + e2_2

    return jnp.stack([t2frg, e0frg, e1frg, e0])


def plan_chunking_for_run(
    sys: System,
    ham_data: HamChol,
    trial_data: Pt2ccsdTrial,
    *,
    n_walkers: int,
    max_memory_mb: float,
    n_chunks: int = 1,
    nchol_chunk: int | None = None,
    mixed_precision: bool = False,
    n_devices: int = 1,
    measure_type: str | None = "bar",
) -> ChunkPlan:
    """
    The MixedRecipe memory hook, with trot's bar memory model. The fragment ctx adds the
    fock matrix and the constant, both negligible against the cholesky copies the model
    already counts, and the e0frg scan reuses the (k, nocc, ...) shapes it sizes.
    """
    nchol = ham_data.nchol
    nchol = int(nchol) if nchol is not None else int(ham_data.chol.shape[0])
    model = pt2ccsd_memory_model(
        norb=trial_data.norb,
        nocc=trial_data.nocc,
        nchol=nchol,
        n_walkers=n_walkers,
        real_bytes=4 if mixed_precision else 8,
        complex_bytes=8 if mixed_precision else 16,
        bar=True,
    )
    return plan_pt2ccsd_chunking(
        model,
        n_walkers=n_walkers,
        nchol=nchol,
        budget_bytes=int(max_memory_mb * (1024**2)),
        n_chunks=n_chunks,
        nchol_chunk=nchol_chunk,
        n_devices=n_devices,
    )


def make_pt2ccsd_meas_ops(
    sys: System,
    measure_type: str = "bar",
    memory_mode: str = "low",
    mixed_precision: bool = False,
    testing: bool = False,
    nchol_chunk: int | None = None,
    **cfg_fields: Any,
) -> MeasOps:
    """
    MeasOps of the fragment pt2CCSD trial. measure_type "bar" is the (only) deterministic
    fragment estimator; "sto_chol" samples the T2-contracted two-body sum of e1frg. The
    same signature as trot's make_pt2ccsd_meas_ops, so setup_mixed drives it unchanged;
    the semistochastic knobs (n_chol_head, head_chol_ratio, n_chol_samples,
    chol_cost_ratio, ...) are Pt2ccsdMeasCfg fields given by name.
    """
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"the fragment pt2CCSD MeasOps support only restricted walkers, got: {sys.walker_kind}"
        )
    if measure_type not in _MEASURE_TYPES:
        raise ValueError(
            f"unknown measure_type {measure_type!r}; the LNO estimator has only {_MEASURE_TYPES}"
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
    energy_kernel = {"bar": energy_kernel_rw_rh_bar, "sto_chol": energy_kernel_rw_rh_sto}[measure_type]

    meas_ops = MeasOps(
        overlap=overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={k_energy: energy_kernel},
        stochastic_kernels=frozenset({k_energy} if samples_tail else ()),
    )
    # the same attribute as trot's ops, so AfqmcMixed.dump_flags prints the cfg unchanged
    object.__setattr__(meas_ops, _PT2CCSD_MEAS_CFG_ATTR, cfg)
    return meas_ops
