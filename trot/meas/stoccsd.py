"""
Measurements against the stochastically sampled CCSD guide (trial/stoccsd.py).

The guide is the N-determinant state |G> = 1/N sum_s |d_s>, so every quantity is the
single-determinant (RHF-like) one of each d_s, weighted by that determinant's overlap
with the walker:

    <G|phi>            = 1/N sum_s o_s,                      o_s = det(d_s^H phi)^2
    <G|L_g|phi>/<G|phi> = sum_s o_s fb_s[g] / sum_s o_s,     fb_s = 2 sum_pq L_g,pq G_s,pq
    <G|H|phi>/<G|phi>   = sum_s o_s E_s / sum_s o_s,         E_s = the RHF local energy
                                                                   of the bra d_s

with G_s = conj(d_s) g_s the one-body green's function of the pair (d_s, phi) and
g_s = (phi (d_s^H phi)^-1)^T its (nocc, norb) half. The force bias contracts the
unrotated cholesky vectors with the full (norb, norb) green's function of each sample
(no per-sample half rotation, since the sample changes every propagation step); the
local energy, measured once per block, half rotates h1 and the cholesky vectors by each
d_s and reuses meas/rhf.py's contractions.

mixed_precision runs the force bias contraction with the cholesky vectors in single
precision; the energy stays double.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax, tree_util

from ..core.ops import MeasOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamChol
from ..trial.stoccsd import StoccsdGuide, det_overlaps_r, overlap_r
from .rhf import _half_green_from_overlap_matrix, _restricted_chol_energy

_STOCCSD_MEAS_CFG_ATTR = "_stoccsd_meas_cfg"


@dataclass(frozen=True)
class StoccsdMeasCfg:
    # single precision cholesky vectors in the force bias contraction
    mixed_precision: bool = False


def get_stoccsd_meas_cfg(meas_ops: MeasOps) -> StoccsdMeasCfg | None:
    cfg = getattr(meas_ops, _STOCCSD_MEAS_CFG_ATTR, None)
    if isinstance(cfg, StoccsdMeasCfg):
        return cfg
    return None


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StoccsdMeasCtx:
    h1: jax.Array  # (norb, norb)
    chol: jax.Array  # (n_chol, norb, norb), double, for the energy
    chol_fb_flat: jax.Array  # (n_chol, norb*norb), in the force bias precision
    cfg: StoccsdMeasCfg = StoccsdMeasCfg()

    def tree_flatten(self):
        children = (self.h1, self.chol, self.chol_fb_flat)
        aux = (self.cfg,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (cfg,) = aux
        h1, chol, chol_fb_flat = children
        return cls(h1=h1, chol=chol, chol_fb_flat=chol_fb_flat, cfg=cfg)


def build_meas_ctx(
    ham_data: HamChol,
    trial_data: StoccsdGuide,
    cfg: StoccsdMeasCfg = StoccsdMeasCfg(),
) -> StoccsdMeasCtx:
    if ham_data.basis != "restricted":
        raise ValueError("stoccsd MeasOps assumes HamChol.basis == 'restricted'.")
    chol = ham_data.chol
    n_chol = chol.shape[0]
    fb_dtype = jnp.float32 if cfg.mixed_precision else chol.dtype
    chol_fb_flat = chol.reshape(n_chol, -1).astype(fb_dtype)
    return StoccsdMeasCtx(h1=ham_data.h1, chol=chol, chol_fb_flat=chol_fb_flat, cfg=cfg)


def _weighted_average(olps: jax.Array, values: jax.Array) -> jax.Array:
    """sum_s o_s v_s / sum_s o_s along the sample axis (the leading one)."""
    olp = jnp.sum(olps)
    return jnp.tensordot(olps, values, axes=(0, 0)) / olp


def force_bias_kernel_rw_rh(
    walker: jax.Array, ham_data: Any, meas_ctx: StoccsdMeasCtx, trial_data: StoccsdGuide
) -> jax.Array:
    norb = walker.shape[0]
    fb_dtype = jnp.result_type(meas_ctx.chol_fb_flat.dtype, jnp.complex64)

    def one(d: jax.Array) -> tuple[jax.Array, jax.Array]:
        m = d.conj().T @ walker  # (nocc, nocc)
        olp = jnp.linalg.det(m) ** 2
        g_half = _half_green_from_overlap_matrix(walker, m)  # (nocc, norb)
        g_full = jnp.conj(d) @ g_half  # (norb, norb)
        # RHF: factor 2 for (up+dn)
        fb = 2.0 * (meas_ctx.chol_fb_flat @ g_full.reshape(norb * norb).astype(fb_dtype))
        return olp, fb

    olps, fbs = jax.vmap(one)(trial_data.dets)
    return _weighted_average(olps, fbs.astype(jnp.complex128))


def energy_kernel_rw_rh(
    walker: jax.Array, ham_data: HamChol, meas_ctx: StoccsdMeasCtx, trial_data: StoccsdGuide
) -> jax.Array:
    e0 = ham_data.h0

    def one(_carry: None, d: jax.Array) -> tuple[None, tuple[jax.Array, jax.Array]]:
        dH = d.conj().T  # (nocc, norb)
        m = dH @ walker
        olp = jnp.linalg.det(m) ** 2
        g_half = _half_green_from_overlap_matrix(walker, m)  # (nocc, norb)
        rot_h1 = dH @ meas_ctx.h1  # (nocc, norb)
        rot_chol = jnp.einsum("pi,gpj->gij", jnp.conj(d), meas_ctx.chol, optimize="optimal")
        e1 = 2.0 * jnp.sum(g_half * rot_h1)
        e2 = _restricted_chol_energy(rot_chol, g_half)
        return None, (olp, e0 + e1 + e2)

    # one sample at a time: the rotated cholesky tensor of a sample is (n_chol, nocc, norb)
    _, (olps, energies) = lax.scan(one, None, trial_data.dets)
    return _weighted_average(olps, energies)


def make_stoccsd_meas_ops(sys: System, *, mixed_precision: bool = False) -> MeasOps:
    """Overlap, force bias and local energy against the sampled CCSD guide."""
    cfg = StoccsdMeasCfg(mixed_precision=bool(mixed_precision))
    wk = sys.walker_kind.lower()
    if wk != "restricted":
        raise ValueError(
            f"stoccsd measurements take restricted walkers, got walker_kind={sys.walker_kind!r}"
        )

    meas_ops = MeasOps(
        overlap=overlap_r,
        build_meas_ctx=lambda ham_data, trial_data: build_meas_ctx(ham_data, trial_data, cfg),
        kernels={
            k_force_bias: force_bias_kernel_rw_rh,
            k_energy: energy_kernel_rw_rh,
        },
    )
    object.__setattr__(meas_ops, _STOCCSD_MEAS_CFG_ATTR, cfg)
    return meas_ops


__all__ = [
    "StoccsdMeasCfg",
    "StoccsdMeasCtx",
    "build_meas_ctx",
    "det_overlaps_r",
    "energy_kernel_rw_rh",
    "force_bias_kernel_rw_rh",
    "get_stoccsd_meas_cfg",
    "make_stoccsd_meas_ops",
]
