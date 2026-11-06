"""
Propagation under the stochastically sampled CCSD guide (trial/stoccsd.py).

Every walker carries its own sample of the guide, n_samples determinants
d_{w,s} = exp(y_{w,s} . tau)|psi~>, redrawn at every step. The importance function of a
step then has two factors, one from the change of the guide and one from the walk,

    <G_w^{(i+1)}|phi_w^{(i+1)}>   <G_w^{(i+1)}|phi_w^{(i)}>   <G_w^{(i+1)}|phi_w^{(i+1)}>
    --------------------------- = ------------------------- x ---------------------------
    <G_w^{(i)}|phi_w^{(i)}>       <G_w^{(i)}|phi_w^{(i)}>     <G_w^{(i+1)}|phi_w^{(i)}>
                                        R_G                          R_w

and the product over the steps telescopes, so dividing the final walker by its final
guide overlap still gives the propagated state exactly. The force bias is taken against
the new guide on the old walker. The weight update is

    w <- w  |exp(...)  R_w|  max(0, cos theta_w)  |R_G|,   theta_w = arg(exp(-sqrt(dt) x.m) R_w)

i.e. the phaseless projection acts on the walk's phase only; the guide's phase change is
dropped by taking |R_G|, the guide being sampled from a real distribution. (Projecting
cos(theta_w + theta_g) instead was found much worse: a redrawn guide is not an
infinitesimal change.)

The sample lives in the propagation state (PropStateStocc.guide_dets, one set per
walker) because the block function needs the determinants that produced state.overlaps
after orthonormalization and reconfiguration. The propagation fields come from the same
subkey as trot's afqmc_step and the guide fields from a fold_in of it, so with no T2
modes and one sample the walk is the RHF-guided one with the Thouless reference as guide.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps, k_energy, k_force_bias
from ..core.system import System
from ..ham.chol import HamBasis, HamChol
from ..trial.stoccsd import StoccsdGuide, sample_dets, with_dets
from ..walkers import init_walkers
from .chol_afqmc_ops import CholAfqmcCtx, TrotterOps, _build_prop_ctx, make_trotter_ops
from .types import PropOps, QmcParamsBase


class PropStateStocc(NamedTuple):
    """prop.types.PropState plus the guide sample of every walker."""

    walkers: Any
    weights: jax.Array
    overlaps: jax.Array
    rng_key: jax.Array
    pop_control_ene_shift: jax.Array
    e_estimate: jax.Array
    node_encounters: jax.Array
    guide_dets: jax.Array  # (n_walkers, n_samples, norb, nocc)


# ------------------------------------------------------------------------------------
# the per walker guide
# ------------------------------------------------------------------------------------


def sample_walker_dets(guide: StoccsdGuide, key: jax.Array, n_walkers: int) -> jax.Array:
    """An independent sample of guide.n_samples determinants for each of n_walkers walkers."""
    keys = jax.random.split(key, int(n_walkers))
    return jax.vmap(lambda k: sample_dets(guide.mo_t, guide.tau, k, guide.n_samples))(keys)


def guide_overlaps(
    meas_ops: MeasOps, walkers: Any, dets: jax.Array, guide: StoccsdGuide, *, n_chunks: int
) -> jax.Array:
    """<G_w|phi_w> for every walker with its own sample."""
    fn = lambda w, d, g: meas_ops.overlap(w, with_dets(g, d))
    return wk.vmap_chunked(fn, n_chunks=n_chunks, in_axes=(0, 0, None))(walkers, dets, guide)


def guide_kernel(
    meas_ops: MeasOps,
    name: str,
    walkers: Any,
    dets: jax.Array,
    ham_data: Any,
    meas_ctx: Any,
    guide: StoccsdGuide,
    *,
    n_chunks: int,
) -> jax.Array:
    """A guide kernel (force bias, energy) for every walker with its own sample."""
    kernel = meas_ops.require_kernel(name)
    fn = lambda w, d, h, c, g: kernel(w, h, c, with_dets(g, d))
    return wk.vmap_chunked(fn, n_chunks=n_chunks, in_axes=(0, 0, None, None, None))(
        walkers, dets, ham_data, meas_ctx, guide
    )


# ------------------------------------------------------------------------------------
# state and step
# ------------------------------------------------------------------------------------


def init_prop_state_stocc(
    *,
    sys: System,
    ham_data: HamChol,
    trial_ops: TrialOps,
    trial_data: StoccsdGuide,
    meas_ops: MeasOps,
    params: QmcParamsBase,
    meas_ctx: Any | None = None,
    initial_walkers: Any | None = None,
    initial_e_estimate: jax.Array | None = None,
    rdm1: jax.Array | None = None,
    mesh: Mesh | None = None,
) -> PropStateStocc:
    """
    prop.afqmc.init_prop_state for the sampled guide: the walkers from the guide's rdm1,
    a first sample per walker from the run seed, the overlaps with it, and the energy
    estimate from the guide's local energy of the first walker.
    """
    if mesh is not None and mesh.size > 1:
        raise NotImplementedError("the stoccsd guide does not shard the walkers over a mesh yet.")
    n_walkers = params.n_walkers
    key = jax.random.PRNGKey(int(params.seed))
    weights = jnp.ones((n_walkers,))

    if initial_walkers is None:
        if rdm1 is None:
            rdm1 = trial_ops.get_rdm1(trial_data)
        initial_walkers = init_walkers(sys=sys, rdm1=rdm1, n_walkers=n_walkers)

    dets = sample_walker_dets(trial_data, jax.random.fold_in(key, 2), n_walkers)
    overlaps = guide_overlaps(meas_ops, initial_walkers, dets, trial_data, n_chunks=params.n_chunks)

    if initial_e_estimate is not None:
        e_est = jnp.asarray(initial_e_estimate)
    else:
        if meas_ctx is None:
            meas_ctx = meas_ops.build_meas_ctx(ham_data, trial_data)
        walker_0 = wk.take_walkers(initial_walkers, jnp.array([0]))
        e_samples = jnp.real(
            guide_kernel(
                meas_ops,
                k_energy,
                walker_0,
                dets[:1],
                ham_data,
                meas_ctx,
                trial_data,
                n_chunks=1,
            )
        )
        e_est = jnp.mean(e_samples)
    e_est = jax.block_until_ready(e_est)

    return PropStateStocc(
        walkers=initial_walkers,
        weights=weights,
        overlaps=overlaps,
        rng_key=key,
        pop_control_ene_shift=e_est,
        e_estimate=e_est,
        node_encounters=jnp.asarray(0),
        guide_dets=dets,
    )


def afqmc_step_stocc(
    state: PropStateStocc,
    *,
    params: QmcParamsBase,
    ham_data: HamChol,
    trial_data: StoccsdGuide,
    meas_ops: MeasOps,
    trotter_ops: TrotterOps,
    prop_ctx: CholAfqmcCtx,
    meas_ctx: Any,
) -> PropStateStocc:
    key, subkey = jax.random.split(state.rng_key)
    nw = wk.n_walkers(state.walkers)
    fields = jax.random.normal(subkey, (nw, prop_ctx.chol_flat.shape[0]))

    # the new guide of every walker, G^{(i+1)}
    dets = sample_walker_dets(trial_data, jax.random.fold_in(subkey, 1), nw)
    # <G^{(i+1)}|phi^{(i)}> and the force bias of the new guide on the old walker
    overlaps_10 = guide_overlaps(
        meas_ops, state.walkers, dets, trial_data, n_chunks=params.n_chunks
    )
    force_bias = guide_kernel(
        meas_ops,
        k_force_bias,
        state.walkers,
        dets,
        ham_data,
        meas_ctx,
        trial_data,
        n_chunks=params.n_chunks,
    )
    field_shifts = -prop_ctx.sqrt_dt * (1.0j * force_bias - prop_ctx.mf_shifts)
    shifted_fields = fields - field_shifts

    shift_term = jnp.sum(shifted_fields * prop_ctx.mf_shifts, axis=1)
    fb_term = jnp.sum(fields * field_shifts - 0.5 * field_shifts * field_shifts, axis=1)
    walkers_new = wk.vmap_chunked(
        trotter_ops.apply_trotter, n_chunks=params.n_chunks, in_axes=(0, 0, None, None)
    )(state.walkers, shifted_fields, prop_ctx, params.n_exp_terms)

    # <G^{(i+1)}|phi^{(i+1)}>
    overlaps_11 = guide_overlaps(meas_ops, walkers_new, dets, trial_data, n_chunks=params.n_chunks)
    ratio_g = overlaps_10 / state.overlaps  # the change of the guide
    ratio_w = overlaps_11 / overlaps_10  # the walk
    exponent = (
        -prop_ctx.sqrt_dt * shift_term
        + fb_term
        + prop_ctx.dt * (state.pop_control_ene_shift + prop_ctx.h0_prop)
    )
    imp_fun = jnp.exp(exponent) * ratio_w * jnp.abs(ratio_g)

    theta = jnp.angle(jnp.exp(-prop_ctx.sqrt_dt * shift_term) * ratio_w)
    imp_ph = jnp.abs(imp_fun) * jnp.cos(theta)

    w_floor = float(getattr(params, "weight_floor", 1.0e-3))
    w_cap = float(getattr(params, "weight_cap", 100.0))

    imp_ph = jnp.where(~jnp.isfinite(imp_ph) | (imp_ph < w_floor), 0.0, imp_ph)
    node_encounters_new = state.node_encounters + jnp.sum(imp_ph <= 0.0)
    imp_ph = jnp.where(imp_ph > w_cap, 0.0, imp_ph)

    weights_new = state.weights * imp_ph
    weights_new = jnp.where(weights_new > w_cap, 0.0, weights_new)

    damping = float(getattr(params, "pop_control_damping", 0.1))
    avg_w = jnp.clip(jnp.mean(weights_new), min=1.0e-300)
    pop_shift_new = state.e_estimate - damping * (jnp.log(avg_w) / prop_ctx.dt)

    return PropStateStocc(
        walkers=walkers_new,
        weights=weights_new,
        overlaps=overlaps_11,
        rng_key=key,
        pop_control_ene_shift=pop_shift_new,
        e_estimate=state.e_estimate,
        node_encounters=node_encounters_new,
        guide_dets=dets,
    )


def make_stocc_prop_ops(
    ham_basis: HamBasis,
    walker_kind: str,
    mixed_precision: bool = False,
    packed_cholesky: bool = False,
) -> PropOps:
    """prop.afqmc.make_prop_ops with the sampled guide's state and step."""
    if walker_kind.lower() != "restricted":
        raise ValueError(f"the stoccsd guide propagates restricted walkers, got {walker_kind!r}")
    trotter_ops = make_trotter_ops(ham_basis, walker_kind, mixed_precision=mixed_precision)

    def step(
        state: PropStateStocc,
        *,
        params: QmcParamsBase,
        ham_data: Any,
        trial_data: Any,
        trial_ops: TrialOps,
        meas_ops: MeasOps,
        meas_ctx: Any,
        prop_ctx: Any,
    ) -> PropStateStocc:
        return afqmc_step_stocc(
            state,
            params=params,
            ham_data=ham_data,
            trial_data=trial_data,
            meas_ops=meas_ops,
            meas_ctx=meas_ctx,
            prop_ctx=prop_ctx,
            trotter_ops=trotter_ops,
        )

    def build_prop_ctx(ham_data: Any, rdm1: jax.Array, params: QmcParamsBase) -> CholAfqmcCtx:
        return _build_prop_ctx(
            ham_data,
            rdm1,
            params.dt,
            chol_flat_precision=jnp.float32 if mixed_precision else jnp.float64,
            packed_cholesky=packed_cholesky,
        )

    return PropOps(init_prop_state=init_prop_state_stocc, build_prop_ctx=build_prop_ctx, step=step)


__all__ = [
    "PropStateStocc",
    "afqmc_step_stocc",
    "guide_kernel",
    "guide_overlaps",
    "init_prop_state_stocc",
    "make_stocc_prop_ops",
    "sample_walker_dets",
]
