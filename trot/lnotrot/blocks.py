from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps, k_energy
from ..core.system import System
from ..prop.blocks import BlockObs
from ..prop.types import PropOps, PropState, QmcParams
from .meas.pt2ccsd import TRIAL_COMPONENTS

# The block function of the fragment AFQMC, mirroring trot/prop/blocks.py:block_mixed
# with the numbers of afqmc's sampler_pt2.block_sample: propagate with the guide,
# measure the fragment estimator against the trial, average both over the walkers.
#
# It has the keyword signature of trot's MixedBlockFn, so trot.driver.make_run_mixed_blocks
# scans it unchanged. What differs from block_mixed is only what the trial kernel returns:
# TRIAL_COMPONENTS names per walker instead of (t2, e0, e1), each averaged with the
# absorbed weight
#
#     wp_i = w_i <T|phi_i> / <G|phi_i>
#
# which is afqmc's wp = wt * t1 with t1 = obar/o0 for an HF guide, now for any guide.


def block_frag(
    state: PropState,
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    guide_data: Any,
    guide_ops: TrialOps,
    guide_meas_ops: MeasOps,
    guide_meas_ctx: Any,
    guide_prop_ops: PropOps,
    guide_prop_ctx: Any,
    trial_data: Any,
    trial_meas_ops: MeasOps,
    trial_meas_ctx: Any,
    observable_names: tuple[str, ...] = (),
    sr_fn: Callable = wk.stochastic_reconfiguration,
    measure_trial: bool = True,
    components: tuple[str, ...] = TRIAL_COMPONENTS,
) -> tuple[PropState, BlockObs]:
    """
    One block: n_prop_steps of guided propagation, then the guide energy (for the
    population control and the outlier rejection) and, unless measure_trial=False
    (equilibration), the fragment estimator.

    Returns BlockObs.scalars {"guide_weight", "guide_energy", "wp", *components}.
    """

    step = lambda st: guide_prop_ops.step(
        st,
        params=params,
        ham_data=ham_data,
        trial_data=guide_data,
        trial_ops=guide_ops,
        meas_ops=guide_meas_ops,
        prop_ctx=guide_prop_ctx,
        meas_ctx=guide_meas_ctx,
    )

    def _scan_step(carry: PropState, _x: Any):
        return step(carry), None

    state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)

    walkers_new = wk.orthonormalize(state.walkers, sys.walker_kind)
    guide_overlaps = wk.vmap_chunked(
        guide_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers_new, guide_data)
    state = state._replace(walkers=walkers_new, overlaps=guide_overlaps)

    # the guide's local energy: population control and outlier rejection, as block_mixed
    guide_e_kernel = guide_meas_ops.require_kernel(k_energy)
    guide_e_samples = wk.vmap_chunked(
        guide_e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )(state.walkers, ham_data, guide_meas_ctx, guide_data)
    guide_e_samples = jnp.real(guide_e_samples)

    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    e_ref = state.e_estimate
    is_bad = ~jnp.isfinite(guide_e_samples) | (jnp.abs(guide_e_samples - e_ref) > thresh)
    guide_e_samples = jnp.where(is_bad, e_ref, guide_e_samples)

    guide_weights = jnp.where(is_bad, 0.0, state.weights)
    guide_w_block = jnp.sum(guide_weights)
    guide_e_block = jnp.sum(guide_weights * guide_e_samples) / guide_w_block
    guide_e_block = jnp.where(guide_w_block == 0, e_ref, guide_e_block)

    alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(guide_e_block))
    state = state._replace(
        weights=guide_weights,
        e_estimate=(1.0 - alpha) * state.e_estimate + alpha * guide_e_block,
    )

    scalars: dict[str, jax.Array] = {"guide_weight": guide_w_block, "guide_energy": guide_e_block}

    if measure_trial:
        trial_e_kernel = trial_meas_ops.require_kernel(k_energy)
        if trial_meas_ops.needs_rng(k_energy):
            rng_key, sub = jax.random.split(state.rng_key)
            state = state._replace(rng_key=rng_key)
            walker_keys = jax.random.split(sub, wk.n_walkers(state.walkers))
            results = wk.vmap_chunked(
                trial_e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None, 0)
            )(state.walkers, ham_data, trial_meas_ctx, trial_data, walker_keys)
        else:
            results = wk.vmap_chunked(
                trial_e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
            )(state.walkers, ham_data, trial_meas_ctx, trial_data)

        trial_overlaps = wk.vmap_chunked(
            trial_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
        )(walkers_new, trial_data)
        # the absorbed weight, afqmc's wp = wt * <T|phi>/<G|phi>
        wp = guide_weights * trial_overlaps / guide_overlaps
        wp_block = jnp.sum(wp)
        scalars["wp"] = wp_block
        for i, name in enumerate(components):
            scalars[name] = jnp.sum(wp * results[:, i]) / wp_block

    # stochastic reconfiguration on the guide weights
    key, subkey = jax.random.split(state.rng_key)
    zeta = jax.random.uniform(subkey)
    w_sr, weights_sr = sr_fn(state.walkers, state.weights, zeta, sys.walker_kind)
    overlaps_sr = wk.vmap_chunked(
        guide_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(w_sr, guide_data)
    state = state._replace(walkers=w_sr, weights=weights_sr, overlaps=overlaps_sr, rng_key=key)

    return state, BlockObs(scalars=scalars, observables={})
