"""
The block function of mixed guide/trial AFQMC under the stochastically sampled CCSD
guide (prop/afqmc_stocc.py): prop/blocks_mixed.block_mixed with a guide that is
different for every walker.

The guide sample of a walker lives in the state (PropStateStocc.guide_dets), so every
guide overlap and local energy here is taken with the walker's own sample, and the
stochastic reconfiguration copies the sample along with the walker: a copy must be
divided by the same guide overlap as its original for sum_w w_w |phi_w>/<G_w|phi_w> to
be unchanged. The overlaps are then permuted rather than recomputed. Everything else,
the guide energy for the population control, the outlier handling
(params.mixed_outlier) and the trial measurement, is block_mixed's.
"""

from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax import lax

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps, k_energy
from ..core.system import System
from .afqmc_stocc import PropStateStocc, guide_kernel, guide_overlaps
from .blocks import BlockObs
from .blocks_mixed import PT2_COMPONENTS
from .types import PropOps, QmcParams


def _reconfigure(state: PropStateStocc, zeta: jax.Array) -> PropStateStocc:
    """Stochastic reconfiguration of walkers, guide samples and overlaps together."""
    n = wk.n_walkers(state.walkers)
    idx = wk._sr_indices(state.weights, zeta, n)
    avg = jnp.sum(jnp.abs(state.weights)) / n
    return state._replace(
        walkers=wk.take_walkers(state.walkers, idx),
        guide_dets=state.guide_dets[idx],
        overlaps=state.overlaps[idx],
        weights=jnp.full((n,), avg, dtype=state.weights.dtype),
    )


def block_mixed_stocc(
    state: PropStateStocc,
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
    components: tuple[str, ...] = PT2_COMPONENTS,
) -> tuple[PropStateStocc, BlockObs]:
    """
    One block: n_prop_steps of guided propagation (the guide redrawn per walker at every
    step), then the guide's local energy and, unless measure_trial=False, the trial's
    components; see block_mixed. sr_fn is not used: the reconfiguration has to carry the
    guide samples, so it is this module's own (no walker sharding).
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

    def _scan_step(carry: PropStateStocc, _x: Any):
        return step(carry), None

    state, _ = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)

    walkers_new = wk.orthonormalize(state.walkers, sys.walker_kind)
    overlaps = guide_overlaps(
        guide_meas_ops, walkers_new, state.guide_dets, guide_data, n_chunks=params.n_chunks
    )
    state = state._replace(walkers=walkers_new, overlaps=overlaps)

    # the local energy with respect to the walker's own guide, <G_w|H|phi_w>/<G_w|phi_w>
    guide_e_samples = guide_kernel(
        guide_meas_ops,
        k_energy,
        state.walkers,
        state.guide_dets,
        ham_data,
        guide_meas_ctx,
        guide_data,
        n_chunks=params.n_chunks,
    )
    guide_e_samples = jnp.real(guide_e_samples)

    # outlier handling, as block_mixed: judged on the guide local energy alone
    if params.mixed_outlier not in ("clamp", "reject", "none"):
        raise ValueError(
            "params.mixed_outlier must be 'clamp', 'reject' or 'none', "
            f"got {params.mixed_outlier!r}"
        )
    thresh = jnp.sqrt(2.0 / jnp.asarray(params.dt))
    e_ref = state.e_estimate
    is_nan = ~jnp.isfinite(guide_e_samples)
    if params.mixed_outlier == "none":
        is_bad = is_nan
    else:
        is_bad = is_nan | (jnp.abs(guide_e_samples - e_ref) > thresh)
    guide_e_samples = jnp.where(is_bad, e_ref, guide_e_samples)

    clamp = params.mixed_outlier in ("clamp", "none")
    guide_weights = jnp.where(is_nan if clamp else is_bad, 0.0, state.weights)
    guide_w_block = jnp.sum(guide_weights)
    guide_w_safe = jnp.where(guide_w_block == 0, 1.0, guide_w_block)
    guide_e_block = jnp.sum(guide_weights * guide_e_samples) / guide_w_safe
    guide_e_block = jnp.where(guide_w_block == 0, e_ref, guide_e_block)

    alpha = jnp.asarray(params.shift_ema, dtype=jnp.result_type(guide_e_block))
    state = state._replace(
        weights=guide_weights,
        e_estimate=(1.0 - alpha) * state.e_estimate + alpha * guide_e_block,
    )

    obs_samples: dict[str, jax.Array] = {}

    if not measure_trial:
        key, subkey = jax.random.split(state.rng_key)
        zeta = jax.random.uniform(subkey)
        state = _reconfigure(state, zeta)._replace(rng_key=key)
        return state, BlockObs(
            scalars={"guide_weight": guide_w_block, "guide_energy": guide_e_block},
            observables=obs_samples,
        )

    # measuring with respect to the trial
    trial_e_kernel = trial_meas_ops.require_kernel(k_energy)
    results = wk.vmap_chunked(
        trial_e_kernel, n_chunks=params.n_chunks, in_axes=(0, None, None, None)
    )(state.walkers, ham_data, trial_meas_ctx, trial_data)
    results = jnp.reshape(results, (wk.n_walkers(state.walkers), -1))
    if results.shape[1] != len(components):
        raise ValueError(
            f"the trial energy kernel returns {results.shape[1]} numbers per walker but the "
            f"recipe names {len(components)} components {components}"
        )
    trial_overlaps = wk.vmap_chunked(
        trial_meas_ops.overlap, n_chunks=params.n_chunks, in_axes=(0, None)
    )(walkers_new, trial_data)
    # wp = w_guide * <T|walker> / <G_w|walker>
    trial_weights = guide_weights * trial_overlaps / overlaps
    if clamp:
        w_good = jnp.where(is_bad, 0.0, trial_weights)
        w_good_block = jnp.sum(w_good)
        has_good = w_good_block != 0
        comps_good = jnp.sum(w_good[:, None] * results, axis=0) / jnp.where(
            has_good, w_good_block, 1.0
        )
        results = jnp.where(is_bad[:, None], comps_good[None, :], results)
        trial_weights = jnp.where(is_bad & ~has_good, 0.0, trial_weights)
    trial_w_block = jnp.sum(trial_weights)
    trial_components = jnp.sum(trial_weights[:, None] * results, axis=0) / trial_w_block

    key, subkey = jax.random.split(state.rng_key)
    zeta = jax.random.uniform(subkey)
    state = _reconfigure(state, zeta)._replace(rng_key=key)

    obs = BlockObs(
        scalars={
            "guide_weight": guide_w_block,
            "guide_energy": guide_e_block,
            "trial_weight": trial_w_block,
            "trial_components": trial_components,
        },
        observables=obs_samples,
    )
    return state, obs


__all__ = ["block_mixed_stocc"]
