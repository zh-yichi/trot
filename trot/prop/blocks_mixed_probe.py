"""
The block function of mixed guide/trial AFQMC with the probing step
(prop/afqmc.afqmc_step_probe, through make_prop_ops_probe / make_prop_ops_u_probe):
prop/blocks_mixed.block_mixed with the quantities of the importance sampling averaged
over the block.

The propagation is block_mixed's, step for step. Each step also returns, per walker, the
norm and the mean square per field of the mean field subtracted force bias, ln|importance function| and cos(theta)
(prop.types.StepProbe). They are averaged over the walkers and the n_prop_steps steps of
the block with the weight of the walker before the step,

    <A> = sum_k sum_i w_i^(k) A_i^(k) / sum_k W_k,   W_k = sum_i w_i^(k)

and returned as the block scalars "probe_fb_norm", "probe_fb_ms", "probe_ln_imp" and
"probe_cos_theta", next to "probe_weight" = sum_k W_k. "probe_fb_norm" is the average of
the length of the force bias vector of a walker, (sum_gamma |f_gamma|^2)^(1/2), and
"probe_fb_ms" the average of its mean square per field, sum_gamma |f_gamma|^2 / N_gamma:
the root of the latter is the root mean square of the force bias per field (the root
taken after the average over the walkers, which the driver prints as "rms fb"). "probe_var_ln_imp" is the variance of ln|I| over the
walkers within a step, averaged over the steps of the block with W_k,

    var ln|I| = sum_k W_k (<ln|I|^2>_k - <ln|I|>_k^2) / sum_k W_k

so it does not contain the drift of the average of ln|I| from step to step. A walker with
a non-finite probed value (ln|I| = -inf at I = 0) is left out of all of them. Everything after the propagation, the guide energy, the
outlier handling (params.mixed_outlier), the trial measurement and the reconfiguration, is
block_mixed's (measure_block_mixed).
"""

from __future__ import annotations

from typing import Any, Callable

import jax.numpy as jnp
from jax import lax

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps
from ..core.system import System
from .blocks import BlockObs
from .blocks_mixed import PT2_COMPONENTS, measure_block_mixed
from .types import PropOps, PropState, QmcParams

PROBE_SCALARS: tuple[str, ...] = (
    "probe_fb_norm",
    "probe_fb_ms",
    "probe_ln_imp",
    "probe_var_ln_imp",
    "probe_cos_theta",
)


def block_mixed_probe(
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
    components: tuple[str, ...] = PT2_COMPONENTS,
) -> tuple[PropState, BlockObs]:
    """
    One block of block_mixed with the probing step; see the module docstring.
    guide_prop_ops.step must be a probing step, returning (PropState, StepProbe).
    """

    def _scan_step(carry: PropState, _x: Any):
        w = jnp.real(carry.weights)
        new, probe = guide_prop_ops.step(  # type: ignore[misc]
            carry,
            params=params,
            ham_data=ham_data,
            trial_data=guide_data,
            trial_ops=guide_ops,
            meas_ops=guide_meas_ops,
            prop_ctx=guide_prop_ctx,
            meas_ctx=guide_meas_ctx,
        )
        ok = (
            jnp.isfinite(probe.fb_norm)
            & jnp.isfinite(probe.fb_ms)
            & jnp.isfinite(probe.ln_imp)
            & jnp.isfinite(probe.cos_theta)
        )
        wo = jnp.where(ok, w, 0.0)
        ln_imp = jnp.where(ok, probe.ln_imp, 0.0)
        w_step = jnp.sum(wo)
        w_step_safe = jnp.where(w_step == 0, 1.0, w_step)
        s_ln = jnp.sum(wo * ln_imp)
        # the variance over the walkers of this step, times the weight of the step
        var_ln = jnp.sum(wo * ln_imp**2) - s_ln**2 / w_step_safe
        sums = jnp.stack(
            [
                w_step,
                jnp.sum(wo * jnp.where(ok, probe.fb_norm, 0.0)),
                jnp.sum(wo * jnp.where(ok, probe.fb_ms, 0.0)),
                s_ln,
                var_ln,
                jnp.sum(wo * jnp.where(ok, probe.cos_theta, 0.0)),
            ]
        )
        return new, sums

    state, step_sums = lax.scan(_scan_step, state, xs=None, length=params.n_prop_steps)
    tot = jnp.sum(step_sums, axis=0)
    w_safe = jnp.where(tot[0] == 0, 1.0, tot[0])

    state, obs = measure_block_mixed(
        state,
        sys=sys,
        params=params,
        ham_data=ham_data,
        guide_data=guide_data,
        guide_meas_ops=guide_meas_ops,
        guide_meas_ctx=guide_meas_ctx,
        trial_data=trial_data,
        trial_meas_ops=trial_meas_ops,
        trial_meas_ctx=trial_meas_ctx,
        sr_fn=sr_fn,
        measure_trial=measure_trial,
        components=components,
    )
    scalars = dict(obs.scalars)
    scalars["probe_weight"] = tot[0]
    for k, name in enumerate(PROBE_SCALARS):
        scalars[name] = tot[k + 1] / w_safe
    return state, BlockObs(scalars=scalars, observables=obs.observables)


__all__ = ["PROBE_SCALARS", "block_mixed_probe"]
