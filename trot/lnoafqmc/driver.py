from __future__ import annotations

import time
from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from .. import walkers as wk
from ..core.ops import MeasOps, TrialOps, k_energy
from ..core.system import System
from ..driver import make_run_mixed_blocks
from ..prop.types import PropOps, PropState, QmcParams
from ..stat_utils import blocking_analysis_ratio, reject_outliers
from ..walkers import stochastic_reconfiguration

print = partial(print, flush=True)

# The run loop of one fragment's AFQMC, mirroring trot/driver.py:run_mixed_qmc with the
# flow of afqmc's script/run_lno_afqmc_pt2ccsd.py:
#
#   tau = 0      the fragment energy of the initial walkers ("Initial Orbital energy")
#   eql          n_eql_blocks blocks, guide only
#   sample       n_blocks blocks, printing every chunk, stopping early once at least
#                min_blocks blocks are in and the fragment error is below
#                stop_ratio * max_error
#   post         outlier blocks dropped (recipe.clean_fn), guide and fragment blocking
#                (recipe.blocking_fn), the final numbers as host floats
#
# The scan itself is trot's make_run_mixed_blocks over lnoafqmc.blocks.block_frag.
# Everything returned is on the host, so nothing of the fragment stays on the device.


class FragQmcResult(NamedTuple):
    guide_mean_energy: float
    guide_stderr_energy: float
    guide_block_energies: np.ndarray
    guide_block_weights: np.ndarray
    frag_mean_energy: float
    frag_stderr_energy: float
    frag_init_energy: float
    frag_block_weights: np.ndarray
    frag_block_components: dict[str, np.ndarray]
    n_blocks_run: int
    n_outliers: int
    weightp_over_weight: float


def _init_frag_energy(
    init_state: PropState,
    ham_data: Any,
    trial_data: Any,
    trial_meas_ops: MeasOps,
    trial_meas_ctx: Any,
    components: tuple[str, ...],
) -> tuple[float, float]:
    """
    The tau = 0 line: the fragment estimator on the first initial walker, and the total
    absorbed weight of the initial population, sum_i w_i <T|phi_i>/<G|phi_i>.
    """
    walker_0 = wk.take_walkers(init_state.walkers, jnp.array([0]))
    kernel = trial_meas_ops.require_kernel(k_energy)
    if trial_meas_ops.needs_rng(k_energy):
        keys = jax.random.split(init_state.rng_key, wk.n_walkers(walker_0))
        out = wk.vmap_chunked(kernel, n_chunks=1, in_axes=(0, None, None, None, 0))(
            walker_0, ham_data, trial_meas_ctx, trial_data, keys
        )
    else:
        out = wk.vmap_chunked(kernel, n_chunks=1, in_axes=(0, None, None, None))(
            walker_0, ham_data, trial_meas_ctx, trial_data
        )
    c = {name: out[0, i] for i, name in enumerate(components)}
    e_init = float(jnp.real(c["e0frg"] + c["e1frg"] - c["t2frg"] * c["e0"]))

    trial_overlaps = wk.vmap_chunked(trial_meas_ops.overlap, n_chunks=1, in_axes=(0, None))(
        init_state.walkers, trial_data
    )
    w_init = float(jnp.real(jnp.sum(init_state.weights * trial_overlaps / init_state.overlaps)))
    return e_init, w_init


def run_frag_qmc(
    *,
    sys: System,
    params: QmcParams,
    ham_data: Any,
    guide_data: Any,
    guide_ops: TrialOps,
    guide_prop_ops: PropOps,
    guide_meas_ops: MeasOps,
    trial_data: Any,
    trial_meas_ops: MeasOps,
    mix_block_fn: Callable[..., Any],
    blocking_fn: Callable[..., Any],
    clean_fn: Callable[..., Any],
    components: tuple[str, ...],
    max_error: float | None = None,
    stop_ratio: float = 0.7,
    min_blocks: int = 120,
    outlier_zeta: float = 20.0,
    state: PropState | None = None,
    guide_meas_ctx: Any | None = None,
    trial_meas_ctx: Any | None = None,
    mesh: Mesh | None = None,
    label: str = "",
) -> FragQmcResult:
    """
    Equilibration then sampling of one fragment. The importance sampling is governed by
    the guide; the fragment energy is measured against the trial from the sampling phase
    on. Returns the fragment correlation energy with its error, and the guide energy.
    """
    guide_prop_ctx = guide_prop_ops.build_prop_ctx(ham_data, guide_ops.get_rdm1(guide_data), params)
    if guide_meas_ctx is None:
        guide_meas_ctx = guide_meas_ops.build_meas_ctx(ham_data, guide_data)
    if trial_meas_ctx is None:
        trial_meas_ctx = trial_meas_ops.build_meas_ctx(ham_data, trial_data)

    if state is None:
        state = guide_prop_ops.init_prop_state(
            sys=sys,
            ham_data=ham_data,
            trial_ops=guide_ops,
            trial_data=guide_data,
            meas_ops=guide_meas_ops,
            params=params,
            mesh=mesh,
        )

    e_init, w_init = _init_frag_energy(
        state, ham_data, trial_data, trial_meas_ops, trial_meas_ctx, components
    )

    if mesh is None or mesh.size == 1:
        block_fn_sr = mix_block_fn
    else:
        data_sh = NamedSharding(mesh, P("data"))
        block_fn_sr = partial(
            mix_block_fn, sr_fn=partial(stochastic_reconfiguration, data_sharding=data_sh)
        )

    run_blocks = make_run_mixed_blocks(
        mixed_block_fn=block_fn_sr,
        sys=sys,
        params=params,
        guide_ops=guide_ops,
        guide_meas_ops=guide_meas_ops,
        guide_prop_ops=guide_prop_ops,
        trial_meas_ops=trial_meas_ops,
        observable_names=(),
    )

    t0 = time.perf_counter()
    t_mark = t0
    block_time = params.dt * params.n_prop_steps
    tag = f"[{label}] " if label else ""

    # ------------------------------------------------------------------ equilibration
    print_every = params.n_eql_blocks // 5 if params.n_eql_blocks >= 5 else 0
    guide_block_e_eq = [float(np.real(state.e_estimate))]
    guide_block_w_eq = [float(np.real(jnp.sum(state.weights)))]

    # the same layout as run_mixed_qmc: the tau = 0 row carries the initial walkers
    # measured against both the guide and the trial (here the fragment correlation
    # energy); afterwards the fragment is not measured until sampling
    print(f"\n{tag}Equilibration:")
    print("The fragment energy is not measured until the sampling phase\n")
    print(
        f"{'':4s}{'block':>9s}  {'tau':>6s}  {'Guide_E_blk':>14s}  {'Guide_W_blk':>12s}  "
        f"{'Frag_E_blk':>14s}  {'Frag_W_blk':>12s}  {'nodes':>10s}  {'t[s]':>8s}"
    )
    print(
        f"[eql {0:4d}/{params.n_eql_blocks}]  {0.0:6.2f}  "
        f"{guide_block_e_eq[0]:14.10f}  {guide_block_w_eq[0]:12.6e}  "
        f"{e_init:14.10f}  {w_init:12.6e}  {int(state.node_encounters):10d}  {0.0:8.1f}"
    )
    chunk = print_every if print_every > 0 else 1
    for start in range(0, params.n_eql_blocks, chunk):
        n = min(chunk, params.n_eql_blocks - start)
        state, scalars, _ = run_blocks(
            state,
            ham_data=ham_data,
            guide_data=guide_data,
            guide_meas_ctx=guide_meas_ctx,
            guide_prop_ctx=guide_prop_ctx,
            trial_data=trial_data,
            trial_meas_ctx=trial_meas_ctx,
            n_blocks=n,
            measure_trial=False,
        )
        e_chunk = np.asarray(scalars["guide_energy"])
        w_chunk = np.asarray(scalars["guide_weight"])
        guide_block_e_eq.extend(e_chunk.tolist())
        guide_block_w_eq.extend(w_chunk.tolist())
        w_avg = float(np.mean(w_chunk))
        e_avg = float(np.mean(e_chunk * w_chunk) / w_avg)
        print(
            f"[eql {start + n:4d}/{params.n_eql_blocks}]  {(start + n) * block_time:6.2f}  "
            f"{e_avg:14.10f}  {w_avg:12.6e}  {'-':>14s}  {'-':>12s}  "
            f"{int(state.node_encounters):10d}  {time.perf_counter() - t0:8.1f}"
        )

    # ------------------------------------------------------------------ sampling
    print(f"\n{tag}Sampling:")
    if max_error is not None and max_error > 0:
        print(
            f"Early stop once the fragment error < {stop_ratio:.2f} x {max_error:.3e} "
            f"= {stop_ratio * max_error:.3e}, after at least {min_blocks} blocks"
        )
    print("")
    print_every = params.n_blocks // 10 if params.n_blocks >= 10 else 0
    chunk = print_every if print_every > 0 else 1

    guide_w_sp: list[float] = []
    guide_e_sp: list[float] = []
    wp_sp: list[complex] = []
    comp_sp: dict[str, list[complex]] = {name: [] for name in components}

    print(
        f"{'':4s}{'block':>9s}  {'Guide_E_avg':>14s}  {'Guide_E_err':>10s}  {'Guide_W':>12s}  "
        f"{'Frag_E_avg':>14s}  {'Frag_E_err':>10s}  {'nodes':>10s}  {'dt[s/bl]':>10s}  {'t[s]':>8s}"
    )

    n_done = 0
    for start in range(0, params.n_blocks, chunk):
        n = min(chunk, params.n_blocks - start)
        state, scalars, _ = run_blocks(
            state,
            ham_data=ham_data,
            guide_data=guide_data,
            guide_meas_ctx=guide_meas_ctx,
            guide_prop_ctx=guide_prop_ctx,
            trial_data=trial_data,
            trial_meas_ctx=trial_meas_ctx,
            n_blocks=n,
            measure_trial=True,
        )
        n_done = start + n
        guide_w_sp.extend(np.asarray(scalars["guide_weight"]).tolist())
        guide_e_sp.extend(np.asarray(scalars["guide_energy"]).tolist())
        wp_sp.extend(np.asarray(scalars["wp"]).tolist())
        for name in components:
            comp_sp[name].extend(np.asarray(scalars[name]).tolist())

        elapsed = time.perf_counter() - t0
        dt_per_block = (time.perf_counter() - t_mark) / float(n)
        t_mark = time.perf_counter()

        stats = blocking_analysis_ratio(
            jnp.asarray(guide_e_sp), jnp.asarray(guide_w_sp), print_q=False
        )
        guide_mu, guide_se = stats["mu"], stats["se_star"]
        guide_w_avg = float(np.mean(guide_w_sp))

        frag_stats = blocking_fn(
            jnp.asarray(wp_sp),
            *(jnp.asarray(comp_sp[name]) for name in components),
            printQ=False,
            final=False,
        )
        frag_e, frag_err = (None, None) if frag_stats is None else frag_stats
        guide_se_s = f"{guide_se:10.3e}" if guide_se is not None else f"{'-':>10s}"
        frag_e_s = f"{float(np.real(frag_e)):14.10f}" if frag_e is not None else f"{'-':>14s}"
        frag_err_s = f"{float(np.real(frag_err)):10.3e}" if frag_err is not None else f"{'-':>10s}"
        print(
            f"[blk {n_done:4d}/{params.n_blocks}]  {guide_mu:14.10f}  {guide_se_s}  {guide_w_avg:12.6e}  "
            f"{frag_e_s}  {frag_err_s}  {int(state.node_encounters):10d}  {dt_per_block:10.3f}  "
            f"{elapsed:8.1f}"
        )
        if (
            max_error is not None
            and max_error > 0
            and n_done >= min_blocks
            and frag_err is not None
            and float(np.real(frag_err)) < stop_ratio * max_error
        ):
            print(
                f"\n{tag}Fragment error {float(np.real(frag_err)):.3e} < {stop_ratio * max_error:.3e} "
                f"at block {n_done}: stopping."
            )
            break

    # ------------------------------------------------------------------ post processing
    guide_w = jnp.asarray(guide_w_sp)
    guide_e = jnp.asarray(guide_e_sp)
    guide_clean, _ = reject_outliers(jnp.column_stack((guide_e, guide_w)), obs=0)
    print(f"\n{tag}Rejected {guide_e.shape[0] - guide_clean.shape[0]} guide outlier blocks.")
    guide_stats = blocking_analysis_ratio(guide_clean[:, 0], guide_clean[:, 1], print_q=True)
    # with too few blocks the guide blocking analysis has no error estimate (None)
    guide_mean, guide_err = (
        float("nan") if guide_stats[k] is None else float(guide_stats[k]) for k in ("mu", "se_star")
    )

    wp = jnp.asarray(wp_sp)
    comps = {name: jnp.asarray(comp_sp[name]) for name in components}
    ept_sp = jnp.real(comps["e0frg"] + comps["e1frg"] - comps["t2frg"] * comps["e0"])
    (wp_c, *comps_c), mask = clean_fn(
        ept_sp, wp, *(comps[name] for name in components), zeta=outlier_zeta
    )
    n_out = int(jnp.sum(~mask))
    print(f"{tag}Rejected {n_out} fragment outlier blocks.")

    frag_stats = blocking_fn(wp_c, *comps_c, printQ=True, final=True)
    if frag_stats is None:
        print(
            f"{tag}Too few sampling blocks for a blocking analysis; falling back to the unblocked error."
        )
        frag_stats = blocking_fn(wp_c, *comps_c, printQ=False, final=False)
    if frag_stats is None:
        frag_mean, frag_err = float("nan"), float("nan")
    else:
        frag_mean, frag_err = float(np.real(frag_stats[0])), float(np.real(frag_stats[1]))

    weightp_over_weight = float(np.real(np.mean(wp_sp) / np.mean(guide_w_sp)))
    print(f"\n{tag}Final AFQMC/{'guide'} energy:            {guide_mean:.6f} +/- {guide_err:.6f}")
    print(f"{tag}Final AFQMC/pt2CCSD fragment energy: {frag_mean:.6f} +/- {frag_err:.6f}")
    print(f"{tag}<t1> = weightp/weight = {weightp_over_weight:.5f}")
    print(f"{tag}Sampling blocks: {n_done}, wall time {time.perf_counter() - t0:.1f} s")

    return FragQmcResult(
        guide_mean_energy=guide_mean,
        guide_stderr_energy=guide_err,
        guide_block_energies=np.concatenate([np.asarray(guide_block_e_eq), np.asarray(guide_e_sp)]),
        guide_block_weights=np.concatenate([np.asarray(guide_block_w_eq), np.asarray(guide_w_sp)]),
        frag_mean_energy=frag_mean,
        frag_stderr_energy=frag_err,
        frag_init_energy=e_init,
        frag_block_weights=np.asarray(wp_sp),
        frag_block_components={name: np.asarray(comp_sp[name]) for name in components},
        n_blocks_run=n_done,
        n_outliers=n_out,
        weightp_over_weight=weightp_over_weight,
    )
