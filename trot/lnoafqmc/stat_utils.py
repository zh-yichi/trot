from __future__ import annotations

import jax.numpy as jnp

# Statistics for the LNO fragment estimator, mirroring the pt2CCSD functions of
# trot/stat_utils.py one for one:
#
#     trot/stat_utils.py                lnoafqmc/stat_utils.py
#     _pt2ccsd_energy                   _frag_pt2ccsd_energy
#     _pt2ccsd_delta_method_error       _frag_pt2ccsd_delta_method_error
#     pt2ccsd_blocking                  frag_pt2ccsd_blocking
#     clean_pt2ccsd                     clean_frag_pt2ccsd
#
# The full-space estimator is E = h0 + <e0> + <e1> - <t2><e0>, three components plus h0.
# The fragment estimator of afqmc's lno_afqmc (ept2frg_blocking) is
#
#     E_F = <e0frg> + <e1frg> - <t2frg><e0>,      <x> = sum(w x) / sum(w)
#
# four components and no h0, where w is the block weight absorbed with the trial/guide
# overlap ratio (afqmc: wp = wt * t1) and e0frg is a quantity of its own rather than e0
# again. So e0 enters only the cross term, which is what changes the delta-method
# partials below; the blocking sweep and the plateau detection are trot's.
#
# The guide energy uses trot.stat_utils.blocking_analysis_ratio / reject_outliers as in
# run_mixed_qmc, so nothing for it is mirrored here.

FRAG_COMPONENTS = ("t2frg", "e0frg", "e1frg", "e0")


def frag_pt2ccsd_energy_fn(h0, t2frg, e0frg, e1frg, e0):
    """
    The fragment energy from its four averaged components, E_F = <e0frg> + <e1frg>
    - <t2frg><e0>. h0 is taken for the signature shared with trot's energy_fn's and
    ignored: the fragment estimator is a correlation energy.
    """
    return e0frg + e1frg - t2frg * e0


def clean_frag_pt2ccsd(ept_sp, weights, t2frg_sp, e0frg_sp, e1frg_sp, e0_sp, zeta=20):
    """
    Drop outlier blocks of the fragment estimator, by the rule of clean_pt2ccsd: a block
    whose energy sits more than zeta median absolute deviations from the median goes.

    Returns the kept (weights, t2frg, e0frg, e1frg, e0) and the boolean mask that kept
    them, so the caller can report what was removed.
    """
    d = jnp.abs(ept_sp - jnp.median(ept_sp))
    d_med = jnp.median(d)
    d_med = jnp.where(d_med == 0, 1e-10, d_med)
    z = d / d_med
    mask = z < zeta
    if int(jnp.sum(~mask)) > 0:
        print(
            f"Remove outlier blocks zeta {z[~mask]} \n"
            f"                    energy {ept_sp[~mask]} \n"
            f"                    weight {weights.real[~mask]} "
        )
    return (weights[mask], t2frg_sp[mask], e0frg_sp[mask], e1frg_sp[mask], e0_sp[mask]), mask


def _frag_pt2ccsd_energy(weights, t2frg_sp, e0frg_sp, e1frg_sp, e0_sp):
    """E_F = <e0frg> + <e1frg> - <t2frg><e0>, with <x> = sum(w x) / sum(w)."""
    wt_avg = jnp.mean(weights)
    t2frg_avg = jnp.mean(weights * t2frg_sp) / wt_avg
    e0frg_avg = jnp.mean(weights * e0frg_sp) / wt_avg
    e1frg_avg = jnp.mean(weights * e1frg_sp) / wt_avg
    e0_avg = jnp.mean(weights * e0_sp) / wt_avg
    return frag_pt2ccsd_energy_fn(None, t2frg_avg, e0frg_avg, e1frg_avg, e0_avg)


def _frag_pt2ccsd_delta_method_error(weights, t2frg_sp, e0frg_sp, e1frg_sp, e0_sp):
    """
    Weight aware naive error for the fragment estimator, without blocking.

    In aggregate form E = E0frg/W + E1frg/W - T2frg E0 / W**2 with E0frg = sum(w e0frg)
    and so on. Each sample's contribution to the aggregates is propagated through a first
    order linearization to its influence on E, then the variance of the mean is taken.
    afqmc's ept2frg_blocking(final=False), term for term.

    Ignores autocorrelation between blocks, so it underestimates the true error; it is
    for progress reporting, not for a final number.
    """
    w = weights
    n = len(w)
    e0frg_agg = jnp.sum(w * e0frg_sp)
    e1frg_agg = jnp.sum(w * e1frg_sp)
    t2frg_agg = jnp.sum(w * t2frg_sp)
    e0_agg = jnp.sum(w * e0_sp)
    w_agg = jnp.sum(w)

    # partials of E with respect to each aggregate; unlike the full-space estimator, e0
    # appears in the cross term only
    d_e0frg = 1.0 / w_agg
    d_e1frg = 1.0 / w_agg
    d_t2frg = -e0_agg / w_agg**2
    d_e0 = -t2frg_agg / w_agg**2
    d_w = -e0frg_agg / w_agg**2 - e1frg_agg / w_agg**2 + 2.0 * t2frg_agg * e0_agg / w_agg**3

    infl = (
        d_e0frg * (w * e0frg_sp)
        + d_e1frg * (w * e1frg_sp)
        + d_t2frg * (w * t2frg_sp)
        + d_e0 * (w * e0_sp)
        + d_w * w
    ).real
    var_mean = jnp.sum(infl**2) * n / (n - 1)
    return jnp.sqrt(var_mean).real


def frag_pt2ccsd_blocking(
    weights,
    t2frg_sp,
    e0frg_sp,
    e1frg_sp,
    e0_sp,
    printQ=False,
    min_blocks=5,
    plateau_window=2,
    plateau_tol=0.04,
    final=True,
):
    """
    Blocking analysis for the LNO fragment estimator

        E_F = <e0frg> + <e1frg> - <t2frg><e0>,     <x> = sum(w x) / sum(w)

    which is nonlinear in the block averages, so the four components are averaged
    separately and combined afterwards. Same sweep and plateau detection as
    pt2ccsd_blocking; only the energy formula and the number of components differ.

    final=True   full blocking sweep over block sizes, with plateau detection.
                 Needs enough samples for at least min_blocks blocks.
    final=False  no blocking. The error comes from a first order (delta method)
                 linearization of the estimator, treating each sample as independent.
                 Cheap and defined from two samples up, so it is what to use for
                 progress reporting while a run is still accumulating blocks.

    Returns
    -------
    (energy, error), or **None** when there are too few samples for the requested
    analysis.
    """
    nsample = len(weights)

    # the energy itself only needs one sample, but an error never does
    if nsample < 2:
        return None

    energy_avg = _frag_pt2ccsd_energy(weights, t2frg_sp, e0frg_sp, e1frg_sp, e0_sp)

    if not final:
        return energy_avg.real, _frag_pt2ccsd_delta_method_error(
            weights, t2frg_sp, e0frg_sp, e1frg_sp, e0_sp
        )

    max_size = max(1, nsample // min_blocks)

    block_errs = []
    block_means = []
    block_sizes = []

    for block_size in range(1, max_size + 1):
        n_blocks = nsample // block_size
        if n_blocks < min_blocks:
            break

        sl = slice(0, n_blocks * block_size)
        wt = weights[sl].reshape(n_blocks, block_size)
        wt_t2frg = (weights[sl] * t2frg_sp[sl]).reshape(n_blocks, block_size)
        wt_e0frg = (weights[sl] * e0frg_sp[sl]).reshape(n_blocks, block_size)
        wt_e1frg = (weights[sl] * e1frg_sp[sl]).reshape(n_blocks, block_size)
        wt_e0 = (weights[sl] * e0_sp[sl]).reshape(n_blocks, block_size)

        block_wt = jnp.sum(wt, axis=1)
        block_t2frg = jnp.sum(wt_t2frg, axis=1) / block_wt
        block_e0frg = jnp.sum(wt_e0frg, axis=1) / block_wt
        block_e1frg = jnp.sum(wt_e1frg, axis=1) / block_wt
        block_e0 = jnp.sum(wt_e0, axis=1) / block_wt

        block_energy = (block_e0frg + block_e1frg - block_t2frg * block_e0).real
        block_mean = jnp.mean(block_energy)
        block_error = jnp.std(block_energy, ddof=1) / jnp.sqrt(n_blocks)

        block_sizes.append(block_size)
        block_means.append(block_mean)
        block_errs.append(block_error)

    if not block_errs:
        # not enough samples for even one block size at this min_blocks
        return None

    # --- Plateau detection ---
    errs = jnp.array(block_errs)
    plateau_idx = None

    if len(errs) >= plateau_window + 1:
        for i in range(1, len(errs) - plateau_window + 1):
            window = errs[i : i + plateau_window]
            rel_changes = jnp.abs(jnp.diff(window) / window[:-1])
            if jnp.all(rel_changes < plateau_tol):
                plateau_idx = i
                break

    if plateau_idx is not None:
        err = jnp.mean(errs[plateau_idx : plateau_idx + plateau_window])
    else:
        err = errs.max()

    # --- Printing ---
    if printQ:
        print("Performing Blocking Analysis for the LNO-AFQMC/pt2CCSD fragment energy...")
        print(f"{'Bsz':>4s}  {'NB':>4s}  {'Nsp':>4s}  {'Energy':>11s}  {'Error':>8s}")

        if plateau_idx is not None:
            print_end = min(len(block_errs), plateau_idx + plateau_window + 3)
        else:
            print_end = len(block_errs)

        for i in range(print_end):
            bs = block_sizes[i]
            nb = nsample // bs
            marker = "  <--" if (plateau_idx is not None and i == plateau_idx) else ""
            print(
                f"{bs:4d}  {nb:4d}  {bs*nb:4d}  {block_means[i]:11.6f}  {block_errs[i]:8.6f}{marker}"
            )

        if plateau_idx is not None:
            print(f"Plateau found at block size {block_sizes[plateau_idx]}, error = {err.real:.6f}")
        else:
            print(f"No plateau found, using max error = {err.real:.6f}")

    return energy_avg.real, err.real
