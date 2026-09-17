from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from ..mixed import MixedRecipe
from .blocks import block_frag
from .integral import build_ham_lno_df, build_ham_ulno_df
from .meas.pt2ccsd import TRIAL_COMPONENTS, make_pt2ccsd_meas_ops, plan_chunking_for_run
from .meas.upt2ccsd import make_upt2ccsd_meas_ops, plan_chunking_for_run_u
from .staging import LnoFragData, stage_pt2ccsd_trial, stage_upt2ccsd_trial
from .stat_utils import clean_frag_pt2ccsd, frag_pt2ccsd_blocking
from .trial.pt2ccsd import make_pt2ccsd_trial_data
from .trial.upt2ccsd import make_upt2ccsd_trial_data

# Recipes for the LNO fragment AFQMC, mirroring trot/mixed.py: one entry per (guide,
# trial) pair, and get_mixed_recipe(trial, guide) to look it up. The table is separate
# from trot.mixed.MIXED_RECIPES because the trial names coincide: "pt2ccsd" there is the
# full-space trial, here it is the fragment one. LnoAfqmcMixed(..., trial="pt2ccsd")
# already says LNO.
#
# The guide is whatever trot stages from the mean field in the fragment basis
# (staging.frag_mf), so nothing guide-specific lives here; a new guide is a new table
# entry with the same trial-side fields.


@dataclass(frozen=True)
class LnoMixedRecipe(MixedRecipe):
    """
    A MixedRecipe plus what the fragment loop needs:

    stage_trial:       LnoFragData -> TrialInput (narrowed from the CC-object version)
    build_ham:         (mf, lno_coeff, lno_frozen, chol_cut=) -> HamInput / HamInputU
    clean_fn:          the outlier filter that goes with blocking_fn
    components:        the names of the kernel's return values, in order
    needs_amplitudes:  whether step 3 (LNO-CCSD) has to run for this pair
    """

    build_ham: Callable[..., Any] = build_ham_lno_df
    clean_fn: Callable[..., Any] = clean_frag_pt2ccsd
    components: tuple[str, ...] = TRIAL_COMPONENTS
    needs_amplitudes: bool = True


def _pt2ccsd_recipe(guide: str, trial: str, measure_type: str) -> LnoMixedRecipe:
    return LnoMixedRecipe(
        guide=guide,
        trial=trial,
        walker_kind="restricted",
        stage_trial=stage_pt2ccsd_trial,
        make_trial_data=make_pt2ccsd_trial_data,
        make_trial_meas_ops=partial(make_pt2ccsd_meas_ops, measure_type=measure_type),
        plan_chunking=partial(plan_chunking_for_run, measure_type=measure_type),
        mixed_block_fn=block_frag,
        blocking_fn=frag_pt2ccsd_blocking,
        ham_basis="restricted",
        build_ham=build_ham_lno_df,
        clean_fn=clean_frag_pt2ccsd,
        components=TRIAL_COMPONENTS,
        needs_amplitudes=True,
    )


def _upt2ccsd_recipe(guide: str, trial: str, measure_type: str) -> LnoMixedRecipe:
    """
    The unrestricted fragment trials, on the uchol fragment hamiltonian (each spin in its
    own LNO basis, one shared cholesky index) with unrestricted walkers under a UHF guide.
    The kernel returns the same TRIAL_COMPONENTS as the restricted one, so the block
    function and the statistics are shared with it.
    """
    return LnoMixedRecipe(
        guide=guide,
        trial=trial,
        walker_kind="unrestricted",
        stage_trial=stage_upt2ccsd_trial,
        make_trial_data=make_upt2ccsd_trial_data,
        make_trial_meas_ops=partial(make_upt2ccsd_meas_ops, measure_type=measure_type),
        plan_chunking=partial(plan_chunking_for_run_u, measure_type=measure_type),
        mixed_block_fn=block_frag,
        blocking_fn=frag_pt2ccsd_blocking,
        ham_basis="uchol",
        build_ham=build_ham_ulno_df,
        clean_fn=clean_frag_pt2ccsd,
        components=TRIAL_COMPONENTS,
        needs_amplitudes=True,
    )


MIXED_RECIPES: dict[tuple[str, str], LnoMixedRecipe] = {
    ("rhf", "pt2ccsd"): _pt2ccsd_recipe("rhf", "pt2ccsd", "bar"),
    ("rhf", "pt2ccsd_sto_chol"): _pt2ccsd_recipe("rhf", "pt2ccsd_sto_chol", "sto_chol"),
    ("uhf", "upt2ccsd"): _upt2ccsd_recipe("uhf", "upt2ccsd", "bar"),
    ("uhf", "upt2ccsd_sto_chol"): _upt2ccsd_recipe("uhf", "upt2ccsd_sto_chol", "sto_chol"),
}

# the guide a trial pairs with when none is asked for
DEFAULT_GUIDE: dict[str, str] = {
    "pt2ccsd": "rhf",
    "pt2ccsd_sto_chol": "rhf",
    "upt2ccsd": "uhf",
    "upt2ccsd_sto_chol": "uhf",
}


def _format_pairs() -> str:
    return ", ".join(f"{guide}+{trial}" for guide, trial in sorted(MIXED_RECIPES))


def get_mixed_recipe(trial: str, guide: str | None = None) -> LnoMixedRecipe:
    """
    The (guide, trial) pipeline of the fragment AFQMC. guide=None takes the trial's
    default guide (HF); an unregistered pair lists the guides the trial pairs with.
    """
    trials = sorted({t for _, t in MIXED_RECIPES})
    if trial not in trials:
        raise ValueError(f"unknown LNO trial {trial!r}; available: {', '.join(trials)}")
    if guide is None:
        guide = DEFAULT_GUIDE[trial]
    try:
        return MIXED_RECIPES[(guide.lower(), trial)]
    except KeyError:
        guides = ", ".join(sorted(g for g, t in MIXED_RECIPES if t == trial))
        raise ValueError(
            f"no LNO recipe pairs guide={guide!r} with trial={trial!r}; guides for this trial: "
            f"{guides} (all pairs: {_format_pairs()})"
        ) from None


def available_mixed_recipes() -> tuple[tuple[str, str], ...]:
    """The registered (guide, trial) pairs."""
    return tuple(sorted(MIXED_RECIPES))


def needs_amplitudes(recipe: LnoMixedRecipe) -> bool:
    return bool(recipe.needs_amplitudes)


__all__ = [
    "LnoMixedRecipe",
    "MIXED_RECIPES",
    "DEFAULT_GUIDE",
    "get_mixed_recipe",
    "available_mixed_recipes",
    "LnoFragData",
]
