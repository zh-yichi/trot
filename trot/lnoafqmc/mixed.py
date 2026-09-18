from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from ..meas.pt2ccsd import get_pt2ccsd_meas_cfg
from ..mixed import GUIDES, GuideSpec, MixedRecipe, TrialSpec, pair_recipe
from .blocks import block_frag
from .integral import build_ham_lno_df, build_ham_ulno_df
from .meas.pt2ccsd import TRIAL_COMPONENTS, make_pt2ccsd_meas_ops, plan_chunking_for_run
from .meas.upt2ccsd import make_upt2ccsd_meas_ops, plan_chunking_for_run_u
from .staging import (
    LnoFragData,
    stage_cisd_guide,
    stage_pt2ccsd_trial,
    stage_ucisd_guide,
    stage_upt2ccsd_trial,
)
from .stat_utils import clean_frag_pt2ccsd, frag_pt2ccsd_blocking, frag_pt2ccsd_energy_fn
from .trial.pt2ccsd import make_pt2ccsd_trial_data
from .trial.upt2ccsd import make_upt2ccsd_trial_data

# Recipes for the LNO fragment AFQMC, mirroring trot/mixed.py. The guides are trot's own
# (trot.mixed.GUIDES): the HF ones are staged by trot from the mean field in the
# fragment basis (staging.frag_mf), the CISD ones from the fragment's CCSD amplitudes
# (staging.stage_cisd_guide / stage_ucisd_guide). The trials are the fragment
# estimators, in a table of their own because the names coincide with trot's: "pt2ccsd"
# there is the full-space trial, here it is the fragment one. get_mixed_recipe(trial,
# guide) pairs the two through trot's pair_recipe, so the same compatibility rules
# (hamiltonian basis, walker kind) apply.


@dataclass(frozen=True)
class LnoMixedRecipe(MixedRecipe):
    """
    A MixedRecipe plus what the fragment loop needs:

    stage_trial:       LnoFragData -> TrialInput (narrowed from the CC-object version)
    stage_guide:       LnoFragData -> TrialInput for a guide built from the amplitudes,
                       None for one staged from the mean field
    build_ham:         (mf, lno_coeff, lno_frozen, chol_cut=) -> HamInput / HamInputU
    energy_fn:         (h0, *averaged components) -> fragment energy; h0 is ignored
    blocking_fn:       (weights, *components, printQ=, final=) -> (E, err) | None
    clean_fn:          (block energies, weights, *components, zeta=) ->
                       ((weights, *components), mask)
    components:        the names of the kernel's return values, in order
    needs_amplitudes:  whether step 3 (LNO-CCSD) has to run for this pair, for the
                       trial or for the guide
    """

    stage_guide: Callable[..., Any] | None = None
    build_ham: Callable[..., Any] = build_ham_lno_df
    clean_fn: Callable[..., Any] = clean_frag_pt2ccsd
    components: tuple[str, ...] = TRIAL_COMPONENTS
    needs_amplitudes: bool = True


# guides built from the fragment amplitudes rather than staged from the mean field
GUIDE_STAGERS: dict[str, Callable[[LnoFragData], Any]] = {
    "cisd": stage_cisd_guide,
    "ucisd": stage_ucisd_guide,
}


def _pt2ccsd_trial(name: str, measure_type: str) -> TrialSpec:
    return TrialSpec(
        name=name,
        kind="pt2ccsd",
        stage=stage_pt2ccsd_trial,
        make_data=make_pt2ccsd_trial_data,
        make_meas_ops=partial(make_pt2ccsd_meas_ops, measure_type=measure_type),
        components=TRIAL_COMPONENTS,
        energy_fn=frag_pt2ccsd_energy_fn,
        blocking_fn=frag_pt2ccsd_blocking,
        clean_fn=clean_frag_pt2ccsd,
        ham_basis="restricted",
        walker_kinds=frozenset({"restricted"}),
        source="cc",
        cc_kind="ccsd",
        default_guide="rhf",
        plan_chunking=partial(plan_chunking_for_run, measure_type=measure_type),
        cfg_getter=get_pt2ccsd_meas_cfg,
    )


def _upt2ccsd_trial(name: str, measure_type: str) -> TrialSpec:
    """
    The unrestricted fragment trials, on the uchol fragment hamiltonian (each spin in its
    own LNO basis, one shared cholesky index) with unrestricted walkers. The kernel
    returns the same TRIAL_COMPONENTS as the restricted one, so the block function and
    the statistics are shared with it.
    """
    return TrialSpec(
        name=name,
        kind="upt2ccsd",
        stage=stage_upt2ccsd_trial,
        make_data=make_upt2ccsd_trial_data,
        make_meas_ops=partial(make_upt2ccsd_meas_ops, measure_type=measure_type),
        components=TRIAL_COMPONENTS,
        energy_fn=frag_pt2ccsd_energy_fn,
        blocking_fn=frag_pt2ccsd_blocking,
        clean_fn=clean_frag_pt2ccsd,
        ham_basis="uchol",
        walker_kinds=frozenset({"unrestricted"}),
        source="cc",
        cc_kind="uccsd",
        default_guide="uhf",
        plan_chunking=partial(plan_chunking_for_run_u, measure_type=measure_type),
        cfg_getter=get_pt2ccsd_meas_cfg,
    )


TRIALS: dict[str, TrialSpec] = {
    "pt2ccsd": _pt2ccsd_trial("pt2ccsd", "bar"),
    "pt2ccsd_sto_chol": _pt2ccsd_trial("pt2ccsd_sto_chol", "sto_chol"),
    "upt2ccsd": _upt2ccsd_trial("upt2ccsd", "bar"),
    "upt2ccsd_sto_chol": _upt2ccsd_trial("upt2ccsd_sto_chol", "sto_chol"),
}

_BUILD_HAM = {"restricted": build_ham_lno_df, "uchol": build_ham_ulno_df}

# the guide a trial pairs with when none is asked for
DEFAULT_GUIDE: dict[str, str] = {name: spec.default_guide for name, spec in TRIALS.items()}


def _lno_recipe(guide: GuideSpec, trial: TrialSpec) -> LnoMixedRecipe:
    base = pair_recipe(guide, trial)
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(MixedRecipe)}
    fields["mixed_block_fn"] = block_frag
    return LnoMixedRecipe(
        **fields,
        stage_guide=GUIDE_STAGERS.get(guide.name),
        build_ham=_BUILD_HAM[trial.ham_basis],
        needs_amplitudes=trial.source == "cc" or guide.name in GUIDE_STAGERS,
    )


def _compatible(guide: GuideSpec, trial: TrialSpec) -> bool:
    return trial.ham_basis in guide.ham_bases and bool(guide.walker_kinds & trial.walker_kinds)


MIXED_RECIPES: dict[tuple[str, str], LnoMixedRecipe] = {
    (g.name, t.name): _lno_recipe(g, t)
    for g in GUIDES.values()
    for t in TRIALS.values()
    if _compatible(g, t)
}


def _format_pairs() -> str:
    return ", ".join(f"{guide}+{trial}" for guide, trial in sorted(MIXED_RECIPES))


def get_mixed_recipe(trial: str, guide: str | None = None) -> LnoMixedRecipe:
    """
    The (guide, trial) pipeline of the fragment AFQMC. guide=None takes the trial's
    default guide (the corresponding HF); an unregistered or incompatible pair lists what
    is available.
    """
    trials = sorted(TRIALS)
    if trial not in trials:
        raise ValueError(f"unknown LNO trial {trial!r}; available: {', '.join(trials)}")
    if guide is None:
        guide = DEFAULT_GUIDE[trial]
    guide = guide.lower()
    if guide not in GUIDES:
        raise ValueError(f"unknown mixed guide {guide!r}; available: {', '.join(sorted(GUIDES))}")
    try:
        return MIXED_RECIPES[(guide, trial)]
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
    "GUIDE_STAGERS",
    "TRIALS",
    "get_mixed_recipe",
    "available_mixed_recipes",
    "LnoFragData",
]
