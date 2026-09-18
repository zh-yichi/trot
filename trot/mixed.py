from __future__ import annotations

import inspect
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

from . import staging
from .core.ops import MeasOps
from .core.system import System, WalkerKind
from .meas.cisd import get_cisd_meas_cfg, make_cisd_meas_ops
from .meas.pt2ccsd import get_pt2ccsd_meas_cfg, make_pt2ccsd_meas_ops, plan_chunking_for_run
from .meas.rhf import get_rhf_meas_cfg, make_rhf_meas_ops
from .meas.ucisd import make_ucisd_meas_ops
from .meas.ucisd_uh import make_ucisd_guide_bundle_uh
from .meas.uhf import make_uhf_meas_ops
from .meas.upt2ccsd import make_upt2ccsd_meas_ops, plan_chunking_for_run_u
from .prop.blocks import MixedBlockFn, block_mixed
from .staging import (
    StagedMfOrCc,
    TrialInput,
    _is_cc_like,
    stage_pt2ccsd_trial,
    stage_upt2ccsd_trial,
)
from .stat_utils import (
    clean_components,
    clean_pt2ccsd,
    eloc_energy_fn,
    make_component_blocking,
    pt2ccsd_blocking,
    pt2ccsd_energy_fn,
)
from .trial.cisd import make_cisd_trial_data
from .trial.pt2ccsd import make_pt2ccsd_trial_data
from .trial.rhf import make_rhf_trial_data
from .trial.ucisd import make_ucisd_trial_data
from .trial.uhf import make_uhf_trial_data
from .trial.upt2ccsd import make_upt2ccsd_trial_data

# Recipes for mixed guide/trial AFQMC, where the walkers propagate under one wavefunction
# (the guide) and the energy is measured against another (the trial):
#
#     |AFQMC> = sum_i w_i |phi_i> / <G|phi_i>
#     E_T     = sum_i wp_i E_loc^T(phi_i) / sum_i wp_i,   wp_i = w_i <T|phi_i> / <G|phi_i>
#
# The guide owns everything that touches propagation (overlap, force bias, the local
# energy used for population control, the rdm1 the initial walkers come from); all of
# that already exists for every trial kind trot stages, and setup._make_trial_bundle
# builds it from the guide's TrialInput. The trial owns everything that touches the
# measurement: its overlap, an energy kernel that returns named components per walker,
# and how the wp-averaged components combine into an energy. An ordinary trial has one
# component (the local energy) and the identity combine; pt2CCSD has three and
# h0 + <e0> + <e1> - <t2><e0>.
#
# So guides and trials are registered separately (GUIDES, TRIALS) and paired on demand:
# get_mixed_recipe(trial, guide) checks that the two agree on the hamiltonian basis and
# on a walker kind, and returns the MixedRecipe that setup_mixed and JobMixed consume. A
# new guide is one GuideSpec; a new trial is one TrialSpec; every compatible pair works
# through the same block function, driver and statistics.

WALKER_KIND_ORDER: tuple[str, ...] = ("restricted", "unrestricted", "generalized")

# kwargs the pipeline hands every trial factory; a factory that has no such knob simply
# does not receive it. Anything else unknown to the factory is an error, so a typo in
# trial_kwargs is never swallowed.
_GENERIC_MEAS_KWARGS = frozenset({"memory_mode", "nchol_chunk", "mixed_precision"})


def _adapt_meas_ops(make_fn: Callable[..., MeasOps]) -> Callable[..., MeasOps]:
    """
    Wrap a MeasOps factory so it can be called as make(sys, **kwargs) with the pipeline's
    generic kwargs whether or not it declares them.
    """
    sig = inspect.signature(make_fn)
    params = sig.parameters
    takes_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())

    def make(sys: System, **kwargs: Any) -> MeasOps:
        if takes_var_kw:
            return make_fn(sys, **kwargs)
        passed = {}
        for k, v in kwargs.items():
            if k in params:
                passed[k] = v
            elif k not in _GENERIC_MEAS_KWARGS:
                raise ValueError(
                    f"{make_fn.__name__} takes no option {k!r}; "
                    f"settable: {sorted(k for k in params if k != 'sys')}"
                )
        return make_fn(sys, **passed)

    make.__name__ = make_fn.__name__
    make.__wrapped__ = make_fn  # type: ignore[attr-defined]
    return make


def _scf_of(obj: Any) -> Any:
    return obj._scf if _is_cc_like(obj) else obj


def _stage_mf_trial(obj: Any, *, frozen: int | None = None) -> TrialInput:
    """The mean field under obj as a trial: what stage() builds from an mf object."""
    return staging._stage_mf_input(StagedMfOrCc(_scf_of(obj), frozen))


def _stage_cisd_trial(obj: Any, *, frozen: int | None = None) -> TrialInput:
    """The projected CISD of a CCSD object, as Afqmc(cc) stages it."""
    return staging._stage_cisd_input(StagedMfOrCc(obj, frozen))


def _stage_ucisd_trial(obj: Any, *, frozen: int | None = None) -> TrialInput:
    """The projected UCISD of a UCCSD object, as Afqmc(ucc) stages it."""
    return staging._stage_ucisd_input(StagedMfOrCc(obj, frozen))


@dataclass(frozen=True)
class GuideSpec:
    """
    A wavefunction that can propagate the walkers.

    name:          "rhf", "uhf", "cisd", ...
    kinds:         the TrialInput.kind values it stages as; the guide is staged by
                   staging.stage from the object that source_obj() picks, and
                   setup._make_trial_bundle builds its data, ops and meas ops from that
    source:        "mf" -- staged from the mean field (the object itself, or the one
                   under a CC object); "cc" -- needs the CC object itself
    ham_bases:     hamiltonians it can propagate on: "restricted" (one orbital basis)
                   and/or "uchol" (alpha and beta each in their own basis)
    walker_kinds:  walker representations its ops accept
    """

    name: str
    kinds: frozenset[str]
    source: str
    ham_bases: frozenset[str]
    walker_kinds: frozenset[str]
    # (sys, staged, mixed_precision) -> (trial_data, trial_ops, meas_ops) per hamiltonian
    # basis, for a guide whose bundle setup._make_trial_bundle does not build (it builds
    # UHF for every kind on the uchol hamiltonian); None means the default bundle
    make_bundle: dict[str, Callable[..., Any]] | None = None

    def source_obj(self, obj: Any) -> Any:
        """The pyscf object the guide is staged from."""
        if self.source == "cc":
            if not _is_cc_like(obj):
                raise ValueError(
                    f"guide={self.name!r} is built from CC amplitudes and needs a pyscf CC "
                    f"object, got {type(obj).__name__}."
                )
            return obj
        return _scf_of(obj)


@dataclass(frozen=True)
class TrialSpec:
    """
    A wavefunction the energy can be measured against.

    name:           "pt2ccsd", "rhf", "cisd", ...
    kind:           the TrialInput.kind stage() must produce, checked after staging
    stage:          (pyscf obj, *, frozen) -> TrialInput
    make_data:      (TrialInput.data, sys) -> trial pytree
    make_meas_ops:  (sys, **kwargs) -> MeasOps with an overlap and an "energy" kernel
    components:     names of the energy kernel's outputs per walker, in order
    energy_fn:      (h0, *averaged components) -> energy
    blocking_fn:    (h0, weights, *components, printQ=, final=) -> (energy, err) | None
    clean_fn:       (block energies, weights, *components, zeta=) -> (weights, *components)
    ham_basis:      the hamiltonian it measures on
    walker_kinds:   walker representations its kernel accepts
    source:         "cc" needs a CC object; "mf" takes the mean field under either
    cc_kind:        "ccsd" / "uccsd" when source == "cc": which CC object it needs
    default_guide:  the guide used when none is asked for (the corresponding HF)
    plan_chunking:  optional memory model so max_memory can be honoured
    cfg_getter:     optional (MeasOps) -> config dataclass, for dump_flags
    """

    name: str
    kind: str
    stage: Callable[..., TrialInput]
    make_data: Callable[[dict, System], Any]
    make_meas_ops: Callable[..., MeasOps]
    components: tuple[str, ...]
    energy_fn: Callable[..., Any]
    blocking_fn: Callable[..., Any]
    clean_fn: Callable[..., Any]
    ham_basis: str
    walker_kinds: frozenset[str]
    source: str
    default_guide: str
    cc_kind: str | None = None
    plan_chunking: Callable[..., Any] | None = None
    cfg_getter: Callable[[MeasOps], Any] | None = None


@dataclass(frozen=True)
class MixedRecipe:
    """
    One (guide, trial) pairing, as setup_mixed / JobMixed / run_mixed_qmc consume it.

    guide, trial:         the names
    walker_kind:          the walker representation both sides accept
    stage_trial:          pyscf obj -> TrialInput
    make_trial_data:      TrialInput.data -> trial pytree
    make_trial_meas_ops:  (sys, ...) -> MeasOps for the trial estimator
    mixed_block_fn:       per block propagate(guide) + measure(trial)
    blocking_fn:          combines the block components into (energy, stderr)
    plan_chunking:        (sys, ham_data, trial_data, ...) -> ChunkPlan, or None if the
                          estimator has no memory model and so cannot honour max_memory
    ham_basis:            the hamiltonian both halves use: "restricted" or "uchol"
    components:           names of the trial kernel's outputs, in order
    energy_fn:            (h0, *averaged components) -> energy
    clean_fn:             the block outlier filter that goes with blocking_fn
    guide_spec, trial_spec: the specs this pairing was built from

    blocking_fn, clean_fn, energy_fn and components belong together: the kernel decides
    what comes out of each block and these are the only things that know how to
    recombine it. They come from one TrialSpec, never independently.
    """

    guide: str
    trial: str
    walker_kind: WalkerKind

    stage_trial: Callable[..., TrialInput]
    make_trial_data: Callable[[dict, System], Any]
    make_trial_meas_ops: Callable[..., MeasOps]

    mixed_block_fn: MixedBlockFn
    blocking_fn: Callable[..., Any]

    plan_chunking: Callable[..., Any] | None = None
    ham_basis: str = "restricted"

    components: tuple[str, ...] = ("t2", "e0", "e1")
    energy_fn: Callable[..., Any] = pt2ccsd_energy_fn
    clean_fn: Callable[..., Any] = clean_pt2ccsd
    guide_spec: GuideSpec | None = None
    trial_spec: TrialSpec | None = None

    @property
    def key(self) -> tuple[str, str]:
        return (self.guide, self.trial)


# ------------------------------------------------------------------------------------
# guides
# ------------------------------------------------------------------------------------

GUIDES: dict[str, GuideSpec] = {
    "rhf": GuideSpec(
        name="rhf",
        kinds=frozenset({"rhf"}),
        source="mf",
        ham_bases=frozenset({"restricted"}),
        walker_kinds=frozenset({"restricted", "unrestricted", "generalized"}),
    ),
    # on the restricted hamiltonian as Afqmc(uhf) runs it, and on the unrestricted one as
    # AfqmcUh does (setup._make_trial_bundle builds the uchol bundle from the kind)
    "uhf": GuideSpec(
        name="uhf",
        kinds=frozenset({"uhf"}),
        source="mf",
        ham_bases=frozenset({"restricted", "uchol"}),
        walker_kinds=frozenset({"unrestricted", "generalized"}),
    ),
    "cisd": GuideSpec(
        name="cisd",
        kinds=frozenset({"cisd"}),
        source="cc",
        ham_bases=frozenset({"restricted"}),
        walker_kinds=frozenset({"restricted"}),
    ),
    # on the restricted hamiltonian as Afqmc(ucc) runs it; on the unrestricted one with
    # the meas.ucisd_uh kernels, each spin's CI coefficients in that spin's own basis
    "ucisd": GuideSpec(
        name="ucisd",
        kinds=frozenset({"ucisd"}),
        source="cc",
        ham_bases=frozenset({"restricted", "uchol"}),
        walker_kinds=frozenset({"unrestricted", "generalized"}),
        make_bundle={"uchol": make_ucisd_guide_bundle_uh},
    ),
}


# ------------------------------------------------------------------------------------
# trials
# ------------------------------------------------------------------------------------

_PT2_COMPONENTS: tuple[str, ...] = ("t2", "e0", "e1")
_ELOC: tuple[str, ...] = ("e_loc",)


def _pt2ccsd_trial(name: str, measure_type: str | None) -> TrialSpec:
    """
    The pt2CCSD trials differ only in which energy kernel they measure with, so the trial
    name is the kernel choice: "pt2ccsd" the plain estimator, "pt2ccsd_chunk" the chunked
    one, "pt2ccsd_bar" the chunked one with exp(T1) on the hamiltonian, and
    "pt2ccsd_sto_chol" that one again with the T2-contracted two-body sum sampled
    semistochastically.

    Only the chunking kernels get a plan_chunking hook: max_memory has nothing to size
    for the unchunked one, and setup_mixed says so rather than ignoring the budget.

    The kernel returns (t2, e0, e1) per walker and the energy is h0 + e0 + e1 - t2 * e0,
    nonlinear in the block averages, so these stay matched with pt2ccsd_blocking.
    """
    return TrialSpec(
        name=name,
        kind="pt2ccsd",
        stage=stage_pt2ccsd_trial,
        make_data=make_pt2ccsd_trial_data,
        make_meas_ops=partial(make_pt2ccsd_meas_ops, measure_type=measure_type),
        components=_PT2_COMPONENTS,
        energy_fn=pt2ccsd_energy_fn,
        blocking_fn=pt2ccsd_blocking,
        clean_fn=clean_pt2ccsd,
        ham_basis="restricted",
        walker_kinds=frozenset({"restricted"}),
        source="cc",
        cc_kind="ccsd",
        default_guide="rhf",
        plan_chunking=(
            partial(plan_chunking_for_run, measure_type=measure_type)
            if measure_type is not None
            else None
        ),
        cfg_getter=get_pt2ccsd_meas_cfg,
    )


def _upt2ccsd_trial(name: str, measure_type: str) -> TrialSpec:
    """
    The unrestricted pt2CCSD trials, on the unrestricted (uchol) hamiltonian AfqmcUh
    uses: each spin in its own MO basis, with unrestricted walkers. The kernels return
    the same (t2, e0, e1) as the restricted ones, so the statistics are shared.
    """
    return TrialSpec(
        name=name,
        kind="upt2ccsd",
        stage=stage_upt2ccsd_trial,
        make_data=make_upt2ccsd_trial_data,
        make_meas_ops=partial(make_upt2ccsd_meas_ops, measure_type=measure_type),
        components=_PT2_COMPONENTS,
        energy_fn=pt2ccsd_energy_fn,
        blocking_fn=pt2ccsd_blocking,
        clean_fn=clean_pt2ccsd,
        ham_basis="uchol",
        walker_kinds=frozenset({"unrestricted"}),
        source="cc",
        cc_kind="uccsd",
        default_guide="uhf",
        plan_chunking=partial(plan_chunking_for_run_u, measure_type=measure_type),
        cfg_getter=get_pt2ccsd_meas_cfg,
    )


def _eloc_trial(
    name: str,
    *,
    kind: str,
    stage: Callable[..., TrialInput],
    make_data: Callable[[dict, System], Any],
    make_meas_ops: Callable[..., MeasOps],
    walker_kinds: frozenset[str],
    source: str,
    default_guide: str,
    cc_kind: str | None = None,
    cfg_getter: Callable[[MeasOps], Any] | None = None,
) -> TrialSpec:
    """
    A trial whose energy kernel returns the full local energy <T|H|phi>/<T|phi> (h0
    included) as one number: the measurement is the plain ratio estimator, one
    component, identity energy_fn.
    """
    return TrialSpec(
        name=name,
        kind=kind,
        stage=stage,
        make_data=make_data,
        make_meas_ops=_adapt_meas_ops(make_meas_ops),
        components=_ELOC,
        energy_fn=eloc_energy_fn,
        blocking_fn=make_component_blocking(eloc_energy_fn, label=name),
        clean_fn=clean_components,
        ham_basis="restricted",
        walker_kinds=walker_kinds,
        source=source,
        cc_kind=cc_kind,
        default_guide=default_guide,
        cfg_getter=cfg_getter,
    )


TRIALS: dict[str, TrialSpec] = {
    **{
        name: _pt2ccsd_trial(name, measure_type)
        for name, measure_type in (
            ("pt2ccsd", None),
            ("pt2ccsd_chunk", "chunk"),
            ("pt2ccsd_bar", "bar"),
            ("pt2ccsd_sto_chol", "sto_chol"),
        )
    },
    **{
        name: _upt2ccsd_trial(name, measure_type)
        for name, measure_type in (
            ("upt2ccsd", "chunk"),
            ("upt2ccsd_bar", "bar"),
            ("upt2ccsd_sto_chol", "sto_chol"),
        )
    },
    "rhf": _eloc_trial(
        "rhf",
        kind="rhf",
        stage=_stage_mf_trial,
        make_data=make_rhf_trial_data,
        make_meas_ops=make_rhf_meas_ops,
        walker_kinds=frozenset({"restricted", "unrestricted", "generalized"}),
        source="mf",
        default_guide="rhf",
        cfg_getter=get_rhf_meas_cfg,
    ),
    "uhf": _eloc_trial(
        "uhf",
        kind="uhf",
        stage=_stage_mf_trial,
        make_data=make_uhf_trial_data,
        make_meas_ops=make_uhf_meas_ops,
        walker_kinds=frozenset({"restricted", "unrestricted", "generalized"}),
        source="mf",
        default_guide="uhf",
    ),
    "cisd": _eloc_trial(
        "cisd",
        kind="cisd",
        stage=_stage_cisd_trial,
        make_data=make_cisd_trial_data,
        make_meas_ops=make_cisd_meas_ops,
        walker_kinds=frozenset({"restricted"}),
        source="cc",
        cc_kind="ccsd",
        default_guide="rhf",
        cfg_getter=get_cisd_meas_cfg,
    ),
    "ucisd": _eloc_trial(
        "ucisd",
        kind="ucisd",
        stage=_stage_ucisd_trial,
        make_data=make_ucisd_trial_data,
        make_meas_ops=make_ucisd_meas_ops,
        walker_kinds=frozenset({"restricted", "unrestricted", "generalized"}),
        source="cc",
        cc_kind="uccsd",
        default_guide="uhf",
    ),
}


# ------------------------------------------------------------------------------------
# pairing
# ------------------------------------------------------------------------------------


def pair_recipe(guide: GuideSpec, trial: TrialSpec) -> MixedRecipe:
    """
    The MixedRecipe of one (guide, trial) pair, or a ValueError saying why the two cannot
    be paired: they must share the hamiltonian basis and at least one walker kind. The
    walker kind is the first common one in WALKER_KIND_ORDER.
    """
    if trial.ham_basis not in guide.ham_bases:
        raise ValueError(
            f"guide={guide.name!r} runs on {sorted(guide.ham_bases)} hamiltonians but "
            f"trial={trial.name!r} needs {trial.ham_basis!r}."
        )
    common = [k for k in WALKER_KIND_ORDER if k in guide.walker_kinds and k in trial.walker_kinds]
    if not common:
        raise ValueError(
            f"guide={guide.name!r} accepts walkers {sorted(guide.walker_kinds)} and "
            f"trial={trial.name!r} accepts {sorted(trial.walker_kinds)}; no walker kind in common."
        )
    return MixedRecipe(
        guide=guide.name,
        trial=trial.name,
        walker_kind=common[0],  # type: ignore[arg-type]
        stage_trial=trial.stage,
        make_trial_data=trial.make_data,
        make_trial_meas_ops=trial.make_meas_ops,
        mixed_block_fn=block_mixed,
        blocking_fn=trial.blocking_fn,
        plan_chunking=trial.plan_chunking,
        ham_basis=trial.ham_basis,
        components=trial.components,
        energy_fn=trial.energy_fn,
        clean_fn=trial.clean_fn,
        guide_spec=guide,
        trial_spec=trial,
    )


def _compatible(guide: GuideSpec, trial: TrialSpec) -> bool:
    return trial.ham_basis in guide.ham_bases and bool(guide.walker_kinds & trial.walker_kinds)


# every compatible pair, for listing; get_mixed_recipe builds the same objects on demand
MIXED_RECIPES: dict[tuple[str, str], MixedRecipe] = {
    (g.name, t.name): pair_recipe(g, t)
    for g in GUIDES.values()
    for t in TRIALS.values()
    if _compatible(g, t)
}


def _format_pairs() -> str:
    return ", ".join(f"{guide}+{trial}" for guide, trial in sorted(MIXED_RECIPES))


def get_mixed_recipe(trial: str, guide: str | None = None) -> MixedRecipe:
    """
    The (guide, trial) pipeline. guide=None takes the trial's default guide, the
    corresponding HF. An unknown name or an incompatible pair is a ValueError that lists
    what is available.
    """
    try:
        trial_spec = TRIALS[trial]
    except KeyError:
        raise ValueError(
            f"unknown mixed trial {trial!r}; available: {', '.join(sorted(TRIALS))}"
        ) from None
    guide_name = trial_spec.default_guide if guide is None else guide.lower()
    try:
        guide_spec = GUIDES[guide_name]
    except KeyError:
        raise ValueError(
            f"unknown mixed guide {guide_name!r}; available: {', '.join(sorted(GUIDES))}"
        ) from None
    try:
        return pair_recipe(guide_spec, trial_spec)
    except ValueError as e:
        raise ValueError(f"{e} Available pairs: {_format_pairs()}") from None


def available_mixed_recipes() -> tuple[tuple[str, str], ...]:
    """The compatible (guide, trial) pairs."""
    return tuple(sorted(MIXED_RECIPES))


def available_guides() -> tuple[str, ...]:
    return tuple(sorted(GUIDES))


def available_trials() -> tuple[str, ...]:
    return tuple(sorted(TRIALS))
