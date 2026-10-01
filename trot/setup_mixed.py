"""
Assembly for mixed guide/trial AFQMC.

The guide side of a mixed run is an ordinary single-bundle job: the walkers propagate
under the guide wavefunction with the guide's own hamiltonian, ops and propagator. So
setup._assemble_job (restricted hamiltonian) or setup_u.setup_uh (unrestricted one)
builds it unchanged, and JobMixed only adds the trial that the energy is measured
against, plus the cholesky chunk plan of a chunked trial.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, ClassVar, Union, cast

from jax.sharding import Mesh

from . import driver_mixed
from .core.ops import MeasOps
from .core.system import WalkerKind
from .driver_mixed import MixedQmcResult
from .meas.pt2ccsd_chunking import (
    DEVICE_MEMORY_FRACTION,
    XLA_DEVICE_MEMORY_FRACTION,
    abstract_walkers,
    device_memory_budget_bytes,
    plan_pt2ccsd_chunking_xla,
    pytree_bytes,
)
from .mixed import MixedRecipe, get_mixed_recipe
from .prop.blocks import block as default_block
from .prop.types import QmcParams, QmcParamsBase
from .setup import Job, _assemble_job, _make_params, _make_prop, _resolve_default_walker_kind
from .setup_u import setup_uh
from .staging import StagedInputs, TrialInput

MemoryBudget = Union[float, str, None]


def resolve_memory_budget(max_memory: MemoryBudget) -> tuple[str, int | None]:
    """
    (mode, budget_bytes) of a chunk plan from the max_memory argument: a number is a
    budget in MB for the analytic model; "analytic" is that model against
    DEVICE_MEMORY_FRACTION of the device memory; "xla" sizes the chunk from the compiled
    kernels against XLA_DEVICE_MEMORY_FRACTION of it; None is "xla". The budget is None
    when the backend reports no device memory (CPU).
    """
    if max_memory is None:
        max_memory = "xla"
    if isinstance(max_memory, str):
        mode = max_memory.strip().lower()
        if mode == "xla":
            return mode, device_memory_budget_bytes(XLA_DEVICE_MEMORY_FRACTION)
        if mode == "analytic":
            return mode, device_memory_budget_bytes(DEVICE_MEMORY_FRACTION)
        raise ValueError(
            f"max_memory must be a number of MB, 'analytic', 'xla' or None, got {max_memory!r}."
        )
    return "analytic", int(float(max_memory) * 1024**2)


def _cholesky_bytes(ham_data: Any) -> int:
    """The bytes of the hamiltonian's cholesky tensor(s)."""
    return sum(
        pytree_bytes(getattr(ham_data, name))
        for name in ("chol", "chol_a", "chol_b")
        if getattr(ham_data, name, None) is not None
    )


def _resident_bytes(
    job: "JobMixed", trial_meas_ops: MeasOps, *, guide_mixed_precision: bool
) -> int:
    """
    What the run keeps on the device whatever the chunking: the hamiltonian, the guide's
    data and contexts (measurement and propagation), the trial's data and context, and
    two copies of the walker population (the state and its resampled successor). The
    measurement contexts are sized by tracing their builders, not built; the propagation
    context is the cholesky tensor again, in the propagator's precision, plus small
    matrices.
    """
    import jax

    assert job.params is not None
    guide_meas_ctx = jax.eval_shape(job.meas_ops.build_meas_ctx, job.ham_data, job.trial_data)
    trial_meas_ctx = jax.eval_shape(trial_meas_ops.build_meas_ctx, job.ham_data, job.mix_trial_data)
    prop_ctx = _cholesky_bytes(job.ham_data) // (2 if guide_mixed_precision else 1)
    walkers = abstract_walkers(job.sys, int(job.params.n_walkers))
    return prop_ctx + sum(
        pytree_bytes(x)
        for x in (
            job.ham_data,
            job.trial_data,
            guide_meas_ctx,
            job.mix_trial_data,
            trial_meas_ctx,
            walkers,
            walkers,
        )
    )


def _plan_xla(
    job: "JobMixed",
    rec: MixedRecipe,
    *,
    budget_bytes: int,
    nchol_chunk: int | None,
    trial_mp: bool,
    guide_mp: bool,
    n_devices: int,
) -> Any:
    """The xla chunk plan of the job's trial and guide; None when XLA reports no sizes."""
    import jax

    from .core.ops import k_energy

    assert job.params is not None
    nchol = int(job.ham_data.nchol)

    def make_trial_meas_ops(k: int) -> MeasOps:
        return rec.make_trial_meas_ops(job.sys, mixed_precision=trial_mp, nchol_chunk=int(k))

    make_chunked = rec.guide_spec.chunked_meas_ops

    def guide_energy(k: int) -> tuple[Any, Any, Any] | None:
        ops = (
            make_chunked(job.sys, rec.ham_basis, int(k), mixed_precision=guide_mp)
            if make_chunked is not None
            else None
        )
        if ops is None:
            ops = job.meas_ops  # the guide's energy as it will run: unchunked
        if not ops.has_kernel(k_energy):
            return None
        ctx = jax.eval_shape(ops.build_meas_ctx, job.ham_data, job.trial_data)
        return ops.require_kernel(k_energy), ctx, job.trial_data

    resident = _resident_bytes(
        job, make_trial_meas_ops(nchol_chunk or nchol), guide_mixed_precision=guide_mp
    )
    return plan_pt2ccsd_chunking_xla(
        sys=job.sys,
        ham_data=job.ham_data,
        trial_data=job.mix_trial_data,
        make_trial_meas_ops=make_trial_meas_ops,
        n_walkers=int(job.params.n_walkers),
        nchol=nchol,
        budget_bytes=int(budget_bytes),
        resident_bytes=resident,
        n_chunks=int(job.params.n_chunks),
        nchol_chunk=nchol_chunk,
        n_devices=n_devices,
        guide_energy=guide_energy,
    )


def _walker_devices(mesh: Mesh | None) -> int:
    """
    How many devices the walker axis is spread over. Walkers are sharded on the "data"
    axis, so that is what divides the population; a mesh without one is treated as whole.
    """
    if mesh is None:
        return 1
    return int(mesh.shape.get("data", mesh.size))


@dataclass
class JobMixed(Job):
    """
    A fully assembled mixed guide/trial AFQMC run bundle.

    The inherited fields carry the GUIDE: trial_data, trial_ops, meas_ops and prop_ops
    are the guide wavefunction's. The trial that the energy is measured against lives in
    mix_trial_data / mix_trial_meas_ops.
    """

    recipe: MixedRecipe = None  # type: ignore[assignment]
    mix_trial_data: Any = None
    mix_trial_meas_ops: MeasOps = None  # type: ignore[assignment]
    # how the memory budget was split between the two chunking knobs, None without a plan
    chunk_plan: Any = None
    # cholesky vectors per step of the guide's local energy, None when it sums them at once
    guide_nchol_chunk: int | None = None
    _runtime_mix_meas_ctx: object | None = field(default=None, init=False, repr=False)

    params_cls: ClassVar[type[QmcParamsBase]] = QmcParams
    driver_fn: ClassVar[Callable[..., Any]] = staticmethod(driver_mixed.run_mixed_qmc)

    def mix_meas_ctx(self) -> Any:
        if self._runtime_mix_meas_ctx is None:
            self._runtime_mix_meas_ctx = self.mix_trial_meas_ops.build_meas_ctx(
                self.ham_data, self.mix_trial_data
            )
        return self._runtime_mix_meas_ctx

    def kernel(self, **driver_kwargs: Any) -> MixedQmcResult:
        """
        Run mixed AFQMC: propagate with the guide, measure with the trial.

        Job.prepare_runtime is deliberately not used here: the runtime layouts may compact
        the hamiltonian's cholesky tensor to a zero sized placeholder once the guide's
        contexts are built, but the trial kernels read it. The mixed driver builds all
        three contexts itself from the uncompacted hamiltonian.
        """
        assert isinstance(self.params, self.params_cls)
        driver_kwargs.setdefault("mesh", self.mesh)
        driver_kwargs.setdefault("trial_meas_ctx", self.mix_meas_ctx())
        return self.driver_fn(
            sys=self.sys,
            params=self.params,
            ham_data=self.ham_data,
            # guide: propagation
            guide_data=self.trial_data,
            guide_ops=self.trial_ops,
            guide_meas_ops=self.meas_ops,
            guide_prop_ops=self.prop_ops,
            # trial: measurement
            trial_data=self.mix_trial_data,
            trial_meas_ops=self.mix_trial_meas_ops,
            mix_block_fn=self.recipe.mixed_block_fn,
            components=self.recipe.components,
            energy_fn=self.recipe.energy_fn,
            trial_name=self.recipe.trial,
            **driver_kwargs,
        )


def setup_mixed(
    obj_or_staged: Union[Any, StagedInputs, str, Path],
    *,
    # the trial half, staged separately from the guide
    recipe: MixedRecipe | str = "pt2ccsd",
    trial_input: TrialInput | None = None,
    nchol_chunk: int | None = None,
    max_memory: MemoryBudget = None,
    # staging options (used only if we need to stage)
    norb_frozen_core: Any = None,
    chol_cut: float = 1e-5,
    cache: Union[str, Path] | None = None,
    overwrite: bool = False,
    verbose: bool = False,
    # system/prop options
    walker_kind: WalkerKind | None = None,
    mesh: Mesh | None = None,
    mixed_precision: bool = True,
    trial_mixed_precision: bool | None = None,
    # params options
    params: QmcParams | None = None,
    # overrides for customized runs
    trial_data: Any = None,
    trial_ops: Any = None,
    meas_ops: Any = None,
    prop_ops: Any = None,
    block_fn: Callable[..., Any] | None = None,
    # extra kwargs
    params_kwargs: dict[str, Any] | None = None,
    prop_kwargs: dict[str, Any] | None = None,
) -> JobMixed:
    """
    Assemble a runnable mixed AFQMC Job.

    obj_or_staged supplies the GUIDE (a mean field or a StagedInputs with the guide's
    hamiltonian and trial input). trial_input supplies the measurement trial and must be
    staged already (the recipe's stage_trial), since the trial comes from the CC object
    while the guide may come from the mean field under it.

    mixed_precision applies to the guide (its propagator and kernels) and, unless
    trial_mixed_precision is given, to the trial estimator; trial_mixed_precision sets the
    trial's precision on its own, so the two sides can be studied separately.

    The cholesky chunk of a chunked trial (pt2ccsd_bar, upt2ccsd, upt2ccsd_bar) is sized
    against a memory budget (resolve_memory_budget): "xla" (the default, None) for a plan
    read from the compiled kernels (pt2ccsd_chunking's plan_pt2ccsd_chunking_xla) against
    a share of the device memory, or "analytic" for the recipe's memory model against a
    smaller share, or max_memory in MB for that model; DEFAULT_NCHOL_CHUNK when the
    backend reports no device memory. The guide's local energy then runs with the same
    cholesky chunk (GuideSpec.chunked_meas_ops). The plan can raise params.n_chunks (the
    walker chunk count), never lower it, and the driver's own automatic walker chunking
    may raise it further after compiling. nchol_chunk, when given, is taken as fixed and
    only n_chunks is derived.

    Basic usage is through AfqmcMixed rather than this function directly.
    """
    rec = get_mixed_recipe(recipe) if isinstance(recipe, str) else recipe
    trial_mp = mixed_precision if trial_mixed_precision is None else bool(trial_mixed_precision)
    if trial_input is None:
        raise ValueError(
            "setup_mixed needs a staged trial_input; the measurement trial comes from the "
            "CC object, so it cannot be inferred from the guide's staged inputs."
        )
    if trial_input.kind != rec.trial_spec.kind:
        raise ValueError(
            f"trial={rec.trial!r} needs a TrialInput of kind {rec.trial_spec.kind!r}, got "
            f"{trial_input.kind!r}."
        )

    wk = walker_kind or cast(WalkerKind, rec.walker_kind)
    if rec.ham_basis == "uchol":
        job = setup_uh(
            obj_or_staged,
            norb_frozen_core=norb_frozen_core,
            chol_cut=chol_cut,
            cache=cache,
            overwrite=overwrite,
            verbose=verbose,
            walker_kind=wk,
            mesh=mesh,
            mixed_precision=mixed_precision,
            params=params,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
            params_kwargs=params_kwargs,
            prop_kwargs=prop_kwargs,
            job_cls=JobMixed,
        )
    else:
        job = _assemble_job(
            obj_or_staged,
            norb_frozen_core=norb_frozen_core,
            chol_cut=chol_cut,
            cache=cache,
            overwrite=overwrite,
            verbose=verbose,
            walker_kind=wk,
            mesh=mesh,
            mixed_precision=mixed_precision,
            params=params,
            trial_data=trial_data,
            trial_ops=trial_ops,
            meas_ops=meas_ops,
            prop_ops=prop_ops,
            block_fn=block_fn,
            params_kwargs=params_kwargs,
            prop_kwargs=prop_kwargs,
            params_builder=_make_params,
            prop_builder=_make_prop,
            default_block_fn=default_block,
            job_cls=JobMixed,
            walker_kind_resolver=_resolve_default_walker_kind,
        )
    job = cast(JobMixed, job)

    # attach the measurement trial
    job.recipe = rec
    job.mix_trial_data = rec.make_trial_data(trial_input.data, job.sys)

    # size the cholesky chunk of a chunked trial against the memory budget
    if rec.plan_chunking is not None:
        assert job.params is not None
        mode, budget = resolve_memory_budget(max_memory)
        if budget is not None:
            plan = None
            if mode == "xla":
                plan = _plan_xla(
                    job,
                    rec,
                    budget_bytes=budget,
                    nchol_chunk=nchol_chunk,
                    trial_mp=trial_mp,
                    guide_mp=bool(mixed_precision),
                    n_devices=_walker_devices(mesh),
                )
                if plan is None:
                    print(
                        "[setup] the backend reports no compiled memory sizes; the chunk plan "
                        "falls back to the analytic model"
                    )
            if plan is None:
                plan = rec.plan_chunking(
                    job.sys,
                    job.ham_data,
                    job.mix_trial_data,
                    n_walkers=job.params.n_walkers,
                    budget_bytes=budget,
                    n_chunks=job.params.n_chunks,
                    nchol_chunk=nchol_chunk,
                    mixed_precision=trial_mp,
                    n_devices=_walker_devices(mesh),
                )
            nchol_chunk = plan.nchol_chunk
            job.chunk_plan = plan
            # the plan only ever raises n_chunks, so this cannot undercut a caller's setting
            if plan.n_chunks != job.params.n_chunks:
                job.params = replace(job.params, n_chunks=plan.n_chunks)
    elif max_memory is not None and not (
        isinstance(max_memory, str) and max_memory.strip().lower() == "analytic"
    ):
        raise ValueError(
            f"trial {rec.trial!r} scans one cholesky vector at a time and has no memory "
            "model, so there is nothing for max_memory to size; use the chunked "
            "'pt2ccsd_bar' trial, or drop max_memory."
        )

    job.mix_trial_meas_ops = rec.make_trial_meas_ops(
        job.sys, mixed_precision=trial_mp, nchol_chunk=nchol_chunk
    )

    # The guide's local energy sums over the same cholesky index, with the same walkers in
    # flight, and holds less per walker per cholesky vector than the trial's kernel (a
    # (nocc, nocc) block of L.G against the trial's (nocc, norb) and larger ones). The
    # two run one after the other within a block, so the trial's plan bounds the guide
    # energy as well: run it with the trial's cholesky chunk. A guide meas_ops passed in
    # by the caller is kept as given.
    make_chunked = rec.guide_spec.chunked_meas_ops
    if nchol_chunk is not None and meas_ops is None and make_chunked is not None:
        chunked = make_chunked(
            job.sys, rec.ham_basis, int(nchol_chunk), mixed_precision=bool(mixed_precision)
        )
        if chunked is not None:
            job.meas_ops = chunked
            job.guide_nchol_chunk = int(nchol_chunk)
    return job
