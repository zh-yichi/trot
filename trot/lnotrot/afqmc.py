from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Union, cast

import numpy as np
from numpy.typing import NDArray

from ..afqmc import Afqmc, AfqmcMixed, banner_afqmc
from ..core.system import WalkerKind
from ..runtime_provenance import print_runtime_provenance
from ..setup_mixed import JobMixed, setup_mixed
from ..staging import StagedInputs, TrialInput
from ..staging import stage as stage_inputs
from . import io as lno_io
from .driver import FragQmcResult, run_frag_qmc
from .mixed import LnoMixedRecipe, get_mixed_recipe
from .staging import LnoFragData, dump_frag, frag_mf, load_frag

print = partial(print, flush=True)

_LNO_DEFAULT_N_BLOCKS = 300
_ALLOCATOR = "XLA_PYTHON_CLIENT_ALLOCATOR"


def _device_bytes_in_use() -> int | None:
    """
    Bytes this process holds on the default device. jax reports them through the pool
    allocator only; with the platform allocator (what the fragment loop wants) the
    driver is asked instead, through nvidia-smi's per-process table.
    """
    try:
        import jax

        stats = jax.devices()[0].memory_stats()
        if stats and "bytes_in_use" in stats:
            return int(stats["bytes_in_use"])
    except Exception:
        pass
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
        for line in out.splitlines():
            pid, used = (x.strip() for x in line.split(",")[:2])
            if pid.isdigit() and int(pid) == os.getpid():
                return int(used) * 1024**2
    except Exception:
        pass
    return None


def _release_device() -> None:
    """Hand the fragment's device memory back: drop the compiled kernels, collect."""
    import jax

    jax.clear_caches()
    gc.collect()


# ======================================================================================
# one fragment
# ======================================================================================


class LnoFragMixed(AfqmcMixed):
    """
    The AFQMC of one LNO fragment: an AfqmcMixed run on the fragment hamiltonian, the
    guide staged by trot from the mean field in the fragment's LNO basis, and the fragment
    energy measured against the pt2CCSD trial.

        frag = LnoFragMixed(mf, frag_data, trial="pt2ccsd", max_error=1e-4)
        e_frag, e_frag_err = frag.kernel()

    or from a file that LnoAfqmcMixed(save_frag_data=...) wrote, with no mf and no LNO,
    CCSD or integral work:

        frag = LnoFragMixed.from_frag_data("frag_data/frag3.h5", n_blocks=400)

    kernel() returns the fragment correlation energy and its error, and keeps the
    FragQmcResult on .qmc_result. The guide energy is on .guide_e_tot / .guide_e_err.

    Parameters
    ----------
    mf : pyscf RHF/UHF, density fitted. None when built from a fragment file.
    frag : LnoFragData, the CPU stage of the fragment (LNOs, amplitudes).
    trial, guide : the recipe (lnotrot.mixed); defaults "pt2ccsd" (RHF) / "upt2ccsd" (UHF)
        with the HF guide.
    max_error : early-stop target of the fragment error; None runs all n_blocks.
    stop_ratio, min_blocks : stop once err < stop_ratio * max_error and at least
        min_blocks sampling blocks are in (0.7, 120 as in afqmc).
    The remaining keywords are AfqmcMixed's (memory_mode, max_memory, nchol_chunk,
    trial_kwargs, mixed_precision, chol_cut, n_eql_blocks, n_blocks, seed, dt,
    n_prop_steps, n_walkers, n_chunks). n_blocks defaults to 300.
    """

    def __init__(
        self,
        mf: Any,
        frag: LnoFragData,
        *,
        trial: str | None = None,
        guide: str | None = None,
        max_error: float | None = None,
        stop_ratio: float = 0.7,
        min_blocks: int = 120,
        outlier_zeta: float = 20.0,
        staged: StagedInputs | None = None,
        emf: float | None = None,
        memory_mode: str = "low",
        max_memory: float | None = None,
        nchol_chunk: int | None = None,
        trial_kwargs: dict[str, Any] | None = None,
        mixed_precision: bool = False,
        chol_cut: float = 1e-5,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_prop_steps: int | None = None,
        n_walkers: int | None = None,
        n_chunks: int | None = None,
    ):
        # Afqmc.__init__ directly: there is no CC object, the trial comes from frag
        Afqmc.__init__(
            self,
            mf,
            norb_frozen_core=None,
            chol_cut=chol_cut,
            cache=None,
            n_eql_blocks=n_eql_blocks,
            n_blocks=_LNO_DEFAULT_N_BLOCKS if n_blocks is None else n_blocks,
            seed=seed,
            dt=dt,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )
        defaults = self.params_cls()
        self.n_prop_steps = defaults.n_prop_steps if n_prop_steps is None else n_prop_steps

        self.frag = frag
        if trial is None:
            trial = "upt2ccsd" if frag.unrestricted else "pt2ccsd"
        self.recipe: LnoMixedRecipe = get_mixed_recipe(trial, guide)
        self.trial: str = self.recipe.trial
        self.guide: str = self.recipe.guide
        if (self.recipe.ham_basis == "uchol") != frag.unrestricted:
            kind = "unrestricted" if frag.unrestricted else "restricted"
            raise ValueError(f"trial={self.trial!r} does not fit {kind} fragment data.")
        if self.recipe.needs_amplitudes and not frag.has_amplitudes:
            raise ValueError(
                f"trial={self.trial!r} needs the fragment CCSD amplitudes, but the fragment "
                "data has none (run_cc=False?)."
            )
        self.basis_a = None
        self.basis_b = None
        self.walker_kind = cast(WalkerKind, self.recipe.walker_kind)
        self.mixed_precision = mixed_precision
        self.memory_mode = memory_mode
        self.max_memory = max_memory
        self.nchol_chunk = nchol_chunk
        self.extra_trial_kwargs: dict[str, Any] = dict(trial_kwargs or {})

        self.max_error = max_error
        self.stop_ratio = float(stop_ratio)
        self.min_blocks = int(min_blocks)
        self.outlier_zeta = float(outlier_zeta)
        self.emf = float(mf.e_tot) if (emf is None and mf is not None) else emf

        self._trial_input: TrialInput | None = None
        self._preloaded_staged = staged
        self.guide_e_tot: Any = None
        self.guide_e_err: Any = None
        self.qmc_result: FragQmcResult | None = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_frag_data(cls, path: Union[str, Path], *, mf: Any = None, **kwargs: Any) -> "LnoFragMixed":
        """
        Re-run one fragment from the file LnoAfqmcMixed(save_frag_data=...) wrote. The
        file carries the fragment hamiltonian and the staged guide, so no mf is needed;
        pass one only to re-stage (a different guide, say).
        """
        frag, staged, attrs = load_frag(path)
        if mf is not None:
            fp = attrs["fingerprint"]
            if fp["nao"] != mf.mol.nao or abs(fp["e_tot"] - float(mf.e_tot)) > 1e-8:
                raise ValueError(f"{path} was written for a different system than mf.")
            staged = None
        return cls(mf, frag, staged=staged, emf=attrs["emf"], **kwargs)

    def save(self, path: Union[str, Path]) -> Path:
        """Write the self-contained fragment file (see staging.dump_frag)."""
        staged = self.stage()
        return dump_frag(path, self.frag, staged, self._scf, emf=self.emf)

    def _key(self) -> tuple:
        return super()._key() + (id(self.frag), self.trial, self.guide)

    # ------------------------------------------------------------------ staging

    def stage(self, *, force: bool = False) -> StagedInputs:
        """
        The fragment hamiltonian (lnotrot.integral) and the guide, staged by trot from the
        mean field in the fragment basis, as AfqmcMixed.stage does from cc._scf.
        """
        key = self._key()
        if self._staged is not None and self._cache_key == key and not force:
            return self._staged

        frag = self.frag
        if self._preloaded_staged is not None:
            staged = self._preloaded_staged
        else:
            if self._scf is None:
                raise ValueError("LnoFragMixed needs mf to build the fragment hamiltonian.")
            ham = self.recipe.build_ham(self._scf, frag.lno_coeff, frag.lno_frozen, chol_cut=self.chol_cut)
            frozen = frag.lno_frozen[0] if frag.unrestricted else frag.lno_frozen
            frozen = np.asarray(frozen, dtype=np.int64).reshape(-1) if not isinstance(frozen, int) else None
            staged = stage_inputs(
                frag_mf(self._scf, frag),
                frozen_orbitals=frozen if frozen is not None and frozen.size else None,
                norb_frozen_core=0 if (frozen is None or frozen.size == 0) else None,
                chol_cut=self.chol_cut,
                verbose=self.verbose,
                ham=cast(Any, ham),
            )
        if staged.trial.kind != self.guide:
            raise ValueError(
                f"guide={self.guide!r} was requested but the mean field stages as "
                f"{staged.trial.kind!r} in the fragment basis."
            )
        self._trial_input = self.recipe.stage_trial(frag)

        self._staged = staged
        self._cache_key = key
        self._job = None
        return staged

    # ------------------------------------------------------------------ run

    def dump_flags(self, job: JobMixed) -> None:
        frag = self.frag
        print(f"\n******** LNO fragment {frag.frag_idx + 1} [{frag.frag_name}] ********")
        print(f" nactocc         = {frag.nactocc}")
        print(f" nactvir         = {frag.nactvir}")
        print(f" nfrozen (LNO)   = {np.size(frag.lno_frozen[0]) if frag.unrestricted else np.size(frag.lno_frozen)}")
        print(f" lno_thresh      = {frag.lno_thresh}")
        print(f" E(LNO-MP2)      = {frag.efrag_mp:.8f}")
        print(f" E(LNO-CCSD)     = {frag.efrag_cc:.8f}")
        print(f" max_error       = {self.max_error}")
        print(f" stop_ratio      = {self.stop_ratio}  min_blocks = {self.min_blocks}")
        super().dump_flags(job)

    def kernel(self, **driver_kwargs: Any) -> tuple[float, float]:
        """Run the fragment AFQMC; returns (e_frag, e_frag_err) as host floats."""
        mesh = driver_kwargs.pop("mesh", None)
        job = self.build_job(mesh=mesh)
        self.dump_flags(job)

        result = run_frag_qmc(
            sys=job.sys,
            params=job.params,
            ham_data=job.ham_data,
            guide_data=job.trial_data,
            guide_ops=job.trial_ops,
            guide_prop_ops=job.prop_ops,
            guide_meas_ops=job.meas_ops,
            trial_data=job.mix_trial_data,
            trial_meas_ops=job.mix_trial_meas_ops,
            trial_meas_ctx=job.mix_meas_ctx(),
            mix_block_fn=self.recipe.mixed_block_fn,
            blocking_fn=self.recipe.blocking_fn,
            clean_fn=self.recipe.clean_fn,
            components=self.recipe.components,
            max_error=self.max_error,
            stop_ratio=self.stop_ratio,
            min_blocks=self.min_blocks,
            outlier_zeta=self.outlier_zeta,
            mesh=mesh,
            label=f"frag {self.frag.frag_idx + 1}",
            **driver_kwargs,
        )
        self.qmc_result = result
        self.guide_e_tot = result.guide_mean_energy
        self.guide_e_err = result.guide_stderr_energy
        self.e_tot = result.frag_mean_energy
        self.e_err = result.frag_stderr_energy
        return self.e_tot, self.e_err

    run = kernel


# ======================================================================================
# the fragment loop
# ======================================================================================


class LnoAfqmcMixed:
    """
    LNO-AFQMC: afqmc's lno_afqmc.run_afqmc as an object.

        lno = LnoAfqmcMixed(mf, lo_coeff, frag_list, frag_name=frag_name, lno_thresh=1e-5,
                            trial="pt2ccsd", target_error=1e-3, seed=17)
        e_qmc, e_qmc_err = lno.kernel()

    kernel() runs, for every fragment in run_frag, 1) make_las 2) LNO-MP2 3) LNO-CCSD
    (when the recipe needs amplitudes) 4) LNO-AFQMC, with steps 1-3 of the next fragment
    on a background thread while step 4 occupies the device. It returns the LNO-AFQMC
    correlation energy and its error; everything else is on the object:

        run_frag, frag_name               fragments run and their names
        lno_size, lno_nocc                active orbitals / occupied per fragment
        lno_emp, lno_ecc                  LNO-MP2 / LNO-CCSD fragment energies
        lno_eqmc, lno_eqmc_err            LNO-AFQMC fragment energies and errors
        lno_cc_time, lno_wait_time, lno_qmc_time
        e_mp, e_cc, e_qmc, e_qmc_err, lno_max

    filled as each fragment finishes.

    Parameters (LNO, as run_afqmc)
    ----------
    mf : density fitted pyscf RHF/UHF
    lo_coeff, frag_list, frag_name : from lnotrot.fragments.iao_fragment
    lno_thresh : float (-> [10 x, x]) or [occ, vir]
    nfrozen : frozen core count, default chemcore
    run_frag : fragment indices to run (0-based), default all
    run_mp, run_cc, run_qmc : which steps to run; run_cc=None follows the recipe
    pipeline, prefetch : run the CPU stage ahead on a thread, and how far ahead
    frag_output, lno_output : optional per-fragment logs / results table (lnotrot.io)
    save_frag_data : directory for the self-contained frag{i}.h5 files
    isolate : run each fragment's AFQMC in a child process (python -m trot.lnotrot.run_frag)
    keep_qmc_results : keep every fragment's FragQmcResult in frag_qmc_results
    debug_memory : raise if device memory does not return to baseline after a fragment

    Parameters (AFQMC, as AfqmcMixed)
    ----------
    trial, guide : the recipe; default "pt2ccsd" / "upt2ccsd" from mf with the HF guide
    target_error : the target error of the total; each fragment stops early once its
        error is below stop_ratio * target_error / sqrt(nfrag)
    chol_cut, memory_mode, max_memory, nchol_chunk, trial_kwargs, mixed_precision,
    n_eql_blocks, n_blocks (default 300), seed, dt, n_prop_steps, n_walkers, n_chunks
    """

    def __init__(
        self,
        mf: Any,
        lo_coeff: Any,
        frag_list: Any,
        *,
        frag_name: Any = None,
        lno_thresh: Any = 1e-6,
        nfrozen: int | None = None,
        run_frag: Any = None,
        run_mp: bool = True,
        run_cc: bool | None = None,
        run_qmc: bool = True,
        pipeline: bool = True,
        prefetch: int = 1,
        frag_output: Union[str, Path] | None = None,
        lno_output: Union[str, Path] | None = None,
        save_frag_data: Union[str, Path] | None = None,
        isolate: bool = False,
        keep_qmc_results: bool = False,
        debug_memory: bool = False,
        memory_tolerance_mb: float = 256.0,
        # AFQMC
        trial: str | None = None,
        guide: str | None = None,
        target_error: float | None = None,
        stop_ratio: float = 0.7,
        min_blocks: int = 120,
        chol_cut: float = 1e-5,
        memory_mode: str = "low",
        max_memory: float | None = None,
        nchol_chunk: int | None = None,
        trial_kwargs: dict[str, Any] | None = None,
        mixed_precision: bool = False,
        n_eql_blocks: int | None = None,
        n_blocks: int | None = None,
        seed: int | None = None,
        dt: float | None = None,
        n_prop_steps: int | None = None,
        n_walkers: int | None = None,
        n_chunks: int | None = None,
    ):
        from pyscf import scf
        from pyscf.data import elements

        from .las import check_span

        if getattr(mf, "with_df", None) is None:
            raise NotImplementedError(
                "LNO-AFQMC builds the fragment integrals from the density fitting tensor; "
                "use a density fitted mean field (mf.density_fit())."
            )
        self._scf = mf
        self.unrestricted = isinstance(mf, scf.uhf.UHF)
        if not self.unrestricted and not isinstance(mf, scf.rhf.RHF):
            raise TypeError(f"unsupported mean-field type: {type(mf)}")

        self.lo_coeff = lo_coeff
        self.frag_list = list(frag_list)
        self.nfrag_tot = len(self.frag_list)
        self.frag_name_all = (
            [str(n) for n in frag_name] if frag_name is not None else [f"frag{i}" for i in range(self.nfrag_tot)]
        )
        if len(self.frag_name_all) != self.nfrag_tot:
            raise ValueError("frag_name and frag_list have different lengths")

        if nfrozen is None:
            nfrozen = int(elements.chemcore(mf.mol))
            print("LNO freezes at least the chemcore orbitals for each element.")
        self.nfrozen = int(nfrozen)
        self.lno_thresh_in = lno_thresh
        self.lno_type = ["1h", "1h"]

        if trial is None:
            trial = "upt2ccsd" if self.unrestricted else "pt2ccsd"
        self.recipe: LnoMixedRecipe = get_mixed_recipe(trial, guide)
        self.trial = self.recipe.trial
        self.guide = self.recipe.guide
        if (self.recipe.ham_basis == "uchol") != self.unrestricted:
            raise ValueError(
                f"trial={self.trial!r} needs {'a UHF' if self.recipe.ham_basis == 'uchol' else 'an RHF'} "
                f"mean field, got {type(mf).__name__}."
            )

        self.run_mp = bool(run_mp)
        self.run_cc = bool(self.recipe.needs_amplitudes) if run_cc is None else bool(run_cc)
        self.run_qmc = bool(run_qmc)
        if self.run_qmc and self.recipe.needs_amplitudes and not self.run_cc:
            raise ValueError(
                f"run_qmc=True with trial={self.trial!r} requires run_cc=True: the trial needs "
                "the fragment t1/t2 amplitudes."
            )

        self.run_frag = self._resolve_run_frag(run_frag)
        self.pipeline = bool(pipeline)
        self.prefetch = int(prefetch)
        self.frag_output = frag_output
        self.lno_output = lno_output
        self.save_frag_data = save_frag_data
        self.isolate = bool(isolate)
        self.keep_qmc_results = bool(keep_qmc_results)
        self.debug_memory = bool(debug_memory)
        self.memory_tolerance_mb = float(memory_tolerance_mb)

        self.target_error = target_error
        self.stop_ratio = float(stop_ratio)
        self.min_blocks = int(min_blocks)
        self.qmc_kwargs: dict[str, Any] = dict(
            chol_cut=chol_cut,
            memory_mode=memory_mode,
            max_memory=max_memory,
            nchol_chunk=nchol_chunk,
            trial_kwargs=trial_kwargs,
            mixed_precision=mixed_precision,
            n_eql_blocks=n_eql_blocks,
            n_blocks=n_blocks,
            dt=dt,
            n_prop_steps=n_prop_steps,
            n_walkers=n_walkers,
            n_chunks=n_chunks,
        )
        self.seed = int(np.random.randint(1, 2**31 - 1)) if seed is None else int(seed)

        # the LOs must span the occupied MOs outside the core; not the other way round
        check_span(mf, lo_coeff, self.nfrozen, thresh=1e-6)

        self._reset_results()

        if os.environ.get(_ALLOCATOR) != "platform":
            warnings.warn(
                f"{_ALLOCATOR} is {os.environ.get(_ALLOCATOR)!r}, not 'platform': jax keeps a memory "
                "pool, so device memory may not return to the driver between fragments. Import "
                "trot.lnotrot before jax, or set the variable, for long fragment loops.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ bookkeeping

    def _resolve_run_frag(self, run_frag: Any) -> list[int]:
        if run_frag is None:
            return list(range(self.nfrag_tot))
        run_frag = [int(i) for i in run_frag]
        dup = sorted({i for i in run_frag if run_frag.count(i) > 1})
        if dup:
            raise ValueError(f"run_frag contains duplicate fragment indices: {dup} (run_frag = {run_frag})")
        bad = sorted({i for i in run_frag if not 0 <= i < self.nfrag_tot})
        if bad:
            raise ValueError(
                f"run_frag contains out-of-range fragment indices: {bad} "
                f"(valid range 0 ... {self.nfrag_tot - 1} for {self.nfrag_tot} fragments)"
            )
        return run_frag

    def _reset_results(self) -> None:
        n = len(self.run_frag)
        self.frag_name = [self.frag_name_all[i] for i in self.run_frag]
        self.lno_size: list[Any] = [None] * n
        self.lno_nocc: list[Any] = [None] * n
        self.lno_emp = np.zeros(n)
        self.lno_ecc = np.zeros(n)
        self.lno_eqmc = np.zeros(n)
        self.lno_eqmc_err = np.zeros(n)
        self.lno_cc_time = np.zeros(n)
        self.lno_wait_time = np.zeros(n)
        self.lno_qmc_time = np.zeros(n)
        self.frag_qmc_results: list[FragQmcResult | None] = [None] * n
        self.frag_data: list[LnoFragData | None] = [None] * n
        self.n_done = 0
        self.e_mp = self.e_cc = self.e_qmc = self.e_qmc_err = 0.0
        self.lno_max = 0
        self.loop_time = 0.0

    def _update_totals(self) -> None:
        k = self.n_done
        self.e_mp = float(np.sum(self.lno_emp[:k]))
        self.e_cc = float(np.sum(self.lno_ecc[:k]))
        self.e_qmc = float(np.sum(self.lno_eqmc[:k]))
        self.e_qmc_err = float(np.sqrt(np.sum(self.lno_eqmc_err[:k] ** 2)))
        self.lno_max = max((int(np.max(s)) for s in self.lno_size[:k] if s is not None), default=0)

    @property
    def max_error(self) -> float | None:
        """The per-fragment early-stop target, target_error / sqrt(nfrag_tot)."""
        if self.target_error is None:
            return None
        return float(self.target_error) / np.sqrt(self.nfrag_tot)

    # ------------------------------------------------------------------ the fragment AFQMC

    def _frag_seed(self, frag_idx: int) -> int:
        return int(self._seeds[frag_idx])

    def _frag_path(self, base: Union[str, Path] | None, frag_idx: int, stem: str) -> Path | None:
        if base is None:
            return None
        return Path(base) / f"{stem}{frag_idx + 1}.h5"

    def lnoafqmc_kernel(self, frag: LnoFragData) -> tuple[float, float, float, FragQmcResult | None]:
        """
        The device stage of one fragment: build the fragment hamiltonian, run the AFQMC,
        and release everything it put on the device before returning host floats.
        """
        frag_idx = frag.frag_idx
        t0 = time.perf_counter()
        baseline = _device_bytes_in_use()
        result: FragQmcResult | None = None
        try:
            if self.isolate:
                e, err, result = self._run_isolated(frag)
            else:
                fm = LnoFragMixed(
                    self._scf,
                    frag,
                    trial=self.trial,
                    guide=self.guide,
                    max_error=self.max_error,
                    stop_ratio=self.stop_ratio,
                    min_blocks=self.min_blocks,
                    seed=self._frag_seed(frag_idx),
                    **self.qmc_kwargs,
                )
                path = self._frag_path(self.save_frag_data, frag_idx, "frag")
                if path is not None:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fm.save(path)
                    print(f"fragment data written to {path}")
                e, err = fm.kernel()
                result = fm.qmc_result
                del fm
        finally:
            _release_device()
            now = _device_bytes_in_use()
            if baseline is not None and now is not None:
                mb = (now - baseline) / 1024**2
                print(f"device memory: {now / 1024**2:.1f} MB in use ({mb:+.1f} MB vs before the fragment)")
                if self.debug_memory and mb > self.memory_tolerance_mb:
                    raise RuntimeError(
                        f"device memory grew by {mb:.1f} MB over fragment {frag_idx + 1} "
                        f"(tolerance {self.memory_tolerance_mb} MB)"
                    )
        return float(e), float(err), time.perf_counter() - t0, result

    def _run_isolated(self, frag: LnoFragData) -> tuple[float, float, FragQmcResult | None]:
        """Run the fragment in a child process from its self-contained file."""
        base = Path(self.save_frag_data) if self.save_frag_data is not None else Path("frag_data")
        base.mkdir(parents=True, exist_ok=True)
        path = base / f"frag{frag.frag_idx + 1}.h5"
        fm = LnoFragMixed(self._scf, frag, trial=self.trial, guide=self.guide, **self.qmc_kwargs)
        fm.save(path)
        del fm
        _release_device()

        opts = dict(self.qmc_kwargs)
        opts.update(
            trial=self.trial,
            guide=self.guide,
            max_error=self.max_error,
            stop_ratio=self.stop_ratio,
            min_blocks=self.min_blocks,
            seed=self._frag_seed(frag.frag_idx),
        )
        opts_path = base / f"frag{frag.frag_idx + 1}.opts.json"
        out_path = base / f"frag{frag.frag_idx + 1}.result.json"
        opts_path.write_text(json.dumps(opts))
        cmd = [sys.executable, "-m", "trot.lnotrot.run_frag", str(path), "--options", str(opts_path), "--out", str(out_path)]
        print(f"running fragment {frag.frag_idx + 1} in a child process: {' '.join(cmd)}")
        env = dict(os.environ)
        env.setdefault(_ALLOCATOR, "platform")
        proc = subprocess.run(cmd, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"fragment {frag.frag_idx + 1} failed in the child process (exit {proc.returncode})")
        res = json.loads(out_path.read_text())
        return float(res["e_frag"]), float(res["e_frag_err"]), None

    # ------------------------------------------------------------------ kernel

    def kernel(self) -> tuple[float, float]:
        from pyscf import lib

        from . import solvers
        from .pipeline import CpuPipeline, cpu_stage

        mf = self._scf
        print(banner_afqmc())
        print_runtime_provenance()
        print("\n ******* LNO-AFQMC (lnotrot) ******* \n")
        print(f"LNO THRESHOLD = {self.lno_thresh_in}")
        print(f"trial = {self.trial}  guide = {self.guide}")

        mlno = solvers.get_lnoccsd(mf, self.lo_coeff, self.frag_list, self.nfrozen, self.lno_thresh_in)
        lno_thresh = mlno.lno_thresh
        eris = mlno.ao2mo()

        run_frag = self.run_frag
        nfrag_run = len(run_frag)
        print(f"Run fragments {run_frag} ({nfrag_run} of {self.nfrag_tot})")
        self._reset_results()
        self._seeds = np.random.default_rng(self.seed).integers(1, 2**31 - 1, size=self.nfrag_tot)
        if self.max_error is not None:
            print(f"target_error = {self.target_error:g}  ->  per fragment max_error = {self.max_error:.3e}")

        lno_pct_occ = [None, None]
        lno_norb = [[None, None]] * self.nfrag_tot

        depth = int(self.prefetch) if self.pipeline else 0
        depth = max(0, min(depth, max(nfrag_run - 1, 0)))
        print(f"\nCPU/GPU pipeline: {'ON' if depth > 0 else 'OFF'} (prefetch depth {depth})")

        def _cpu_task(i: int) -> LnoFragData:
            fi = run_frag[i]
            return cpu_stage(
                mlno, mf, self.lo_coeff, self.frag_list[fi], lno_thresh, lno_pct_occ, lno_norb,
                self.lno_type, eris, i, fi, self.frag_name_all[fi], self.run_mp, self.run_cc, self.nfrozen,
            )

        pipe = CpuPipeline(_cpu_task, nfrag_run, depth)
        loop_time0 = time.perf_counter()
        try:
            pipe.fill(0)
            for ifrag, frag_idx in enumerate(run_frag):
                width = 80
                msg = f" LNO-FRAGMENT [{self.frag_name_all[frag_idx]}] {ifrag + 1}/({nfrag_run},{self.nfrag_tot}) "
                print("\n" + msg.center(width, "="))
                print(f"Fragment Num.  {ifrag + 1}")
                print(f"Fragment Idx.  {frag_idx + 1}")
                print(f"Fragment Name  {self.frag_name_all[frag_idx]}")
                print(f"LNO THRESHOLD  {lno_thresh}")
                print(f"PySCF Threads  {lib.num_threads()}")

                # ---------------- CPU stage (possibly already finished)
                time0 = time.perf_counter()
                frag = pipe.get(ifrag)
                wait_time = time.perf_counter() - time0
                pipe.fill(ifrag + 1)

                if frag.log:
                    print(frag.log, end="")
                print(f"LNO-MP2 Fragment Energy:  {frag.efrag_mp:.8f}")
                print(f"LNO-CCSD Fragment Energy: {frag.efrag_cc:.8f}")
                hidden = 100.0 * (1.0 - wait_time / frag.t_cpu) if frag.t_cpu > 0 else 0.0
                print(
                    f"LNO-CPU time (s):         {frag.t_cpu:.2f} "
                    f"(LAS {frag.t_las:.2f} | MP2 {frag.t_mp:.2f} | CCSD {frag.t_cc:.2f})"
                )
                print(f"LNO-CPU wait time (s):    {wait_time:.2f} ({hidden:.1f}% hidden)")

                # ---------------- device stage
                out_path = lno_io.frag_output_path(self.frag_output, frag_idx) if self.frag_output else None
                if self.run_qmc:
                    with lno_io.tee_to_file(out_path, mode="w"):
                        efrag_qmc, efrag_qmc_err, t_qmc, result = self.lnoafqmc_kernel(frag)
                    print(f"LNO-AFQMC time (s):       {t_qmc:.2f}")
                else:
                    efrag_qmc, efrag_qmc_err, t_qmc, result = 0.0, 0.0, 0.0, None

                self.lno_size[ifrag] = frag.nact
                self.lno_nocc[ifrag] = frag.nactocc
                self.lno_emp[ifrag] = frag.efrag_mp
                self.lno_ecc[ifrag] = frag.efrag_cc
                self.lno_eqmc[ifrag] = efrag_qmc
                self.lno_eqmc_err[ifrag] = efrag_qmc_err
                self.lno_cc_time[ifrag] = frag.t_cpu
                self.lno_wait_time[ifrag] = wait_time
                self.lno_qmc_time[ifrag] = t_qmc
                if self.keep_qmc_results:
                    self.frag_qmc_results[ifrag] = result
                self.frag_data[ifrag] = LnoFragData(
                    **{**frag.__dict__, "t1": None, "t2": None, "lno_coeff": None, "uocc_loc": None, "log": ""}
                )
                self.n_done = ifrag + 1
                self._update_totals()
                self.loop_time = time.perf_counter() - loop_time0

                if out_path is not None:
                    lno_io.write_frag_summary(
                        out_path,
                        frag_idx=frag_idx, frag_name=self.frag_name_all[frag_idx], nactocc=frag.nactocc,
                        norb=frag.nact, efrag_mp=frag.efrag_mp, efrag_cc=frag.efrag_cc, efrag_qmc=efrag_qmc,
                        efrag_qmc_err=efrag_qmc_err, t_cc=frag.t_cpu, t_wait=wait_time, t_qmc=t_qmc,
                    )
                if self.lno_output is not None:
                    self._write_lno_output(lno_thresh, depth)

                del frag, result
        except BaseException:
            pipe.shutdown(cancel=True)
            raise
        else:
            pipe.shutdown()

        self.loop_time = time.perf_counter() - loop_time0
        tot_cc = float(np.sum(self.lno_cc_time))
        tot_wait = float(np.sum(self.lno_wait_time))
        tot_qmc = float(np.sum(self.lno_qmc_time))
        print("\n" + "=" * 80)
        print(f"E(LNO-MP2)   = {self.e_mp:.8f}")
        print(f"E(LNO-CCSD)  = {self.e_cc:.8f}")
        print(f"E(LNO-AFQMC) = {self.e_qmc:.6f} +/- {self.e_qmc_err:.6f}")
        print(f"Loop wall time:              {self.loop_time:.2f} s")
        print(f"Serial equivalent (CPU+GPU): {tot_cc + tot_qmc:.2f} s")
        print(f"CPU time hidden behind GPU:  {tot_cc - tot_wait:.2f} s of {tot_cc:.2f} s")
        print("=" * 80)
        return self.e_qmc, self.e_qmc_err

    run = kernel

    def _write_lno_output(self, lno_thresh: Any, depth: int) -> None:
        k = self.n_done
        lno_io.write_lno_result(
            self.lno_output,
            run_frag=self.run_frag[:k],
            frag_name=self.frag_name[:k],
            lno_size=self.lno_size[:k],
            lno_emp=self.lno_emp[:k],
            lno_ecc=self.lno_ecc[:k],
            lno_eqmc=self.lno_eqmc[:k],
            lno_eqmc_err=self.lno_eqmc_err[:k],
            lno_cc_time=self.lno_cc_time[:k],
            lno_wait_time=self.lno_wait_time[:k],
            lno_qmc_time=self.lno_qmc_time[:k],
            lno_thresh=lno_thresh,
            depth=depth,
            loop_time=self.loop_time,
        )
