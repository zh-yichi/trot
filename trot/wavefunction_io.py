"""
Saving the guided AFQMC wavefunction, and enough of its basis to rebuild it from the AOs.

The importance sampled walker population represents

    |psi> = sum_i w_i / <G|phi_i> |phi_i>

with w_i the walker weights, |phi_i> the walker determinants and <G|phi_i> their overlaps
with the guide. The walkers are expressed in the orbital basis the AFQMC hamiltonian was
built in (canonical MOs, an LNO fragment basis, natural orbitals, ...), restricted to the
active orbitals: the frozen core is a fixed determinant the full wavefunction is the
fermionic product with,

    |Psi> = |core> x |psi>,     |core> = the frozen occupied orbitals, doubly occupied
                                         (per spin for an unrestricted basis)

so a file carries, next to the population, the AO coefficients of the whole orbital
set, which columns are the active ones (in the walkers' orbital order), which are the
frozen occupied ones and which the frozen virtual ones (those take no part in the
wavefunction). With C = coeff[:, active] the AO coefficients of walker i's occupied
orbitals are C @ walker_i (per spin for unrestricted walkers), and the core determinant
is coeff[:, frozen_occ].

WavefunctionBasis     the orbital set and the three column index sets, per spin
dump_wavefunction     the population + basis + run metadata to an h5 file
load_wavefunction     the file back as a dict of numpy arrays and attributes
walker_ao_coefficients
                      C @ walker for a loaded file, the walkers' orbitals in the AO basis
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import h5py
import numpy as np
from numpy.typing import NDArray

WAVEFUNCTION_FORMAT = "trot_afqmc_wavefunction"
WAVEFUNCTION_FORMAT_VERSION = 1
# where AfqmcMixed(save_wavefunction=True) puts its snapshots
WAVEFUNCTION_SNAPSHOT_DIR = "./wfn_snaps"


@dataclass(frozen=True)
class WavefunctionBasis:
    """
    The orbital set of one spin: coeff (nao, nmo) in the AO basis, and the column indices
    of the active orbitals (in the order the AFQMC hamiltonian and the walkers use them),
    the frozen occupied orbitals and the frozen virtual orbitals. kind names the set
    ("canonical_mo", "lno", ...).
    """

    coeff: NDArray
    active: NDArray
    frozen_occ: NDArray
    frozen_vir: NDArray
    kind: str = "canonical_mo"

    def __post_init__(self) -> None:
        coeff = np.asarray(self.coeff)
        if coeff.ndim != 2:
            raise ValueError(f"coeff must be (nao, nmo), got shape {coeff.shape}.")
        nmo = coeff.shape[1]
        idx = [
            np.asarray(x, dtype=np.int64).reshape(-1)
            for x in (self.active, self.frozen_occ, self.frozen_vir)
        ]
        allidx = np.concatenate(idx)
        if allidx.size and (allidx.min() < 0 or allidx.max() >= nmo):
            raise ValueError(
                f"orbital indices must lie in [0, {nmo}), got {allidx.min()}..{allidx.max()}."
            )
        if np.unique(allidx).size != allidx.size:
            raise ValueError("an orbital cannot be active and frozen at once.")
        object.__setattr__(self, "coeff", coeff)
        object.__setattr__(self, "active", idx[0])
        object.__setattr__(self, "frozen_occ", idx[1])
        object.__setattr__(self, "frozen_vir", idx[2])

    @property
    def nao(self) -> int:
        return int(self.coeff.shape[0])

    @property
    def norb(self) -> int:
        """The active orbital count, the walkers' orbital dimension."""
        return int(self.active.size)

    @property
    def active_coeff(self) -> NDArray:
        """(nao, norb): the AO coefficients of the active orbitals, in the walkers' order."""
        return self.coeff[:, self.active]

    @property
    def frozen_occ_coeff(self) -> NDArray:
        """(nao, nfrozen): the AO coefficients of the frozen occupied orbitals."""
        return self.coeff[:, self.frozen_occ]

    @classmethod
    def leading_core(
        cls, coeff: NDArray, norb_frozen: int, *, kind: str = "canonical_mo"
    ) -> "WavefunctionBasis":
        """
        The layout of trot's staging: the first norb_frozen columns of coeff are the frozen
        core, the rest are the active orbitals in order, nothing is frozen above.
        """
        nmo = int(np.asarray(coeff).shape[1])
        n = int(norb_frozen)
        if not 0 <= n < nmo:
            raise ValueError(f"norb_frozen={n} must lie in [0, {nmo}).")
        return cls(
            coeff=coeff,
            active=np.arange(n, nmo),
            frozen_occ=np.arange(n),
            frozen_vir=np.zeros((0,), dtype=np.int64),
            kind=kind,
        )

    @classmethod
    def from_frozen_indices(
        cls, coeff: NDArray, frozen: NDArray, nocc_full: int, *, kind: str
    ) -> "WavefunctionBasis":
        """
        An explicit frozen list (the LNO fragments): the frozen columns below nocc_full,
        the occupied count of the whole orbital set, are the core; the active orbitals
        are every other column, in order.
        """
        nmo = int(np.asarray(coeff).shape[1])
        frozen = np.asarray(frozen, dtype=np.int64).reshape(-1)
        active = np.setdiff1d(np.arange(nmo), frozen)
        return cls(
            coeff=coeff,
            active=active,
            frozen_occ=frozen[frozen < int(nocc_full)],
            frozen_vir=frozen[frozen >= int(nocc_full)],
            kind=kind,
        )


def _walker_arrays(walkers: Any, walker_kind: str) -> dict[str, NDArray]:
    kind = walker_kind.lower()
    if kind == "unrestricted":
        wa, wb = walkers
        return {"walkers_a": np.asarray(wa), "walkers_b": np.asarray(wb)}
    if kind in ("restricted", "generalized"):
        return {"walkers": np.asarray(walkers)}
    raise ValueError(f"unknown walker_kind: {walker_kind}")


def dump_wavefunction(
    path: Union[str, Path],
    *,
    walkers: Any,
    weights: Any,
    overlaps: Any,
    walker_kind: str,
    basis: WavefunctionBasis | tuple[WavefunctionBasis, WavefunctionBasis],
    nelec: tuple[int, int],
    meta: dict[str, Any] | None = None,
) -> Path:
    """
    Write the population and its basis. basis is one WavefunctionBasis for a restricted
    hamiltonian (both spins share the orbital set) or an (alpha, beta) pair for an
    unrestricted one. nelec is the active electron count per spin. meta holds run
    attributes (guide, trial, dt, tau, seed, ...); scalars and strings become h5
    attributes, anything else is stored as json.
    """
    path = Path(path)
    bases = basis if isinstance(basis, tuple) else (basis,)
    arrays = _walker_arrays(walkers, walker_kind)
    weights = np.asarray(weights)
    overlaps = np.asarray(overlaps)
    n = int(weights.shape[0])
    for name, w in arrays.items():
        if w.shape[0] != n or overlaps.shape[0] != n:
            raise ValueError(
                f"{name} has {w.shape[0]} walkers but weights {n} and overlaps {overlaps.shape[0]}."
            )
    for i, (name, w) in enumerate(arrays.items()):
        b = bases[min(i, len(bases) - 1)]
        norb = w.shape[1] if walker_kind.lower() != "generalized" else w.shape[1] // 2
        if norb != b.norb:
            raise ValueError(
                f"{name} has {norb} orbitals but the basis has {b.norb} active columns."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = WAVEFUNCTION_FORMAT
        f.attrs["format_version"] = WAVEFUNCTION_FORMAT_VERSION
        f.attrs["timestamp_unix"] = time.time()
        f.attrs["walker_kind"] = walker_kind.lower()
        f.attrs["n_walkers"] = n
        f.attrs["nelec"] = np.asarray(nelec, dtype=np.int64)
        f.attrs["description"] = (
            "|psi> = sum_i weights[i] / overlaps[i] |walkers[i]>, overlaps[i] = <guide|walker_i>; "
            "the walkers are in the active orbitals basis/coeff[:, basis/active]; the full "
            "wavefunction is the fermionic product with the frozen occupied orbitals "
            "basis/coeff[:, basis/frozen_occ] (doubly occupied; per spin for basis_a / basis_b)."
        )
        for name, w in arrays.items():
            f.create_dataset(name, data=w)
        f.create_dataset("weights", data=weights)
        f.create_dataset("overlaps", data=overlaps)
        names = ("basis",) if len(bases) == 1 else ("basis_a", "basis_b")
        for name, b in zip(names, bases):
            g = f.create_group(name)
            g.attrs["kind"] = b.kind
            g.attrs["nao"] = b.nao
            g.attrs["nmo"] = int(b.coeff.shape[1])
            g.attrs["norb"] = b.norb
            g.create_dataset("coeff", data=b.coeff)
            g.create_dataset("active", data=b.active)
            g.create_dataset("frozen_occ", data=b.frozen_occ)
            g.create_dataset("frozen_vir", data=b.frozen_vir)
        extra: dict[str, Any] = {}
        for key, value in (meta or {}).items():
            if isinstance(value, (bool, int, float, str, np.integer, np.floating)):
                f.attrs[key] = value
            elif isinstance(value, np.ndarray) or (
                isinstance(value, (tuple, list)) and all(isinstance(v, (int, float)) for v in value)
            ):
                f.attrs[key] = np.asarray(value)
            elif value is not None:
                extra[key] = value
        if extra:
            f.attrs["meta_json"] = json.dumps(extra, default=str)
    return path


def _read_basis(g: h5py.Group) -> WavefunctionBasis:
    return WavefunctionBasis(
        coeff=np.array(g["coeff"]),
        active=np.array(g["active"]),
        frozen_occ=np.array(g["frozen_occ"]),
        frozen_vir=np.array(g["frozen_vir"]),
        kind=str(g.attrs.get("kind", "")),
    )


def load_wavefunction(path: Union[str, Path]) -> dict[str, Any]:
    """
    The file back: "walkers" (or "walkers_a" / "walkers_b"), "weights", "overlaps",
    "walker_kind", "nelec", "basis" (a WavefunctionBasis, or an (alpha, beta) pair under
    "basis_a" / "basis_b"), and every attribute of the file under "attrs".
    """
    path = Path(path)
    out: dict[str, Any] = {}
    with h5py.File(path, "r") as f:
        if f.attrs.get("format") != WAVEFUNCTION_FORMAT:
            raise ValueError(f"{path} is not a {WAVEFUNCTION_FORMAT} file.")
        for name in ("walkers", "walkers_a", "walkers_b", "weights", "overlaps"):
            if name in f:
                out[name] = np.array(f[name])
        out["walker_kind"] = str(f.attrs["walker_kind"])
        out["nelec"] = tuple(int(x) for x in np.asarray(f.attrs["nelec"]))
        if "basis" in f:
            out["basis"] = _read_basis(f["basis"])
        else:
            out["basis_a"] = _read_basis(f["basis_a"])
            out["basis_b"] = _read_basis(f["basis_b"])
        attrs = {}
        for key, value in f.attrs.items():
            attrs[key] = value.tolist() if isinstance(value, np.ndarray) else value
        if "meta_json" in attrs:
            attrs.update(json.loads(attrs.pop("meta_json")))
        out["attrs"] = attrs
    return out


def walker_ao_coefficients(loaded: dict[str, Any]) -> Any:
    """
    The AO coefficients of every walker's occupied orbitals, C @ walker_i: an array
    (n_walkers, nao, nocc) for a restricted file, an (alpha, beta) pair of them for an
    unrestricted one.
    """
    if "walkers" in loaded:
        c = loaded["basis"].active_coeff
        return np.einsum("pq,wqi->wpi", c, loaded["walkers"])
    ca, cb = loaded["basis_a"].active_coeff, loaded["basis_b"].active_coeff
    return (
        np.einsum("pq,wqi->wpi", ca, loaded["walkers_a"]),
        np.einsum("pq,wqi->wpi", cb, loaded["walkers_b"]),
    )
