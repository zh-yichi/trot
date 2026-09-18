from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Union, cast

import h5py
import numpy as np
from numpy.typing import NDArray

from ..staging import HamInputU, StagedInputs, TrialInput
from ..staging import _dump_frozen, _freeze_from_meta_value, _load_frozen, _to_json_str
from ..staging import _load_h5 as _load_staged_h5

# The handoff between the CPU half of a fragment (LNO + MP2 + CCSD) and its AFQMC half,
# and the fragment trial staging. Mirrors trot/staging.py in role: plain numpy in, a
# TrialInput of arrays out, nothing jax.
#
#   LnoFragData          what cpu_stage produces for one fragment
#   frag_mf              the mean field in the fragment's LNO basis, for trot's guide staging
#   stage_pt2ccsd_trial  LnoFragData -> TrialInput {mo_t, t2 (projected), prjlo, t1}
#   stage_upt2ccsd_trial the unrestricted counterpart
#   dump_frag / load_frag  the self-contained frag{i}.h5 a fragment can be re-run from


def _as_frozen_array(frozen: Any) -> NDArray:
    """make_las returns 0 when nothing is frozen; keep a (possibly empty) int array."""
    if isinstance(frozen, (int, np.integer)):
        return np.zeros((0,), dtype=np.int64) if int(frozen) == 0 else np.arange(int(frozen))
    return np.asarray(frozen, dtype=np.int64).reshape(-1)


@dataclass(frozen=True)
class LnoFragData:
    """
    One fragment's LNO data, as produced by pipeline.cpu_stage.

    Restricted: lno_coeff (nao, nmo), lno_frozen (nfrozen,), uocc_loc (nactocc, nlo),
    t1 (nactocc, nactvir), t2 (nactocc, nactocc, nactvir, nactvir) in pyscf's (i,j,a,b)
    layout, all in the LNO basis. Unrestricted: each is a pair (alpha, beta), and t2 is
    (t2aa, t2ab, t2bb).

    lno_coeff is ordered [frz_occ | act_occ | act_vir | frz_vir], so the core is the
    leading columns and the active block is contiguous; integral.py relies on that.
    """

    frag_idx: int
    frag_name: str
    lno_coeff: Any
    lno_frozen: Any
    uocc_loc: Any
    nactocc: Any
    nactvir: Any
    t1: Any = None
    t2: Any = None
    efrag_mp: float = 0.0
    efrag_cc: float = 0.0
    lno_thresh: tuple = (None, None)
    nfrozen: int = 0
    t_las: float = 0.0
    t_mp: float = 0.0
    t_cc: float = 0.0
    t_cpu: float = 0.0
    log: str = field(default="", repr=False)

    @property
    def unrestricted(self) -> bool:
        return isinstance(self.lno_coeff, (tuple, list))

    @property
    def has_amplitudes(self) -> bool:
        return self.t1 is not None and self.t2 is not None

    @property
    def nact(self):
        """Active-space size, an int or an (alpha, beta) pair."""
        if self.unrestricted:
            return (
                int(self.nactocc[0]) + int(self.nactvir[0]),
                int(self.nactocc[1]) + int(self.nactvir[1]),
            )
        return int(self.nactocc) + int(self.nactvir)

    @staticmethod
    def _split_frozen(frozen: Any) -> tuple[int, int]:
        # the LAS ordering [frz_occ | act_occ | act_vir | frz_vir] puts the frozen occupied
        # at the leading indices 0, 1, ..., so they are the initial run of the sorted list
        idx = np.sort(_as_frozen_array(frozen))
        nfrzocc = int(np.sum(idx == np.arange(idx.size)))
        return nfrzocc, int(idx.size - nfrzocc)

    @property
    def nfrzocc(self):
        """Frozen occupied count, an int or an (alpha, beta) pair."""
        if self.unrestricted:
            return tuple(self._split_frozen(f)[0] for f in self.lno_frozen)
        return self._split_frozen(self.lno_frozen)[0]

    @property
    def nfrzvir(self):
        """Frozen virtual count, an int or an (alpha, beta) pair."""
        if self.unrestricted:
            return tuple(self._split_frozen(f)[1] for f in self.lno_frozen)
        return self._split_frozen(self.lno_frozen)[1]


def frag_mf(mf: Any, frag: LnoFragData) -> Any:
    """
    A shallow copy of mf whose orbitals are the fragment's LNOs.

    trot.staging.stage builds the guide from mf.mo_coeff, so handing it this copy (and
    the frozen LNO indices) stages the guide in the fragment basis: for HF that is the
    identity on the active LNOs, with the reference in the leading columns. mo_occ is
    left as it is, since the LAS ordering keeps occupied before virtual.
    """
    mf2 = copy.copy(mf)
    if frag.unrestricted:
        mf2.mo_coeff = np.array([np.asarray(frag.lno_coeff[0]), np.asarray(frag.lno_coeff[1])])
    else:
        mf2.mo_coeff = np.asarray(frag.lno_coeff)
    return mf2


def _thouless(t1: NDArray) -> NDArray:
    # |psi'> = exp(t1_ia a+_a a_i)|psi>, |psi> the leading nocc orbitals; the generator is
    # nilpotent so exp(X) = 1 + X exactly. Returns the mo_coeff of psi' in the MO basis.
    nocc, nvir = t1.shape
    exp_t1 = np.eye(nocc + nvir, dtype=np.float64)
    exp_t1[:nocc, nocc:] = t1
    return exp_t1.T[:, :nocc]


def stage_pt2ccsd_trial(frag: LnoFragData) -> TrialInput:
    """
    The fragment pt2CCSD trial.

    As trot's stage_pt2ccsd_trial, plus the fragment projector prjlo = U U^T with
    U = <act_occ|lo>, and t2 projected on its first occupied index,
    t2_kajb = sum_i t2_iajb prjlo_ik (afqmc's prep.proj_cc_amplitude). t1 is kept
    unprojected: the meas ctx needs it for e0t1orb.
    """
    if frag.unrestricted:
        raise ValueError(
            "stage_pt2ccsd_trial needs restricted fragment data; use stage_upt2ccsd_trial."
        )
    if not frag.has_amplitudes:
        raise ValueError("the pt2CCSD trial needs the fragment CCSD amplitudes (run_cc=True).")

    t1 = np.asarray(frag.t1, dtype=np.float64)
    t2 = np.asarray(frag.t2, dtype=np.float64).transpose(0, 2, 1, 3)  # (i,j,a,b) -> (i,a,j,b)
    uocc = np.asarray(frag.uocc_loc)
    prjlo = uocc @ uocc.T.conj()
    t2 = np.einsum("iajb,ik->kajb", t2, prjlo, optimize="optimal")

    data = {"mo_t": _thouless(t1), "t2": t2, "prjlo": prjlo, "t1": t1}
    return TrialInput(
        kind="pt2ccsd", data=data, frozen=_as_frozen_array(frag.lno_frozen), source_kind="mf"
    )


def stage_upt2ccsd_trial(frag: LnoFragData) -> TrialInput:
    """
    The unrestricted fragment pt2CCSD trial, in the conventions of trot's
    stage_upt2ccsd_trial: same-spin blocks antisymmetrized, everything (i,a,j,b), each
    spin in its own LNO basis. The projection on the first index makes t2ab and t2ba
    distinct, so both are staged.
    """
    if not frag.unrestricted:
        raise ValueError("stage_upt2ccsd_trial needs unrestricted fragment data.")
    if not frag.has_amplitudes:
        raise ValueError("the pt2CCSD trial needs the fragment CCSD amplitudes (run_cc=True).")

    t1a, t1b = (np.asarray(t, dtype=np.float64) for t in frag.t1)
    t2aa, t2ab, t2bb = (np.asarray(t, dtype=np.float64) for t in frag.t2)
    t2aa = 0.5 * (t2aa - t2aa.transpose(0, 1, 3, 2))
    t2bb = 0.5 * (t2bb - t2bb.transpose(0, 1, 3, 2))
    t2aa = t2aa.transpose(0, 2, 1, 3)
    t2ab = t2ab.transpose(0, 2, 1, 3)
    t2bb = t2bb.transpose(0, 2, 1, 3)

    ua, ub = (np.asarray(u) for u in frag.uocc_loc)
    prjlo_a = ua @ ua.T.conj()
    prjlo_b = ub @ ub.T.conj()

    data = {
        "mo_t_a": _thouless(t1a),
        "mo_t_b": _thouless(t1b),
        "t2aa": np.einsum("iajb,ik->kajb", t2aa, prjlo_a, optimize="optimal"),
        "t2ab": np.einsum("iajb,ik->kajb", t2ab, prjlo_a, optimize="optimal"),
        "t2ba": np.einsum("jbia,ik->kajb", t2ab, prjlo_b, optimize="optimal"),
        "t2bb": np.einsum("iajb,ik->kajb", t2bb, prjlo_b, optimize="optimal"),
        "prjlo_a": prjlo_a,
        "prjlo_b": prjlo_b,
        "t1a": t1a,
        "t1b": t1b,
    }
    frozen = tuple(_as_frozen_array(f) for f in frag.lno_frozen)
    return TrialInput(kind="upt2ccsd", data=data, frozen=frozen[0], source_kind="mf")


# --------------------------------------------------------------------------- frag{i}.h5

FRAG_FILE_VERSION = 1


def _fingerprint(mf: Any) -> dict[str, Any]:
    mol = mf.mol
    with_df = getattr(mf, "with_df", None)
    return {
        "atom_charges": [int(c) for c in mol.atom_charges()],
        "atom_coords": np.asarray(mol.atom_coords()).tolist(),
        "basis": str(mol.basis)[:1024],
        "charge": int(mol.charge),
        "spin": int(mol.spin),
        "nao": int(mol.nao),
        "naux": int(with_df.get_naoaux()) if with_df is not None else None,
        "e_tot": float(mf.e_tot),
    }


def _write_pair_or_array(group: h5py.Group, name: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, (tuple, list)):
        for s, tag in enumerate(("a", "b")):
            group.create_dataset(f"{name}_{tag}", data=np.asarray(value[s]))
    else:
        group.create_dataset(name, data=np.asarray(value))


def _read_pair_or_array(group: h5py.Group, name: str) -> Any:
    if name in group:
        return np.array(group[name])
    if f"{name}_a" in group:
        return (np.array(group[f"{name}_a"]), np.array(group[f"{name}_b"]))
    return None


def _dump_staged_uchol(staged: StagedInputs, path: Path) -> None:
    """
    trot.staging._dump_h5 for a HamInputU in the ham slot. trot's own dump only knows the
    restricted HamInput, so the uchol fragment file is written here in the same layout,
    with the per-spin arrays as h1_a/h1_b and chol_a/chol_b and norb as a pair.
    """
    ham: Any = staged.ham
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["meta_json"] = json.dumps(staged.meta)

        gham = f.create_group("ham")
        gham.create_dataset("h0", data=np.array(ham.h0))
        for name in ("h1_a", "h1_b", "chol_a", "chol_b"):
            gham.create_dataset(name, data=np.asarray(getattr(ham, name)))
        gham.create_dataset("nelec", data=np.array(ham.nelec, dtype=np.int64))
        gham.create_dataset("norb", data=np.array(ham.norb, dtype=np.int64))
        gham.attrs["chol_cut"] = ham.chol_cut
        _dump_frozen(gham, ham.frozen)
        gham.attrs["source_kind"] = ham.source_kind
        gham.attrs["basis"] = ham.basis

        gtr = f.create_group("trial")
        gtr.attrs["kind"] = staged.trial.kind
        _dump_frozen(gtr, staged.trial.frozen)
        gtr.attrs["source_kind"] = staged.trial.source_kind
        gdata = gtr.create_group("data")
        for k, v in staged.trial.data.items():
            gdata.create_dataset(k, data=np.asarray(v))


def _load_staged_uchol(path: Path) -> StagedInputs:
    with h5py.File(path, "r") as f:
        meta = json.loads(_to_json_str(f.attrs["meta_json"]))
        if "frozen" in meta:
            meta["frozen"] = _freeze_from_meta_value(meta["frozen"])
        gham: Any = f["ham"]
        nelec = np.array(gham["nelec"])
        norb = np.array(gham["norb"])
        ham = HamInputU(
            h0=float(np.array(gham["h0"]).item()),
            h1_a=np.array(gham["h1_a"]),
            h1_b=np.array(gham["h1_b"]),
            chol_a=np.array(gham["chol_a"]),
            chol_b=np.array(gham["chol_b"]),
            nelec=(int(nelec[0]), int(nelec[1])),
            norb=(int(norb[0]), int(norb[1])),
            chol_cut=float(gham.attrs["chol_cut"]),
            frozen=_load_frozen(gham),
            source_kind=str(gham.attrs["source_kind"]),
            basis=str(gham.attrs["basis"]),
        )
        gtr: Any = f["trial"]
        trial = TrialInput(
            kind=str(gtr.attrs["kind"]),
            data={k: np.array(gtr["data"][k]) for k in gtr["data"].keys()},
            frozen=_load_frozen(gtr),
            source_kind=str(gtr.attrs["source_kind"]),
        )
    # StagedInputs.ham is typed as the restricted HamInput; the unrestricted path carries
    # a HamInputU through the same slot, as in AfqmcMixed.stage
    return StagedInputs(ham=cast(Any, ham), trial=trial, meta=meta)


def _is_uchol_file(path: Path) -> bool:
    with h5py.File(path, "r") as f:
        return str(f["ham"].attrs.get("basis", "")) == "uchol"


def dump_frag(
    path: Union[str, Path],
    frag: LnoFragData,
    staged: StagedInputs,
    mf: Any,
    *,
    emf: float | None = None,
) -> Path:
    """
    Write the self-contained fragment file: the LNO data and amplitudes, the fragment
    hamiltonian and the staged guide (in trot's staged-inputs layout, so trot.staging.load
    reads the restricted ones; the uchol hamiltonian is written per spin, see
    _dump_staged_uchol), and a fingerprint of the system. A re-run needs nothing else.
    """
    from ..staging import dump as dump_staged

    path = Path(path)
    if getattr(staged.ham, "basis", None) == "uchol":
        _dump_staged_uchol(staged, path)
    else:
        dump_staged(staged, path)  # ham/, trial/ (= guide), meta_json
    with h5py.File(path, "a") as f:
        f.attrs["frag_file_version"] = FRAG_FILE_VERSION
        f.attrs["emf"] = float(mf.e_tot if emf is None else emf)
        f.attrs["fingerprint_json"] = json.dumps(_fingerprint(mf))
        f.attrs["timestamp_unix"] = time.time()

        g = f.create_group("frag")
        g.attrs["frag_idx"] = int(frag.frag_idx)
        g.attrs["frag_name"] = str(frag.frag_name)
        g.attrs["unrestricted"] = bool(frag.unrestricted)
        g.attrs["nfrozen"] = int(frag.nfrozen)
        g.attrs["lno_thresh_json"] = json.dumps(
            [None if x is None else float(x) for x in frag.lno_thresh]
        )
        g.attrs["efrag_mp"] = float(frag.efrag_mp)
        g.attrs["efrag_cc"] = float(frag.efrag_cc)
        for k in ("t_las", "t_mp", "t_cc", "t_cpu"):
            g.attrs[k] = float(getattr(frag, k))
        _write_pair_or_array(g, "lno_coeff", frag.lno_coeff)
        _write_pair_or_array(g, "lno_frozen", frag.lno_frozen)
        _write_pair_or_array(g, "uocc_loc", frag.uocc_loc)
        g.create_dataset("nactocc", data=np.asarray(frag.nactocc, dtype=np.int64))
        g.create_dataset("nactvir", data=np.asarray(frag.nactvir, dtype=np.int64))

        if frag.has_amplitudes:
            ga = f.create_group("amplitudes")
            _write_pair_or_array(ga, "t1", frag.t1)
            if frag.unrestricted:
                for name, t in zip(("t2aa", "t2ab", "t2bb"), frag.t2):
                    ga.create_dataset(name, data=np.asarray(t))
            else:
                ga.create_dataset("t2", data=np.asarray(frag.t2))
    return path


def load_frag(path: Union[str, Path]) -> tuple[LnoFragData, StagedInputs, dict[str, Any]]:
    """Read back what dump_frag wrote: (frag data, staged ham + guide, file attributes)."""
    path = Path(path)
    staged = _load_staged_uchol(path) if _is_uchol_file(path) else _load_staged_h5(path)
    with h5py.File(path, "r") as h5file:
        f: Any = h5file  # h5py's item types are unions pyright cannot narrow
        version = int(f.attrs.get("frag_file_version", -1))
        if version != FRAG_FILE_VERSION:
            raise ValueError(
                f"{path}: fragment file version {version}, expected {FRAG_FILE_VERSION}"
            )
        attrs = {
            "emf": float(f.attrs["emf"]),
            "fingerprint": json.loads(str(f.attrs["fingerprint_json"])),
            "timestamp_unix": float(f.attrs["timestamp_unix"]),
        }
        g = f["frag"]
        unrestricted = bool(g.attrs["unrestricted"])
        nactocc = np.array(g["nactocc"])
        nactvir = np.array(g["nactvir"])
        kw: dict[str, Any] = dict(
            frag_idx=int(g.attrs["frag_idx"]),
            frag_name=str(g.attrs["frag_name"]),
            lno_coeff=_read_pair_or_array(g, "lno_coeff"),
            lno_frozen=_read_pair_or_array(g, "lno_frozen"),
            uocc_loc=_read_pair_or_array(g, "uocc_loc"),
            nactocc=tuple(int(x) for x in nactocc) if unrestricted else int(nactocc),
            nactvir=tuple(int(x) for x in nactvir) if unrestricted else int(nactvir),
            efrag_mp=float(g.attrs["efrag_mp"]),
            efrag_cc=float(g.attrs["efrag_cc"]),
            lno_thresh=tuple(json.loads(str(g.attrs["lno_thresh_json"]))),
            nfrozen=int(g.attrs["nfrozen"]),
        )
        for k in ("t_las", "t_mp", "t_cc", "t_cpu"):
            kw[k] = float(g.attrs[k])
        if "amplitudes" in f:
            ga = f["amplitudes"]
            kw["t1"] = _read_pair_or_array(ga, "t1")
            if unrestricted:
                kw["t2"] = tuple(np.array(ga[n]) for n in ("t2aa", "t2ab", "t2bb"))
            else:
                kw["t2"] = np.array(ga["t2"])
    return LnoFragData(**kw), staged, attrs


def frag_data_fields() -> tuple[str, ...]:
    return tuple(f.name for f in fields(LnoFragData))
