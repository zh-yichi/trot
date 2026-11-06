"""
Stochastically sampled CCSD guide (restricted, closed shell).

The CCSD state is rewritten as a Gaussian average of single determinants. T1 is absorbed
into the reference by the Thouless theorem,

    |psi~> = exp(T1) |HF>,

and the doubles amplitudes, as the symmetric (ia, jb) matrix t_{iajb}, are diagonalized,

    t_{iajb} = sum_g tau_{g,ia} tau_{g,jb},    tau = Q sqrt(lambda)   (complex, since
                                                                      lambda can be < 0)

so that T2 = 1/2 sum_g tau_g^2 with tau_g = sum_ia tau_{g,ia} E_ai, and the
Hubbard-Stratonovich identity gives

    exp(T2) |psi~> = E_y[ exp(y . tau) |psi~> ],    y ~ N(0, 1)^{n_modes}.

Every sample exp(y . tau)|psi~> is one determinant (a second Thouless rotation of psi~),
so N samples give an N-determinant guide,

    |G(y_1..y_N)> = 1/N sum_s exp(y_s . tau) |psi~>,

whose overlap, force bias and local energy are those of a multi-Slater trial
(meas/stoccsd.py). The sample is redrawn at every propagation step (prop/afqmc_stocc.py),
which is why the guide pytree carries the *current* determinants as data.

StoccsdGuide holds mo_t (the Thouless reference), tau and the current dets; sample_dets
draws a new set; decompose_t2 is the host-side factorization used at staging.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np
from jax import tree_util

from ..core.ops import TrialOps
from ..core.system import System


@tree_util.register_pytree_node_class
@dataclass(frozen=True)
class StoccsdGuide:
    """
    Arrays (all in the MO basis of the hamiltonian, the reference occupying the first
    nocc orbitals):
      mo_t:  (norb, nocc)             exp(T1)|HF> as orbital coefficients (Thouless)
      tau:   (n_modes, nocc, nvir)    t_{iajb} = sum_g tau_{g,ia} tau_{g,jb}, complex
      dets:  (n_samples, norb, nocc)  the current sample exp(y_s . tau)|psi~>, complex
    """

    mo_t: jax.Array
    tau: jax.Array
    dets: jax.Array

    @property
    def norb(self) -> int:
        return int(self.mo_t.shape[0])

    @property
    def nocc(self) -> int:
        return int(self.mo_t.shape[1])

    @property
    def nvir(self) -> int:
        return int(self.tau.shape[2])

    @property
    def n_modes(self) -> int:
        return int(self.tau.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.dets.shape[0])

    def tree_flatten(self):
        return (self.mo_t, self.tau, self.dets), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        mo_t, tau, dets = children
        return cls(mo_t=mo_t, tau=tau, dets=dets)


# ------------------------------------------------------------------------------------
# T2 factorization (host side, staging)
# ------------------------------------------------------------------------------------


def decompose_t2(t2: np.ndarray, thresh: float = 1e-8) -> np.ndarray:
    """
    tau (n_modes, nocc, nvir), complex, with sum_g tau_{g,ia} tau_{g,jb} = t_{iajb}, from
    the eigen-decomposition of t2 (nocc, nvir, nocc, nvir) as the symmetric (ia, jb)
    matrix; modes with |eigenvalue| <= thresh are dropped. Negative eigenvalues make tau
    imaginary, which is why the sampled determinants are complex.
    """
    t2 = np.asarray(t2)
    nocc, nvir, nocc2, nvir2 = t2.shape
    if (nocc, nvir) != (nocc2, nvir2):
        raise ValueError(f"t2 must be (nocc, nvir, nocc, nvir), got {t2.shape}")
    npair = nocc * nvir
    mat = t2.reshape(npair, npair)
    asym = np.abs(mat - mat.T).max() if npair else 0.0
    if asym > 1e-10:
        raise ValueError(f"t2 is not symmetric under (ia) <-> (jb): max asymmetry {asym:.3e}")
    mat = 0.5 * (mat + mat.T)
    e_val, e_vec = np.linalg.eigh(mat)
    keep = np.abs(e_val) > float(thresh)
    e_val = e_val[keep]
    e_vec = e_vec[:, keep]
    tau = e_vec * np.sqrt(e_val.astype(np.complex128))[None, :]  # (npair, n_modes)
    err = np.linalg.norm(mat - (tau @ tau.T).real) if tau.shape[1] else np.linalg.norm(mat)
    dropped = (~keep).sum()
    # what is left out is the dropped modes; their norm is bounded by thresh each
    if err > 10.0 * float(thresh) * max(1, int(dropped)) + 1e-12:
        raise RuntimeError(f"t2 factorization error {err:.3e} exceeds the threshold budget")
    return np.ascontiguousarray(tau.T.reshape(-1, nocc, nvir))


# ------------------------------------------------------------------------------------
# sampling
# ------------------------------------------------------------------------------------


def thouless_r(mo: jax.Array, t: jax.Array) -> jax.Array:
    """
    exp(sum_ia t_ia a+_a a_i) |mo> as orbital coefficients: (I + t)^T applied to mo,
    with t in the (occ, vir) block. The orbitals of the result are
    phi_i -> phi_i + sum_a t_ia phi_a, and only the first nocc rows of mo are used by
    the rotation of the virtual rows.
    """
    nocc = t.shape[0]
    top = mo[:nocc]
    bot = mo[nocc:] + t.T @ top
    return jnp.concatenate([top, bot], axis=0)


def sample_dets(mo_t: jax.Array, tau: jax.Array, key: jax.Array, n_samples: int) -> jax.Array:
    """
    n_samples determinants exp(y_s . tau)|psi~>, y_s ~ N(0, 1)^{n_modes}, as a
    (n_samples, norb, nocc) complex array. With no modes every sample is psi~ itself.
    """
    n_modes = tau.shape[0]
    y = jax.random.normal(key, (int(n_samples), int(n_modes)), dtype=jnp.float64)
    ytau = jnp.einsum("wg,gia->wia", y, tau)  # (n_samples, nocc, nvir)
    mo_c = jnp.asarray(mo_t, dtype=jnp.complex128)
    return jax.vmap(lambda t: thouless_r(mo_c, t))(ytau)


def with_dets(guide: StoccsdGuide, dets: jax.Array) -> StoccsdGuide:
    return replace(guide, dets=dets)


def resample(guide: StoccsdGuide, key: jax.Array) -> StoccsdGuide:
    """The guide with a fresh sample of the same size."""
    return with_dets(guide, sample_dets(guide.mo_t, guide.tau, key, guide.n_samples))


# ------------------------------------------------------------------------------------
# trial ops
# ------------------------------------------------------------------------------------


def det_overlaps_r(walker: jax.Array, dets: jax.Array) -> jax.Array:
    """<d_s|walker> = det(d_s^H walker)^2 for every sampled determinant, (n_samples,)."""
    return jax.vmap(lambda d: jnp.linalg.det(d.conj().T @ walker) ** 2)(dets)


def overlap_r(walker: jax.Array, guide: StoccsdGuide) -> jax.Array:
    """<G|walker> = 1/N sum_s <d_s|walker>."""
    return jnp.mean(det_overlaps_r(walker, guide.dets))


def get_rdm1(guide: StoccsdGuide) -> jax.Array:
    """
    The density of the HF reference (the first nocc orbitals), per spin: (2, norb, norb),
    as the CISD guide's; the walkers start from the HF determinant.
    """
    occ = jnp.arange(guide.norb) < guide.nocc
    dm = jnp.diag(occ).astype(jnp.float64)
    return jnp.stack([dm, dm], axis=0)


def make_stoccsd_trial_ops(sys: System) -> TrialOps:
    if sys.nup != sys.ndn:
        raise ValueError("the stoccsd guide is closed shell: nelec[0] == nelec[1].")
    if sys.walker_kind.lower() != "restricted":
        raise ValueError(
            f"the stoccsd guide propagates restricted walkers, got walker_kind={sys.walker_kind!r}"
        )
    return TrialOps(overlap=overlap_r, get_rdm1=get_rdm1)


def make_stoccsd_trial_data(
    data: dict, sys: System, *, n_samples: int, seed: int = 0
) -> StoccsdGuide:
    """
    The guide from a staged TrialInput of kind "stoccsd" (staging_stoccsd): mo_t and tau,
    with a first sample of n_samples determinants drawn from seed. The propagation
    redraws the sample at every step from the walker key, so seed only sets the
    determinants the initial population and the tau = 0 energies see.
    """
    if int(n_samples) < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    mo_t = jnp.asarray(data["mo_t"], dtype=jnp.float64)[:, : sys.nup]
    tau = jnp.asarray(data["tau"], dtype=jnp.complex128)
    if tau.ndim != 3 or tau.shape[1] != sys.nup or tau.shape[1] + tau.shape[2] != sys.norb:
        raise ValueError(
            f"tau must be (n_modes, nocc={sys.nup}, nvir={sys.norb - sys.nup}), got {tau.shape}"
        )
    dets = sample_dets(mo_t, tau, jax.random.PRNGKey(int(seed)), int(n_samples))
    return StoccsdGuide(mo_t=mo_t, tau=tau, dets=dets)
