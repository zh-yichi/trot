"""
Saving the AFQMC wavefunction: triplet O2 in cc-pVDZ, UHF guide, upt2ccsd_bar trial.

The guided walker population represents

    |psi> = sum_i w_i / <G|phi_i> |phi_i>

with w_i the weights, |phi_i> the walker determinants and <G|phi_i> their overlaps with
the guide. AfqmcMixed(save_wavefunction=dir) writes that population, with the orbital
basis the walkers are expressed in, at tau = 0 and at every printed row of the
equilibration and the sampling; af.save_wavefunction(path) writes the final population
of a finished run to one file. trot/wavefunction_io.py documents the file layout.
"""

import json
from pathlib import Path

import numpy as np
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed
from trot.wavefunction_io import load_wavefunction, walker_ao_coefficients

mol = gto.M(atom="O 0 0 0; O 0 0 1.20577", basis="ccpvdz", spin=2, verbose=3)
mf = scf.UHF(mol).density_fit()
mf.kernel()

# follow the UHF solution down to a stable one
for _ in range(5):
    mo_i, _, stable, _ = mf.stability(return_status=True)
    if stable:
        break
    mf.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))
print(f"UHF energy: {mf.e_tot:.10f}")

# the frozen core of the hamiltonian follows cc.frozen
mycc = cc.CCSD(mf).set_frozen()
mycc.kernel()

snap_dir = Path("./wfn_snaps")  # save_wavefunction=True uses this directory as well

af = AfqmcMixed(
    mycc,
    guide="uhf",
    trial="upt2ccsd_bar",
    n_walkers=200,
    n_blocks=100,
    tau_eql=20,
    seed=7,
    mixed_precision=True,
    save_wavefunction=snap_dir,  # one snapshot per printed row, each holds every walker
)
e, err = af.kernel()
print(f"AFQMC/upt2CCSD (UHF guide) energy: {e:.6f} +/- {err:.6f}")
print(f"  guide energy: {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f}")

# the final population of the finished run, with the final energies in its attributes
final = af.save_wavefunction("o2_wavefunction.h5")
print(f"final wavefunction written to {final}")

# ---- reading the snapshots back
# snapshots.json lists the files in order of tau, with the phase ("init", "eql",
# "sample"), the block and tau of each
manifest = json.loads((snap_dir / "snapshots.json").read_text())
print(f"\n{len(manifest)} snapshots in {snap_dir}:")
for entry in (manifest[0], manifest[len(manifest) // 2], manifest[-1]):
    print(f"  {entry['file']}  phase={entry['phase']:<6}  tau={entry['tau']:.3f}")

wf = load_wavefunction(snap_dir / manifest[-1]["file"])
# an unrestricted run carries one walker set and one orbital basis per spin; a
# restricted one has "walkers" and "basis" instead
walkers_a, walkers_b = wf["walkers_a"], wf["walkers_b"]  # (n_walkers, norb, nocc_s)
weights, overlaps = wf["weights"], wf["overlaps"]  # (n_walkers,), overlaps = <G|phi_i>
basis_a, basis_b = wf["basis_a"], wf["basis_b"]
print(f"\nlast snapshot: {wf['walker_kind']} walkers, active nelec = {wf['nelec']}")
print(f"  walkers_a {walkers_a.shape}, walkers_b {walkers_b.shape}")
print(
    f"  alpha basis ({basis_a.kind}): {basis_a.nao} AOs, {basis_a.norb} active orbitals, "
    f"frozen core columns {basis_a.frozen_occ.tolist()}"
)
print(
    f"  guide = {wf['attrs']['guide']}, trial = {wf['attrs']['trial']}, tau = {wf['attrs']['tau']}"
)

# the coefficients of |psi> = sum_i c_i |phi_i>
coeffs = weights / overlaps
print(f"  sum_i |w_i / <G|phi_i>| = {np.abs(coeffs).sum():.6f}")

# the walkers' occupied orbitals in the AO basis, C_active @ walker_i per spin; the full
# wavefunction is their product with the frozen core basis.frozen_occ_coeff
ao_a, ao_b = walker_ao_coefficients(wf)  # (n_walkers, nao, nocc_s)
s1e = mol.intor("int1e_ovlp")
core_a = basis_a.frozen_occ_coeff
print(f"  AO orbitals: alpha {ao_a.shape}, beta {ao_b.shape}")
print(
    "  max overlap of a walker orbital with the frozen core: "
    f"{np.abs(np.einsum('pc,pq,wqi->wci', core_a, s1e, ao_a)).max():.1e}"
)
