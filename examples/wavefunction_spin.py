"""
<S^2> and <S_z> of the AFQMC wavefunction along imaginary time, from saved snapshots.

Run examples/save_wavefunction.py first: it writes the walker population of triplet O2
(cc-pVDZ, UHF guide, upt2ccsd_bar trial) to ./wfn_snaps at tau = 0 and at every printed
row of the run. This script only reads those files; no AFQMC is run here.

For the population |Psi> = sum_i c_i |phi_i>, c_i = w_i / <G|phi_i>, two estimators of
S^2 are printed (trot/wavefunction_spin.py has the formula):

    mixed   <G|S^2|Psi> / <G|Psi>        the guide on the left
    pure    <Psi|S^2|Psi> / <Psi|Psi>    the population on both sides, all walker pairs

Every walker keeps (N_alpha, N_beta), so <S_z> = (N_alpha - N_beta) / 2 exactly.
"""

from pathlib import Path

from pyscf import gto

from trot.wavefunction_io import load_wavefunction
from trot.wavefunction_spin import guide_from_snapshots, spin_along_tau, transition_spin

snap_dir = Path("./wfn_snaps")

# the walkers are rebuilt in the AO basis, so the only extra input is the AO overlap
# matrix of the molecule the run was made for
mol = gto.M(atom="O 0 0 0; O 0 0 1.20577", basis="ccpvdz", spin=2, verbose=0)
s_ao = mol.intor("int1e_ovlp")

wf0 = load_wavefunction(snap_dir / "wfn_0000.h5")
print(f"guide = {wf0['attrs']['guide']}, trial = {wf0['attrs']['trial']}, ", end="")
print(f"{wf0['weights'].size} {wf0['walker_kind']} walkers, active nelec = {wf0['nelec']}")

# the walkers start from the guide, so walker 0 at tau = 0 is the UHF determinant:
# its occupied orbitals in the AO basis, frozen core included
guide_a, guide_b = guide_from_snapshots(snap_dir)
m = 0.5 * (guide_a.shape[1] - guide_b.shape[1])
_, s2_guide = transition_spin(guide_a, guide_b, guide_a, guide_b, s_ao)
print(f"UHF guide: <S^2> = {s2_guide.real:.6f}; a pure S = {m:.0f} state has {m * (m + 1):.0f}\n")

rows = spin_along_tau(snap_dir, s_ao, guide=(guide_a, guide_b))

print(f"{'tau':>7s} {'phase':>7s} {'<S^2> mixed':>12s} {'<S^2> pure':>12s} {'<S_z>':>9s}")
for r in rows:
    print(
        f"{r['tau']:7.2f} {r['phase']:>7s} {r['s2_mixed']:12.6f} {r['s2_pure']:12.6f} "
        f"{r['sz']:9.6f}"
    )
