"""
Spin of the LNO-AFQMC wavefunction of one fragment along imaginary time
=======================================================================

Quintet [Fe(H2O)6]2+ (UHF, density fitted), one IAO fragment per atom, and only the
first fragment, the Fe atom, is solved: unrestricted LNO-AFQMC/pt2CCSD (trial="upt2ccsd")
under the UHF guide. save_snapshots writes the fragment's walker population at tau = 0
and at every printed row of the run, as AfqmcMixed(save_wavefunction=dir) does for a
canonical calculation; the snapshots are then read back and <S^2> and <S_z> of the
wavefunction are evaluated along tau (trot/wavefunction_spin.py).

In an LNO calculation both occupied and virtual orbitals are frozen. The walkers live in
the fragment's active LNOs only, so in the whole LNO basis of a spin, ordered
[frozen occ | active | frozen vir], a walker W (n_active, n_active_occ) is the determinant

    [[ 1, 0 ],      frozen occupied   (n_frozen_occ, n_frozen_occ) identity
     [ 0, W ],      active
     [ 0, 0 ]]      frozen virtual

of all the electrons of that spin: the frozen occupied LNOs stay occupied, the frozen
virtual ones stay empty. Each snapshot carries the LNO coefficients and these three index
sets per spin, and S^2 is evaluated on those full determinants in the AO basis. Alpha and
beta have their own LNO bases and their own frozen sets.

Run from the repository root:  python examples/lnopt2/lnoafqmc_wavefunction_spin.py
"""

# importing trot.lnoafqmc before jax initialises selects the allocator that hands device
# memory back between fragments (trot/lnoafqmc/__init__.py); it must come before trot.afqmc
import trot.lnoafqmc  # noqa: F401
from pathlib import Path

import numpy as np
from pyscf import gto, lib, scf
from pyscf.data.elements import chemcore

from trot.lnoafqmc import LnoAfqmcMixed, iao_fragment
from trot.wavefunction_io import load_wavefunction, walker_full_coefficients
from trot.wavefunction_spin import (
    guide_from_snapshots,
    occupied_ao_orbitals,
    spin_along_tau,
    transition_spin,
)

atom = """
Fe -0.64147387529051 0.51990405379180 0.11450483185168
O 0.33915564253394 2.18819453520299 -0.84159903476570
H 0.65823736209818 2.99799985286513 -0.40575899522591
O -1.29914529040912 -0.20924466129350 -1.80322550106302
H -2.11419692217693 0.03267722517314 -2.27701147100162
O 0.01237770373627 1.24396717111997 2.03533113813669
H 0.83097217272839 1.00756306178418 2.50577204385612
O -1.62284672685044 -1.15048542622041 1.07007991103536
H -1.94592790838151 -1.95766677806096 0.63237073377438
O -2.41866907277644 1.66373032346654 0.25658821238371
H -2.55381601393967 2.56024091790599 -0.09930153080408
O 1.13404022498401 -0.62623224778001 -0.02436092147139
H 1.26370551457750 -1.52400168391753 0.33034524187694
H -3.28110369794222 1.35873477131317 0.59094893393907
H -0.81032015579443 -0.82248643308581 -2.37966496653734
H 1.99795010138314 -0.32692539271158 -0.36001715946551
H 0.57998322602479 2.26642229799559 -1.78148703231078
H -0.47668197075682 1.85673194452426 2.61208351578631
H -1.85653031374812 -1.23356353207299 2.011352050005
"""

# quintet: 2S = 4
mol = gto.M(atom=atom, basis="def2-svp", charge=2, spin=4, verbose=3)
mf = scf.UHF(mol).density_fit()  # the fragment integrals come from the DF tensor
mf.max_cycle = 200
mf.kernel()

# follow the UHF solution down to a stable one
for _ in range(5):
    mo_i, _, stable, _ = mf.stability(return_status=True)
    if stable:
        break
    mf.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))
s2_uhf, _ = mf.spin_square()
print(f"UHF energy: {mf.e_tot:.10f}   <S^2> = {s2_uhf:.6f}")

nfrozen = int(chemcore(mol))

# one fragment per atom; for a UHF mean field lo_coeff is an (alpha, beta) pair and each
# fragment a pair of LO lists. Fe is the first atom, so its fragment is the first one
lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen, frag_type="atom")
print(f"{len(frag_list)} fragments; solving fragment 1: {frag_name[0]}")

snap_root = Path("./lno_wfn_snaps")

lno = LnoAfqmcMixed(
    mf,
    lo_coeff,
    frag_list,
    frag_name=frag_name,
    run_frag=[0],  # the Fe fragment only
    lno_thresh=1e-4,
    nfrozen=nfrozen,
    trial="upt2ccsd",  # the default for a UHF mf; guide=None -> the UHF guide
    n_walkers=100,
    tau_eql=4,
    n_blocks=100,
    dt=0.005,
    seed=27,
    mixed_precision=False,
    save_snapshots=snap_root,  # one subdirectory per fragment: snap_root/snapshots{i}
)
lno.kernel()
print(f"\nfragment 1 [{frag_name[0]}]: LNO sizes (alpha, beta) = {lno.lno_size[0]}")
print(f"  E(LNO-MP2)   = {lno.lno_emp[0]:.8f}")
print(f"  E(LNO-CCSD)  = {lno.lno_ecc[0]:.8f}")
print(f"  E(LNO-AFQMC) = {lno.lno_eqmc[0]:.6f} +/- {lno.lno_eqmc_err[0]:.6f}")

# ---- the snapshots of the Fe fragment
snap_dir = snap_root / "snapshots1"
files = sorted(snap_dir.glob("wfn_*.h5"))
wf = load_wavefunction(files[-1])
print(f"\n{len(files)} snapshots in {snap_dir}; the last one:")
for spin, key in (("alpha", "basis_a"), ("beta", "basis_b")):
    b = wf[key]
    print(
        f"  {spin:5s} {b.kind} basis: {b.coeff.shape[1]} orbitals = {b.frozen_occ.size} frozen occ "
        f"+ {b.norb} active + {b.frozen_vir.size} frozen vir"
    )
print(f"  walkers_a {wf['walkers_a'].shape}, walkers_b {wf['walkers_b'].shape}")

# the walkers as determinants of all the electrons in the whole LNO basis: the block
# matrix of the header, identity on the frozen occupied LNOs, zero on the frozen virtual
full_a, full_b = walker_full_coefficients(wf)  # (n_walkers, nmo, n_frozen_occ + nocc)
for spin, full, key, wkey in (
    ("alpha", full_a, "basis_a", "walkers_a"),
    ("beta", full_b, "basis_b", "walkers_b"),
):
    b = wf[key]
    ncore = b.frozen_occ.size
    print(f"  {spin:5s} full walker {full.shape[1:]}:")
    print(
        f"    frozen occ block = identity: {np.allclose(full[:, b.frozen_occ, :ncore], np.eye(ncore))}"
        f",  active block = walker: {np.array_equal(full[:, b.active, ncore:], wf[wkey])}"
        f",  frozen vir rows = 0: {not full[:, b.frozen_vir, :].any()}"
    )
# the same determinants in the AO basis, which is what the spin analysis works with
phi_a, phi_b = occupied_ao_orbitals(wf)
assert np.allclose(lib.einsum("pq,wqi->wpi", wf["basis_a"].coeff, full_a), phi_a)
assert np.allclose(lib.einsum("pq,wqi->wpi", wf["basis_b"].coeff, full_b), phi_b)
print(
    f"  electrons per walker (alpha, beta) = ({phi_a.shape[2]}, {phi_b.shape[2]}); mol.nelec = {mol.nelec}"
)

# ---- <S^2> and <S_z> along tau
s_ao = mol.intor("int1e_ovlp")
# the walkers start from the guide, so walker 0 at tau = 0 is the UHF determinant: the
# frozen occupied and the active occupied LNOs span the UHF occupied space of each spin
guide_a, guide_b = guide_from_snapshots(snap_dir)
m = 0.5 * (guide_a.shape[1] - guide_b.shape[1])
_, s2_guide = transition_spin(guide_a, guide_b, guide_a, guide_b, s_ao)
print(f"\nUHF guide from the snapshots: <S^2> = {s2_guide.real:.6f} (pyscf: {s2_uhf:.6f})")
print(f"a pure S = {m:.0f} state has <S^2> = {m * (m + 1):.0f}, <S_z> = {m:.0f}\n")

rows = spin_along_tau(snap_dir, s_ao, guide=(guide_a, guide_b))
print(f"{'tau':>7s} {'phase':>7s} {'<S^2> mixed':>12s} {'<S^2> pure':>12s} {'<S_z>':>9s}")
for r in rows:
    print(
        f"{r['tau']:7.2f} {r['phase']:>7s} {r['s2_mixed']:12.6f} {r['s2_pure']:12.6f} "
        f"{r['sz']:9.6f}"
    )
