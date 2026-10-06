"""
Stage an AfqmcMixed run on the CPU: triplet O2 in cc-pVDZ, UHF guide, upt2ccsd_bar trial.

The mean field and the CCSD are done here, and save_staged writes everything that guide
and trial need to one h5 file: the hamiltonian (h0, h1, cholesky vectors), the guide's
data, the pt2CCSD amplitudes, and the MO basis for the wavefunction files. Nothing is
put on a device. mixed_afqmc.py then runs the AFQMC from that file alone.
"""

from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed

STAGED_PATH = "o2_mixed.h5"

mol = gto.M(atom="O 0 0 0; O 0 0 1.20577", basis="ccpvdz", spin=2, verbose=3)
mf = scf.UHF(mol).density_fit()
mf.kernel()

# follow the UHF solution down to a stable one
for _ in range(5):
    mo_i, _, stable, _ = mf.stability(return_status=True)
    if stable:
        break
    mf.kernel(dm0=mf.make_rdm1(mo_i, mf.mo_occ))

# the frozen core of the hamiltonian follows cc.frozen
mycc = cc.CCSD(mf).set_frozen()
mycc.kernel()

# the guide and the trial decide what is staged; chol_cut is fixed here as well
af = AfqmcMixed(mycc, guide="uhf", trial="upt2ccsd_bar", chol_cut=1e-5)
af.save_staged(STAGED_PATH)
