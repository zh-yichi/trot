"""
Mixed guide/trial AFQMC with the stochastically sampled CCSD guide: every walker is
guided by its own sample of n_samples determinants exp(y.tau)|exp(T1)HF>, redrawn at
every step; the energy is measured against the trial named on the command line (here
the RHF trial, as in the original C2 experiment; any of trot.mixed.available_trials()
on the restricted hamiltonian works, e.g. "cisd" or "pt2ccsd_bar").

    python stoccsd_guide.py [guide] [n_samples] [seed]     guide: rhf | cisd | stoccsd

C2 at 2 bohr in STO-6G with the core frozen, 300 walkers, dt = 0.005, 50 steps per
block, 80 equilibration blocks (tau = 20) and 420 sampling blocks.
"""

import sys

from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed

guide = sys.argv[1] if len(sys.argv) > 1 else "stoccsd"
n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 10
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 17
trial = sys.argv[4] if len(sys.argv) > 4 else "rhf"

mol = gto.M(atom="C 0 0 0; C 2.0 0 0", basis="sto6g", unit="B", verbose=0)
mf = scf.RHF(mol).run()
mycc = cc.CCSD(mf).set_frozen()
mycc.kernel()
print(
    f"E_RHF = {mf.e_tot:.6f}   E_CCSD = {mycc.e_tot:.6f}   E_CCSD(T) = {mycc.e_tot + mycc.ccsd_t():.6f}"
)

options = dict(guide_n_samples=n_samples, guide_t2_thresh=1e-6) if guide == "stoccsd" else {}
af = AfqmcMixed(
    mycc,
    guide=guide,
    trial=trial,
    n_walkers=300,
    dt=0.005,
    n_prop_steps=50,
    n_eql_blocks=80,
    n_blocks=420,
    seed=seed,
    mixed_precision=False,
    **options,
)
e, err = af.kernel()
label = f"{guide}({n_samples})" if guide == "stoccsd" else guide
print(
    f"\nRESULT guide={label:<12s} trial={trial:<8s} seed={seed}  E = {e:.5f} +/- {err:.5f}   "
    f"guide E = {af.guide_e_tot:.5f} +/- {af.guide_e_err:.5f}"
)
