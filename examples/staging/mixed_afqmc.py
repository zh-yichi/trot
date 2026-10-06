"""
Run AfqmcMixed from the file mixed_stage.py wrote, with no mean field or CC object.

The guide and the trial are the ones the file was staged for (guide="uhf",
trial="upt2ccsd_bar"); the QMC settings are chosen here.
"""

from trot.afqmc import AfqmcMixed

STAGED_PATH = "o2_mixed.h5"

af = AfqmcMixed.from_staged(
    STAGED_PATH,
    n_walkers=100,
    n_blocks=50,
    tau_eql=2,
    seed=7,
    mixed_precision=False,
    # save_wavefunction="./wfn_snaps",  # the file carries the MO basis the snapshots need
)
e, err = af.kernel()
print(f"AFQMC/upt2CCSD (UHF guide) energy: {e:.6f} +/- {err:.6f}")
print(f"  guide energy: {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f}")
