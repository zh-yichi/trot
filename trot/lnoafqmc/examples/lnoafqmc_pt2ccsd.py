"""
Example: LNO-AFQMC/pt2CCSD with trot.lnoafqmc
=============================================

O2 (singlet RHF, sto-3g, density fitted) split into two atomic fragments. With a tight
LNO threshold each fragment's local active space is the whole active space, so the two
fragment energies add up to the full-space AFQMC/pt2CCSD correlation energy: exactly at
tau = 0 (the CCSD energy), within error bars after sampling. Loosen lno_thresh on a real
system to get the truncated local active spaces LNO is for.

Per fragment, kernel() runs
    1) make_las        the LNOs of the fragment
    2) LNO-MP2         fragment MP2 energy
    3) LNO-CCSD        fragment CCSD energy and the t1/t2 the trial needs
    4) LNO-AFQMC       an AfqmcMixed run on the fragment hamiltonian: the HF guide staged
                       by trot in the LNO basis, the fragment energy measured against the
                       pt2CCSD trial (the exp(T1)-transformed "bar" estimator)
with steps 1-3 of the next fragment on a background thread while step 4 runs on the GPU.

Run from the repository root:  python trot/lnoafqmc/examples/lnoafqmc_pt2ccsd.py
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"  # LNO orbitals are thread sensitive; keeps runs reproducible

import trot.lnoafqmc  # noqa: F401  before jax: chooses the allocator that frees memory between fragments
from pyscf import cc, gto, scf, mp
from pyscf.data.elements import chemcore

from trot.afqmc import AfqmcMixed
from trot.lnoafqmc import LnoAfqmcMixed, LnoFragMixed, iao_fragment

a = 2  # intra-dimer bond length (Bohr)
d = 5  # centre-to-centre distance between dimers (Bohr)
na, nc = 2, 4  # atoms per monomer, number of monomers

atoms = ""
for n in range(nc * na):
    shift = ((n - n % na) // na) * (d - a)
    atoms += f"N {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto-6g", spin=0, unit='b', verbose=4)
mf = scf.RHF(mol).density_fit()  # the fragment integrals come from the DF tensor
mf.kernel()
nfrozen = chemcore(mol)

# IAO local orbitals grouped by atom ("h2heavy" attaches hydrogens to their heavy atom).
# The LOs must span the occupied orbitals outside the frozen core; LnoAfqmcMixed checks.
# lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen, frag_type="atom")

# lno = LnoAfqmcMixed(
#     mf,
#     lo_coeff,
#     frag_list,
#     frag_name=frag_name,
#     lno_thresh=1e-6,  # float -> [10 x, x] for occ / vir; or give [thresh_occ, thresh_vir]
#     nfrozen=nfrozen,
#     run_frag=None,  # e.g. [0] to run one fragment
#     trial="pt2ccsd",  # guide=None -> the HF guide. "pt2ccsd_sto_chol" samples the T2-contracted
#     #                   cholesky sum: add e.g. trial_kwargs={"chol_cost_ratio": 0.2} (the default;
#     #                   head : samples = 3 : 1), or n_chol_head / n_chol_samples, as in AfqmcMixed
#     target_error=1e-5,  # each fragment stops once its error < 0.7 * target / sqrt(nfrag),
#     #                     after at least min_blocks=120 sampling blocks
#     n_walkers=300,
#     n_eql_blocks=80,
#     n_blocks=600,
#     dt=0.005,
#     seed=27,
#     frag_output="./fragment.out",  # -> ./fragment.out1, ./fragment.out2 (optional)
#     lno_output="./lno_result.out",  # the results table (optional)
#     # save_frag_data="./frag_data",  # -> ./frag_data/frag{i}.h5, re-runnable without mf
# )
# e_qmc, e_qmc_err = lno.kernel()

# print(f"\nE(LNO-MP2)   = {lno.e_mp:.8f}")
# print(f"E(LNO-CCSD)  = {lno.e_cc:.8f}")
# print(f"E(LNO-AFQMC) = {e_qmc:.6f} +/- {e_qmc_err:.6f}")
# print("per fragment:", lno.lno_eqmc, lno.lno_eqmc_err, "sizes", lno.lno_size)

# the full-space reference, AfqmcMixed with the same trial (bar estimator)
mymp = mp.MP2(mf, frozen=nfrozen)
mymp.kernel()

mycc = cc.CCSD(mf, frozen=nfrozen)
mycc.kernel()

af = AfqmcMixed(mycc, trial="pt2ccsd_bar", norb_frozen_core=nfrozen, n_walkers=300, n_eql_blocks=80, n_blocks=1200, seed=27)
e_ref, err_ref = af.kernel()

# print(f"\nLNO-AFQMC E_corr = {e_qmc:.6f} +/- {e_qmc_err:.6f}")
# print(f"LNO-MP2   E_corr = {lno.e_mp:.8f}")
# print(f"    MP2   E_corr = {mymp.e_corr:.8f}")
# print(f"LNO-CCSD  E_corr = {lno.e_cc:.8f}")
# print(f"    CCSD  E_corr = {mycc.e_corr:.8f}")
# print(f"LNO-AFQMC E_corr + dMP2 = {e_qmc + mymp.e_corr - lno.e_mp:.6f} +/- {e_qmc_err:.6f}")
print(f"AFQMC E_corr  = {e_ref - mf.e_tot:.6f} +/- {err_ref:.6f}")

# # one fragment again, from its file: no LNO, CCSD or integral work, straight to the QMC
# frag = LnoFragMixed.from_frag_data("./frag_data/frag1.h5", n_blocks=300, seed=27, n_walkers=300, n_eql_blocks=80)
# e1, err1 = frag.kernel()
# print(f"\nfragment 1 re-run: {e1:.6f} +/- {err1:.6f}")
