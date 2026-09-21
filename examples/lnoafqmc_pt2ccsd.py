"""
LNO-AFQMC/pt2CCSD with trot.lnoafqmc
====================================

A chain of N2 dimers (sto-6g, RHF, density fitted), one IAO fragment per atom. Per
fragment, kernel() runs
    1) make_las        the local natural orbitals of the fragment (pyscf-forge)
    2) LNO-MP2         fragment MP2 energy
    3) LNO-CCSD        fragment CCSD energy and the t1/t2 the trial needs
    4) LNO-AFQMC       an AfqmcMixed run on the fragment hamiltonian: the HF guide staged
                       in the LNO basis, the fragment energy measured against the pt2CCSD
                       trial on the similarity transformed hamiltonian (the exp(T1) "bar"
                       estimator, chunked over the cholesky index)
with steps 1-3 of the next fragment on a background thread while step 4 runs on the GPU.
The fragment energies add up to the LNO-AFQMC correlation energy; with a tight
lno_thresh every local active space is the whole active space and the sum reproduces
the full-space AfqmcMixed(trial="pt2ccsd_bar") run within error bars.

Run from the repository root:  python examples/lnoafqmc_pt2ccsd.py
"""

import os
from typing import Any

os.environ["OMP_NUM_THREADS"] = "1"  # LNO orbitals are thread sensitive; keeps runs reproducible

import trot.lnoafqmc  # noqa: F401  before jax: chooses the allocator that frees memory between fragments
from pyscf import cc, gto, mp, scf
from pyscf.data.elements import chemcore

from trot.afqmc import AfqmcMixed
from trot.lnoafqmc import LnoAfqmcMixed, LnoFragMixed, iao_fragment

a = 2.2  # intra-dimer bond length (Bohr)
d = 5  # centre-to-centre distance between dimers (Bohr)
na, nc = 2, 2  # atoms per monomer, number of monomers

atoms = ""
for n in range(nc * na):
    shift = ((n - n % na) // na) * (d - a)
    atoms += f"N {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto-6g", spin=0, unit="B", verbose=3)
mf: Any = scf.RHF(mol).density_fit()  # the fragment integrals come from the DF tensor
mf.kernel()
nfrozen = int(chemcore(mol))

# IAO local orbitals grouped by atom ("h2heavy" attaches hydrogens to their heavy atom),
# optionally localized further ("pm" / "boys"). The LOs must span the occupied orbitals
# outside the frozen core; LnoAfqmcMixed checks.
lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen, frag_type="h2heavy", more_loc="pm")

lno = LnoAfqmcMixed(
    mf,
    lo_coeff,
    frag_list,
    frag_name=frag_name,
    lno_thresh=1e-5,  # float -> [10 x, x] for occ / vir; or give [thresh_occ, thresh_vir]
    nfrozen=nfrozen,
    run_frag=None,  # e.g. [0] to run one fragment
    trial="pt2ccsd",  # guide=None -> the HF guide; guide="cisd" propagates with the fragment CISD
    target_error=1e-3,  # each fragment stops once its error < 0.7 * target / sqrt(nfrag),
    #                     after at least min_blocks=120 sampling blocks; None runs all n_blocks
    n_walkers=200,
    n_eql_blocks=40,
    n_blocks=400,
    dt=0.005,
    seed=27,
    # mixed_precision=True by default: single precision T2 contractions, double sums
    # max_memory=4000,  # MB budget of the trial measurement; by default a share of the device
    frag_output="./fragment.out",  # -> ./fragment.out1, ./fragment.out2, ... (optional)
    lno_output="./lno_result.out",  # the results table (optional)
    save_frag_data="./frag_data",  # -> ./frag_data/frag{i}.h5, re-runnable without mf (optional)
)
e_qmc, e_qmc_err = lno.kernel()

# the canonical references
mymp = mp.MP2(mf, frozen=nfrozen)
mymp.kernel()
mycc = cc.CCSD(mf, frozen=nfrozen)
mycc.kernel()

print(f"\nLNO-MP2   E_corr = {lno.e_mp:.8f}     MP2  E_corr = {mymp.e_corr:.8f}")
print(f"LNO-CCSD  E_corr = {lno.e_cc:.8f}     CCSD E_corr = {mycc.e_corr:.8f}")
print(f"LNO-AFQMC E_corr = {e_qmc:.6f} +/- {e_qmc_err:.6f}")
print(f"LNO-AFQMC E_corr + dMP2 = {e_qmc + mymp.e_corr - lno.e_mp:.6f} +/- {e_qmc_err:.6f}")
print("per fragment:", lno.lno_eqmc, lno.lno_eqmc_err, "sizes", lno.lno_size)

# the full-space AFQMC with the same trial (bar estimator), for comparison
af = AfqmcMixed(mycc, trial="pt2ccsd_bar", n_walkers=200, n_eql_blocks=40, n_blocks=400, seed=27)
e_ref, err_ref = af.kernel()
print(f"\nAFQMC/pt2CCSD E_corr (full space) = {e_ref - mf.e_tot:.6f} +/- {err_ref:.6f}")

# one fragment again, from its file: no LNO, CCSD or integral work, straight to the QMC
frag = LnoFragMixed.from_frag_data("./frag_data/frag1.h5", n_walkers=200, n_blocks=200, seed=27)
e1, err1 = frag.kernel()
print(f"\nfragment 1 re-run: {e1:.6f} +/- {err1:.6f}")
