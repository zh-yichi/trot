"""
Example: unrestricted LNO-AFQMC/pt2CCSD with trot.lnoafqmc
=========================================================

Triplet O2 (UHF, sto-6g, density fitted) split into two atomic fragments. A UHF mean field
selects the unrestricted trial, trial="upt2ccsd": each fragment runs on the uchol fragment
hamiltonian (alpha and beta each in their own LNO basis, one joint cholesky index from
trot.cholesky.joint_df2chol), with unrestricted walkers under the UHF guide, and the
fragment energy measured against the unrestricted pt2CCSD trial (the exp(T1)-transformed
"bar" estimator).

With a tight LNO threshold each fragment's local active space is the whole active space,
so the two fragment energies add up to the full-space AFQMC/pt2CCSD correlation energy:
exactly at tau = 0 (the UCCSD energy), within error bars after sampling. With a looser
threshold the alpha and beta active spaces generally differ in size; lno_size and
lno_nocc are then (alpha, beta) pairs per fragment.

Run from the repository root:  python trot/lnoafqmc/examples/lnoafqmc_upt2ccsd.py
"""

import os

# os.environ["OMP_NUM_THREADS"] = "1"  # LNO orbitals are thread sensitive; keeps runs reproducible

import trot.lnoafqmc  # noqa: F401  before jax: chooses the allocator that frees memory between fragments
from pyscf import cc, gto, scf, mp
from pyscf.data.elements import chemcore

from trot.afqmc import AfqmcMixed
from trot.lnoafqmc import LnoAfqmcMixed, LnoFragMixed, iao_fragment

a = 1.20577  # intra-dimer bond length (Bohr)
d = 100  # centre-to-centre distance between dimers (Bohr)
na, nc = 2, 2  # atoms per monomer, number of monomers

atoms = ""
for n in range(nc * na):
    shift = ((n - n % na) // na) * (d - a)
    atoms += f"O {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto-6g", spin=2*nc, verbose=4)
mf = scf.UHF(mol).density_fit()  # the fragment integrals come from the DF tensor
mf.kernel()

stable = False
while not stable:
    print(f'mean-field stability test')
    if not stable:
        mo_i, _, stable,_ = mf.stability(return_status=True)
        dm = mf.make_rdm1(mo_i,mf.mo_occ)
        mf.kernel(dm0=dm)
    elif stable:
        print(f'HF Energy: {mf.e_tot}, stability {stable}')
        break

nfrozen = chemcore(mol)

# for a UHF mean field lo_coeff is an (alpha, beta) pair and each fragment a pair of LO lists
lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen, frag_type="atom")

lno = LnoAfqmcMixed(
    mf,
    lo_coeff,
    frag_list,
    frag_name=frag_name,
    lno_thresh=1e-5,
    nfrozen=nfrozen,
    trial="upt2ccsd",  # the default for a UHF mf; guide=None -> the UHF guide.
    #                    "upt2ccsd_sto_chol" samples the T2-contracted cholesky sum; its knobs go in
    #                    trial_kwargs, e.g. {"chol_cost_ratio": 0.2}, as in AfqmcMixed\
    run_frag = [0, 1],
    target_error=1e-4,
    n_walkers=300,
    n_eql_blocks=80,
    n_blocks=600,
    dt=0.005,
    seed=27,
    mixed_precision = True,
    # frag_output="./fragment.out",
    # lno_output="./lno_result.out",
    # save_frag_data="./frag_data",
)
e_qmc, e_qmc_err = lno.kernel()

print(f"\nE(LNO-MP2)   = {lno.e_mp:.8f}")
print(f"E(LNO-CCSD)  = {lno.e_cc:.8f}")
print(f"E(LNO-AFQMC) = {e_qmc:.6f} +/- {e_qmc_err:.6f}")
print("per fragment:", lno.lno_eqmc, lno.lno_eqmc_err, "sizes (alpha, beta)", lno.lno_size)

# the full-space reference, AfqmcMixed with the same trial (bar estimator)
mymp = mp.MP2(mf, frozen=nfrozen)
mymp.kernel()

mycc = cc.UCCSD(mf, frozen=nfrozen)
mycc.kernel()
af = AfqmcMixed(mycc, trial="upt2ccsd_bar", norb_frozen_core=nfrozen, n_walkers=300, n_eql_blocks=80, n_blocks=600, seed=27)
e_ref, err_ref = af.kernel()

print(f"\nLNO-AFQMC E_corr = {e_qmc:.6f} +/- {e_qmc_err:.6f}")
print(f"LNO-MP2   E_corr = {lno.e_mp:.8f}")
print(f"    MP2   E_corr = {mymp.e_corr:.8f}")
print(f"LNO-CCSD  E_corr = {lno.e_cc:.8f}")
print(f"    CCSD  E_corr = {mycc.e_corr:.8f}")
# print(f"LNO-AFQMC E_corr + dMP2 = {e_qmc + mymp.e_corr - lno.e_mp:.6f} +/- {e_qmc_err:.6f}")
print(f"AFQMC E_corr  = {(e_ref - mf.e_tot)/2:.6f} +/- {err_ref/2:.6f}")

# one fragment again, from its file: no LNO, UCCSD or integral work, straight to the QMC
# frag = LnoFragMixed.from_frag_data("./frag_data/frag1.h5", n_blocks=300, seed=27, n_walkers=300, n_eql_blocks=80)
# e1, err1 = frag.kernel()
# print(f"\nfragment 1 re-run: {e1:.6f} +/- {err1:.6f}")
