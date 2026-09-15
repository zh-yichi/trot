"""
Example: the pt2CCSD energy kernel with exp(T1) applied to the right
====================================================================

The pt2CCSD trial is exp(T1)|HF> plus a perturbative T2. Where exp(T1) sits is a choice:

  "pt2ccsd_chunk"  keeps it on the trial. The trial orbitals are mo_t = exp(T1)|HF>, and
                   the greens function against them is a full (norb, norb) matrix.

  "pt2ccsd_bar"    moves it onto the hamiltonian and the walker instead. The trial
                   becomes the bare reference determinant, and the hamiltonian is
                   similarity transformed:

                 exp_t1   = 1 + X,  X[:nocc, nocc:] = t1     (X**2 = 0, so this is exact)
                 exp_mt1  = 1 - X
                 h1_bar   = exp_t1 @ h1   @ exp_mt1
                 chol_bar = exp_t1 @ chol @ exp_mt1
                 walker_bar = exp_t1 @ walker

Both compute the same estimator and must agree to round-off; this example checks that
rather than asserting it. What "bar" buys is the greens function: measured against the
bare reference, only its first nocc rows are nonzero, so the kernel carries an
(nocc, norb) half green and every chunk intermediate shrinks with it. What it costs is
chol_bar, a second copy of the cholesky tensor.

Only the trial side is transformed. The guide stays RHF and its propagator still sees
the bare hamiltonian -- chol_bar lives on the measurement context, not in ham_data.

See examples/pt2ccsd_chunked.py for the chunking and max_memory machinery this reuses.
"""

from trot import config

config.configure_once()

import jax.numpy as jnp
import numpy as np
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed
from trot.meas.pt2ccsd import (
    Pt2ccsdMeasCfg,
    build_bar_intermediates,
    build_meas_ctx,
    energy_kernel_rw_rh,
    energy_kernel_rw_rh_bar,
    energy_kernel_rw_rh_chunk,
    plan_pt2ccsd_chunking,
    pt2ccsd_memory_model,
)
from trot.trial.pt2ccsd import overlap_r

MB = 1024**2

# =============================================================================
# Molecular system: 8 H2 dimers, far enough apart to be non-interacting
# =============================================================================

a, d = 2, 100  # intra-dimer bond length, centre-to-centre distance (Bohr)
na, nc = 2, 8  # atoms per monomer, number of monomers

atoms = ""
for n in range(nc * na):
    shift = ((n - n % na) // na) * (d - a)
    atoms += f"H {n*a+shift:.5f} 0.00000 0.00000 \n"

mol = gto.M(atom=atoms, basis="sto6g", unit="b", verbose=0)

mf = scf.RHF(mol)
mf.kernel()
mycc = cc.CCSD(mf)
mycc.kernel()
print(f"RHF  energy: {mf.e_tot:.10f} Ha")
print(f"CCSD energy: {mycc.e_tot:.10f} Ha")

# =============================================================================
# Selecting the bar kernel
# =============================================================================
# The trial name is the whole switch: it picks the energy kernel, and with it whatever
# the measurement context has to prepare. guide names the mean field that propagates the
# walkers, and is checked against the recipe rather than assumed. build_job() stages and
# builds the context without running, which is where the transformed tensors get made.

af = AfqmcMixed(
    mycc,
    trial="pt2ccsd_bar",
    guide="rhf",
    max_memory=20,  # MB, per device -- sizes the cholesky chunk as usual
    dt=0.005,
    n_walkers=200,
    n_blocks=200,
    n_eql_blocks=40,
    seed=17,
    mixed_precision=True,
)

job = af.build_job()
ham_data = job.ham_data
trial_data = job.mix_trial_data
meas_ctx = job.mix_meas_ctx()

nocc, nvir, norb = trial_data.nocc, trial_data.nvir, trial_data.norb
print(f"\nnorb={norb}  nocc={nocc}  nvir={nvir}  nchol={int(ham_data.nchol)}")
print(f"plan: {job.chunk_plan.describe()}")

# =============================================================================
# The intermediates
# =============================================================================
# build_bar_intermediates derives t1 from the staged trial, so nothing extra has to be
# staged. trot stores mo_t as exp_t1[:nocc].T -- occupied block the identity, virtual
# block t1.T -- and the gauge is divided out anyway, so any equivalent mo_t works.

bar = build_bar_intermediates(ham_data, trial_data)
exp_t1, exp_mt1 = bar["exp_t1"], bar["exp_mt1"]
eye = jnp.eye(norb)

print("\nintermediates")
print(f"  exp_t1 @ exp_mt1 == 1        : {bool(jnp.allclose(exp_t1 @ exp_mt1, eye))}")
print(f"  mo_t == exp_t1[:nocc].T      : {bool(jnp.allclose(trial_data.mo_t, exp_t1[:nocc, :].T))}")
print(f"  h1_bar == exp_t1 h1 exp_mt1  : "
      f"{bool(jnp.allclose(bar['h1_bar'], exp_t1 @ ham_data.h1 @ exp_mt1))}")
print(f"  chol_bar[g] likewise         : "
      f"{bool(jnp.allclose(bar['chol_bar'][0], exp_t1 @ ham_data.chol[0] @ exp_mt1))}")
# a similarity transform cannot move the spectrum
h1_bar_eigs = jnp.sort(jnp.linalg.eigvals(bar["h1_bar"]).real)
h1_eigs = jnp.sort(jnp.linalg.eigvalsh(ham_data.h1))
print(f"  h1 eigenvalues unchanged     : {bool(jnp.allclose(h1_bar_eigs, h1_eigs))}")
# how far the transform moves the tensor is set by t1, which is tiny for dimers this far
# apart -- a strongly correlated system would show a much larger shift. Either way
# ham_data keeps the bare tensors: the transformed copies live on the measurement
# context, which only the trial estimator ever sees.
print(f"  |chol_bar - chol| (~ |t1|)   : "
      f"{float(jnp.max(jnp.abs(bar['chol_bar'] - ham_data.chol))):.3e}")
print(f"  meas_ctx holds chol_bar      : {meas_ctx.chol_bar is not None}")
print(f"  guide still propagates on    : ham_data.chol, shape {tuple(ham_data.chol.shape)}")

# =============================================================================
# The overlap is the same object either way
# =============================================================================
#   det(mo_t.T @ walker) == det((exp_t1 @ walker)[:nocc])
# so trial.overlap_r needs no bar variant, and the kernel returns only (t2, e0, e1).

rng = np.random.default_rng(0)
walker_0 = jnp.asarray(job.trial_data.mo_coeff) + 0.0j  # the RHF guide determinant
noise = jnp.asarray(rng.normal(size=walker_0.shape) + 1j * rng.normal(size=walker_0.shape))
walkers = {"tau=0": walker_0, "perturbed": walker_0 + 0.1 * noise}

print("\noverlap")
for name, w in walkers.items():
    o_bar = jnp.linalg.det((exp_t1 @ w)[:nocc, :]) ** 2
    print(f"  {name:<10} |det(mo @ walker_bar) - det(mo_t @ walker)| = "
          f"{abs(o_bar - overlap_r(w, trial_data)):.2e}")

# =============================================================================
# All three kernels agree
# =============================================================================
# Same (t2, e0, e1) out of every one, for any chunk size, including one that overshoots
# nchol and pads the tail.

print("\nkernels, max |bar - other| over (t2, e0, e1)")
ctx_plain = build_meas_ctx(ham_data, trial_data, Pt2ccsdMeasCfg())  # measure_type=None
for name, w in walkers.items():
    ref = np.asarray(energy_kernel_rw_rh(w, ham_data, ctx_plain, trial_data))
    for k in (1, 5, int(ham_data.nchol) + 3):
        cfg_bar = Pt2ccsdMeasCfg(measure_type="bar", nchol_chunk=k)
        cfg_chunk = Pt2ccsdMeasCfg(measure_type="chunk", nchol_chunk=k)
        ctx_bar = build_meas_ctx(ham_data, trial_data, cfg_bar)
        ctx_chunk = build_meas_ctx(ham_data, trial_data, cfg_chunk)
        out = np.asarray(energy_kernel_rw_rh_bar(w, ham_data, ctx_bar, trial_data))
        chk = np.asarray(energy_kernel_rw_rh_chunk(w, ham_data, ctx_chunk, trial_data))
        # asking for more than nchol is clamped, which is how the tail chunk gets padded
        print(f"  {name:<10} nchol_chunk={k:<3} (used {ctx_bar.nchol_chunk:<3}) "
              f"vs plain {np.max(np.abs(out - ref)):.2e}"
              f"   vs chunked {np.max(np.abs(out - chk)):.2e}")

# =============================================================================
# The memory trade
# =============================================================================
# bar pays resident memory (chol_bar and its half rotation) to make the chunk term
# cheap. Worth it when nocc << norb, which is the regime that needs chunking at all --
# the example molecule above is far too small to show it, so this is a realistic size.

print("\nmemory model at norb=400, nocc=60, nchol=1600, 200 walkers, 24 GB budget")
for is_bar in (False, True):
    m = pt2ccsd_memory_model(norb=400, nocc=60, nchol=1600, n_walkers=200, bar=is_bar)
    p = plan_pt2ccsd_chunking(m, n_walkers=200, nchol=1600, budget_bytes=24_000 * MB)
    print(f"  {'bar' if is_bar else 'plain':<6} resident {m.resident / MB:7.0f} MB"
          f"   per_walker_chol {m.per_walker_chol / MB:6.3f} MB"
          f"   -> nchol_chunk={p.nchol_chunk}, n_chunks={p.n_chunks}")

# =============================================================================
# Run it
# =============================================================================

mean, err = af.kernel()
print(f"\nGuide  (AFQMC/RHF)    : {af.guide_e_tot:.6f} +/- {af.guide_e_err:.6f} Ha")
print(f"Trial  (AFQMC/pt2CCSD): {mean:.6f} +/- {err:.6f} Ha")
print(f"Reference (CCSD)      : {mycc.e_tot:.6f} Ha")
