"""
Example: the pt2CCSD energy kernel with a semistochastic cholesky sum
=====================================================================

trial="pt2ccsd_sto_chol" is the bar estimator (see examples/pt2ccsd_bar.py) with the
expensive half of the two-body energy sampled instead of summed.

What is exact and what is sampled:

  e2_0, and with it e2_2_1 = e2_0 * gt2g     exact, every cholesky vector. It only needs
                                            gl = green.chol, and the sampling proposal is
                                            built from it anyway
  e2_2_2_1, e2_2_2_2, e2_2_3                split into an exactly summed head and an
                                            importance sampled tail. These carry the
                                            "iajb" contractions, nocc^2 nvir^2 per
                                            cholesky vector, and are what the cost is

The proposal is pi_g ~ |e2_0_g|, mixed with a uniform floor so that every pi_g > 0 --
a vector that could never be drawn but still contributes to the energy would be a bias,
not merely extra variance. Tail draws carry weight 1 / (n_samples * pi_g), so the
estimator is unbiased: averaging over keys converges to the exact value, as checked below.

Sizing the head and the tail (all through trial_kwargs):

  n_chol_head      head size; "full" puts every vector in the head, which removes the
                   sampling entirely and reproduces pt2ccsd_bar exactly -- the in-place
                   reference run
  head_chol_ratio  head as a fraction of nchol, when n_chol_head == 0. Default 0.125
  n_chol_samples   tail draws per walker per block. Default 128
  chol_cost_ratio  set a per-walker budget C = ratio * nchol instead, split
                   head : samples = head_sample_ratio : 1 (default 3:1)

Memory: the head is a contiguous prefix, so it is a plain slice shared across the walkers,
and the sampled tail is scanned over *indices* with the gather inside the scan body. The
drawn vectors differ per walker, so materialising them would cost
n_walkers * n_samples * norb^2 all at once -- and n_samples is not bounded by nchol_chunk,
so shrinking the chunk would not recover it. Measured at the bottom.
"""

from trot import config

config.configure_once()

import contextlib
import io

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed
from trot.core.system import System
from trot.ham.chol import HamChol
from trot.meas.pt2ccsd import (
    Pt2ccsdMeasCfg,
    build_meas_ctx,
    energy_kernel_rw_rh,
    energy_kernel_rw_rh_bar,
    energy_kernel_rw_rh_sto,
    resolve_chol_budget,
)
from trot.staging import stage, stage_pt2ccsd_trial
from trot.trial.pt2ccsd import Pt2ccsdTrial
from trot import walkers as wk

MB = 1024**2

# =============================================================================
# A hydrogen chain, big enough that sampling has something to sample
# =============================================================================

mol = gto.M(
    atom="; ".join(f"H 0 0 {1.6 * i}" for i in range(10)),
    basis="631g",
    unit="b",
    verbose=0,
)
mf = scf.RHF(mol)
mf.kernel()
mycc = cc.CCSD(mf)
mycc.kernel()
print(f"RHF  energy: {mf.e_tot:.10f} Ha")
print(f"CCSD energy: {mycc.e_tot:.10f} Ha")

staged = stage(mf)
ham = staged.ham
sys = System(norb=int(ham.norb), nelec=ham.nelec, walker_kind="restricted")
ham_data = HamChol(
    jnp.asarray(ham.h0), jnp.asarray(ham.h1), jnp.asarray(ham.chol), basis=ham.basis
)
trial_input = stage_pt2ccsd_trial(mycc)
trial_data = Pt2ccsdTrial(
    mo_t=jnp.array(trial_input.data["mo_t"]), t2=jnp.array(trial_input.data["t2"])
)
nchol = int(ham_data.nchol)
print(f"\nnorb={trial_data.norb}  nocc={trial_data.nocc}  nchol={nchol}")

# =============================================================================
# How the knobs resolve
# =============================================================================
# resolve_chol_budget is pure arithmetic, so the split can be inspected before running
# anything. Each half falls back independently: an explicit setting wins, then
# chol_cost_ratio's split, then the defaults.

print("\nhead / tail split")
print(f"  {'setting':<46} {'head':>6} {'samples':>8} {'cost/nchol':>11}")
for label, kw in (
    ("defaults", {}),
    ("n_chol_head='full'", {"n_chol_head": "full"}),
    ("head_chol_ratio=0.25", {"head_chol_ratio": 0.25}),
    ("n_chol_samples=32", {"n_chol_samples": 32}),
    ("chol_cost_ratio=0.25", {"chol_cost_ratio": 0.25}),
    ("chol_cost_ratio=0.5, head_sample_ratio=1", {"chol_cost_ratio": 0.5, "head_sample_ratio": 1.0}),
    ("chol_cost_ratio=0.25, n_chol_samples=8", {"chol_cost_ratio": 0.25, "n_chol_samples": 8}),
):
    cfg = Pt2ccsdMeasCfg(measure_type="sto_chol", **kw)
    n_head, n_samp = resolve_chol_budget(
        nchol,
        cfg.n_chol_head,
        cfg.head_chol_ratio,
        cfg.n_chol_samples,
        cfg.chol_cost_ratio,
        cfg.head_sample_ratio,
    )
    # a full head leaves no tail, so the sample count is moot however it resolved
    n_eff = 0 if n_head >= nchol else n_samp
    print(f"  {label:<46} {n_head:>6} {n_eff:>8} {(n_head + n_eff) / nchol:>11.2f}")

# =============================================================================
# The deterministic limit is an exact reference
# =============================================================================
# n_chol_head="full" leaves the tail empty, so no draw is made and no key is needed. It
# must agree with the bar kernel to round-off -- that is the check that the sampled
# machinery is wrapped around the right estimator.

rng = np.random.default_rng(0)
walker_0 = jnp.asarray(staged.trial.data["mo"][:, : sys.nup]) + 0.0j
noise = jnp.asarray(rng.normal(size=walker_0.shape) + 1j * rng.normal(size=walker_0.shape))
walkers = {"tau=0": walker_0, "perturbed": walker_0 + 0.1 * noise}

ctx_plain = build_meas_ctx(ham_data, trial_data, Pt2ccsdMeasCfg())
print('\nn_chol_head="full", max |sto - other| over (t2, e0, e1)')
for name, w in walkers.items():
    ref = np.asarray(energy_kernel_rw_rh(w, ham_data, ctx_plain, trial_data))
    for k in (1, 8, nchol):
        ctx_bar = build_meas_ctx(ham_data, trial_data, Pt2ccsdMeasCfg(measure_type="bar", nchol_chunk=k))
        ctx_sto = build_meas_ctx(
            ham_data, trial_data,
            Pt2ccsdMeasCfg(measure_type="sto_chol", nchol_chunk=k, n_chol_head="full"),
        )
        bar = np.asarray(energy_kernel_rw_rh_bar(w, ham_data, ctx_bar, trial_data))
        sto = np.asarray(energy_kernel_rw_rh_sto(w, ham_data, ctx_sto, trial_data))  # no key
        print(f"  {name:<10} nchol_chunk={k:<3} vs bar {np.max(np.abs(sto - bar)):.2e}"
              f"   vs plain {np.max(np.abs(sto - ref)):.2e}")

# =============================================================================
# The sampled estimator is unbiased
# =============================================================================
# Averaging one walker's energy over many keys must converge to the exact value. t2 and e0
# come out exact every time, since e2_0 is never sampled; only e1 carries the noise.

w = walkers["perturbed"]
exact = np.asarray(energy_kernel_rw_rh(w, ham_data, ctx_plain, trial_data))
n_keys = 1000
print(f"\nunbiasedness, averaging one walker over {n_keys} keys")
# both the bias estimate and its error shrink as 1/sqrt(n_keys), so bias/sem stays of
# order 1 however many keys are drawn -- a few tenths to about 2 is what unbiased looks
# like. It is a systematic offset growing past that which would signal a problem.
print(f"  {'head':>5} {'samples':>8} {'e1 mean':>13} {'e1 exact':>13} {'sem':>10} {'bias/sem':>9}")
for kw in ({"n_chol_samples": 32}, {"n_chol_samples": 256}, {"head_chol_ratio": 0.5, "n_chol_samples": 32}):
    cfg = Pt2ccsdMeasCfg(measure_type="sto_chol", nchol_chunk=8, **kw)
    ctx = build_meas_ctx(ham_data, trial_data, cfg)
    n_head, n_samp = resolve_chol_budget(
        nchol, cfg.n_chol_head, cfg.head_chol_ratio, cfg.n_chol_samples,
        cfg.chol_cost_ratio, cfg.head_sample_ratio,
    )
    f = jax.jit(lambda key: energy_kernel_rw_rh_sto(w, ham_data, ctx, trial_data, key))
    out = np.array([np.asarray(f(k)) for k in jax.random.split(jax.random.PRNGKey(1), n_keys)])
    mean, sem = out.mean(0), out.std(0) / np.sqrt(n_keys)
    assert abs(mean[0] - exact[0]) < 1e-12 and abs(mean[1] - exact[1]) < 1e-12  # t2, e0 exact
    print(f"  {n_head:>5} {n_samp:>8} {mean[2].real:>13.8f} {exact[2].real:>13.8f}"
          f" {sem[2].real:>10.2e} {abs(mean[2] - exact[2]) / sem[2].real:>9.2f}")
print("  (t2 and e0 asserted exact in every case -- e2_0 is never sampled)")

# =============================================================================
# The tail costs indices, not cholesky vectors
# =============================================================================
# Peak temporary memory against n_samples. The growth is the index and weight arrays,
# about 20 bytes per sample per walker. Gathering the drawn vectors up front instead would
# cost norb^2 * 8 bytes each.

n_walk = 64
w_batch = jnp.broadcast_to(walker_0, (n_walk, trial_data.norb, trial_data.nocc))
keys = jax.random.split(jax.random.PRNGKey(0), n_walk)
per_vector = n_walk * trial_data.norb**2 * 8 / 1024  # KiB for one gathered vector, all walkers

print(f"\npeak temporary memory, {n_walk} walkers")
print(f"  {'n_samples':>10} {'peak (KiB)':>12} {'growth':>12} {'if gathered':>13}")
try:
    prev = None
    for n_samp in (16, 64, 256, 1024):
        cfg = Pt2ccsdMeasCfg(
            measure_type="sto_chol", nchol_chunk=8, n_chol_head=8, n_chol_samples=n_samp
        )
        ctx = build_meas_ctx(ham_data, trial_data, cfg)
        f = jax.jit(
            lambda ww, kk: wk.vmap_chunked(
                energy_kernel_rw_rh_sto, n_chunks=1, in_axes=(0, None, None, None, 0)
            )(ww, ham_data, ctx, trial_data, kk)
        )
        peak = f.lower(w_batch, keys).compile().memory_analysis().temp_size_in_bytes / 1024
        growth = "-" if prev is None else f"+{peak - prev:,.0f}"
        would_be = f"+{(n_samp - (prev_n if prev else n_samp)) * per_vector:,.0f}" if prev else "-"
        print(f"  {n_samp:>10} {peak:>12,.0f} {growth:>12} {would_be:>13}")
        prev, prev_n = peak, n_samp
except Exception as exc:  # memory_analysis is backend dependent
    print(f"  (unavailable on this backend: {type(exc).__name__})")

# =============================================================================
# Full runs
# =============================================================================
# The sampling noise adds to the stochastic error of the walk, so the honest comparison is
# against the bar run's error bar, not against CCSD.

print("\nfull runs, 60 blocks")
params = dict(dt=0.005, n_prop_steps=2, n_blocks=60, n_eql_blocks=10, n_walkers=60, seed=7)


def run(label, **kwargs):
    """Each run prints its own banner and flags; swallow them so the table stays legible.
    Drop the redirect to see the flags dump, which names the kernel each trial uses."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        af = AfqmcMixed(mycc, **params, **kwargs)
        mean, err = af.kernel()
    print(f"  {label:<44} {mean:+.6f} +/- {err:.6f} Ha")
    return mean, err


e_bar, s_bar = run('trial="pt2ccsd_bar"', trial="pt2ccsd_bar")
e_ref, _ = run(
    'sto_chol, n_chol_head="full"',
    trial="pt2ccsd_sto_chol",
    trial_kwargs={"n_chol_head": "full"},
)
print(f"    -> reproduces the bar run exactly: {abs(e_ref - e_bar) < 1e-12}")

for n_samp in (16, 64, 256):
    e, s = run(
        f"sto_chol, head=1/8, n_chol_samples={n_samp}",
        trial="pt2ccsd_sto_chol",
        trial_kwargs={"n_chol_samples": n_samp},
    )
    print(f"    -> {abs(e - e_bar) / s:.2f} sigma from the bar run")

print(f"\n  bar reference : {e_bar:+.6f} +/- {s_bar:.6f} Ha")
print(f"  CCSD          : {mycc.e_tot:+.6f} Ha")
