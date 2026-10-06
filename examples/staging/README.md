# Staged CPU/GPU Workflow

This example directory shows how to split AFQMC into two stages:

1. run the PySCF calculation and write the staged AFQMC inputs to disk
2. read the staged file later and run AFQMC

This is useful when the trial generation step is expensive and is done on CPU, while the AFQMC propagation is best run on GPU. Large systems are a common case: building the Hamiltonian and trial can take a long time, so it is convenient to separate the CPU and GPU calculations.

## Files

- `cisd.py`: runs a PySCF RHF + CCSD calculation and writes `h2o_afqmc.h5` with a staged CISD trial
- `afqmc.py`: reads the staged file and runs a single device AFQMC calculation
- `afqmc_multi_gpu.py`: reads the staged file and runs AFQMC with sharding across all visible GPUs
- `mixed_stage.py`: runs a PySCF UHF + UCCSD calculation and writes `o2_mixed.h5` for an `AfqmcMixed` run (UHF guide, `upt2ccsd_bar` trial) with `AfqmcMixed.save_staged`
- `mixed_afqmc.py`: reads that file with `AfqmcMixed.from_staged` and runs the guide + trial AFQMC without any PySCF object

## Typical Usage

First create the staged input on CPU:

```bash
python cisd.py
```

Then run AFQMC from the staged file:

```bash
python afqmc.py
```

Or run on multiple GPUs:

```bash
python afqmc_multi_gpu.py
```

## Sharding details

The multi-GPU example here uses the `data` axis only.

- `data` shards the walker data across devices
- each device still keeps the full Hamiltonian and trial data for the run

In code, this is just:

```python
from trot.sharding import make_data_mesh
mesh = make_data_mesh()
```

For large Hamiltonians, there is also a `data x model` mesh:

- `data` shards the walker batch
- `model` shards the Hamiltonian Cholesky axis

The intended picture is:

- walker-like state such as walkers, weights, and overlaps is split across the `data` axis
- large Hamiltonian objects such as `chol` are split across the `model` axis
- small global quantities such as the RNG key and energy shift stay replicated

Some useful examples are:

- `make_data_model_mesh(n_data=2, n_model=1)`: pure walker sharding across 2 GPUs
- `make_data_model_mesh(n_data=1, n_model=2)`: pure Hamiltonian/model sharding across 2 GPUs
- `make_data_model_mesh(n_data=2, n_model=2)`: split both walkers and Hamiltonian across 4 GPUs

The main use case for the `model` axis is large Hamiltonians where `chol` or related tensors would otherwise not fit comfortably on a single GPU. For small and medium problems, pure `data` sharding is usually simpler.

One practical detail is that changing `n_model` changes the floating-point reduction order along the Cholesky axis. So runs with different `model` layouts can differ slightly at the level of roundoff when using mixed precision, but they are usually statistically consistent.
