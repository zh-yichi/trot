"""
The stoccsd guide on a data mesh: the sharded run of AfqmcMixed reproduces the
unsharded one (same seed, same keys, the reconfiguration gathers by global index), and
the state's samples are split over the devices with the walkers. Needs several devices:
    JAX_PLATFORM_NAME=cpu XLA_FLAGS=--xla_force_host_platform_device_count=4 pytest tests/test_stoccsd_sharding.py
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

from trot import config

config.configure_once()

import contextlib
import io

import jax
import numpy as np
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from pyscf import cc, gto, scf

from trot.afqmc import AfqmcMixed
from trot.prop.afqmc_stocc import PropStateStocc
from trot.sharding import make_data_mesh

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(
    jax.device_count() < 2, reason="needs more than one device"
)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def h4():
    mol = gto.M(
        atom="H 0 0 0; H 0 0 1.2; H 0 0 2.4; H 0 0 3.6", basis="sto-3g", verbose=0
    )
    mf = scf.RHF(mol).run()
    return cc.CCSD(mf).run()


def _run(h4, mesh, trial):
    af = AfqmcMixed(
        h4,
        guide="stoccsd",
        trial=trial,
        guide_n_samples=3,
        n_walkers=8,
        n_blocks=6,
        n_eql_blocks=2,
        n_prop_steps=4,
        seed=5,
        chol_cut=1e-8,
        mixed_precision=False,
    )
    e, err = _quiet(af.kernel, mesh=mesh)
    return af


@pytest.mark.parametrize("trial", ["rhf", "pt2ccsd_bar"])
def test_sharded_run_matches_unsharded(h4, trial):
    mesh = make_data_mesh()
    af_u = _run(h4, None, trial)
    af_s = _run(h4, mesh, trial)
    r_u, r_s = af_u.qmc_result, af_s.qmc_result
    np.testing.assert_allclose(
        r_s.guide_block_energies, r_u.guide_block_energies, rtol=1e-9, atol=1e-10
    )
    np.testing.assert_allclose(
        r_s.trial_block_weights, r_u.trial_block_weights, rtol=1e-9
    )
    for name in r_u.trial_block_components:
        np.testing.assert_allclose(
            r_s.trial_block_components[name],
            r_u.trial_block_components[name],
            rtol=1e-9,
            atol=1e-10,
        )
    np.testing.assert_allclose(af_s.e_tot, af_u.e_tot, rtol=1e-9)
    state = r_s.final_state
    assert isinstance(state, PropStateStocc)
    assert state.guide_dets.shape == (8, 3, 4, 2)
    for a in (state.walkers, state.weights, state.overlaps, state.guide_dets):
        assert isinstance(a.sharding, NamedSharding) and a.sharding.spec == P("data")


def test_walker_count_must_divide_the_mesh(h4):
    mesh = make_data_mesh()
    af = AfqmcMixed(
        h4,
        guide="stoccsd",
        trial="rhf",
        guide_n_samples=2,
        n_walkers=mesh.size + 1,
        n_blocks=1,
        n_eql_blocks=1,
        seed=5,
    )
    with pytest.raises(ValueError, match="multiple of the mesh"):
        _quiet(af.kernel, mesh=mesh)
