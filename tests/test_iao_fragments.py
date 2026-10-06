"""IAO reference basis from free-atom SCF (trot.lnoafqmc.fragments)."""

import numpy as np
import pytest
from pyscf import gto, lo, scf

pytest.importorskip("pyscf.lno")
from trot.lnoafqmc.fragments import (  # noqa: E402
    _is_ccpvxz_family,
    free_atom_minao,
    iao_fragment,
    iao_labels,
    resolve_minao,
)


def test_free_atom_minao_fe_has_4s():
    # pyscf's atomic SCF default configuration (NRSRHF) fills Fe as 3d8 4s0; the reference
    # basis must come from the physical 3d6 4s2 ground state so the 4s is included
    mol = gto.M(atom="Fe 0 0 0; O 0 0 2.0", basis={"Fe": "cc-pwcvtz-dk", "O": "cc-pvdz-dk"},
                spin=4, verbose=0)
    assert not _is_ccpvxz_family(mol.basis)
    minao = free_atom_minao(mol)
    assert [sh[0] for sh in minao["Fe"]] == [0, 0, 0, 0, 1, 1, 2]
    assert [sh[0] for sh in minao["O"]] == [0, 0, 1]
    labels = iao_labels(mol, minao)
    fe = [l.split()[2] for l in labels if l.startswith("0 Fe")]
    assert fe[:4] == ["1s", "2s", "3s", "4s"] and sum(l.startswith("3d") for l in fe) == 5
    assert len(labels) == 15 + 5


def test_resolve_minao_keeps_minao_for_ccpvxz():
    mol = gto.M(atom="O 0 0 0; H 0 0 1; H 0 1 0", basis="cc-pvdz", verbose=0)
    assert resolve_minao(mol) == "minao"
    assert resolve_minao(mol, {"O": [[0, [1.0, 1.0]]]}) == {"O": [[0, [1.0, 1.0]]]}


@pytest.mark.parametrize("basis", ["6-31g", "cc-pvdz"])
def test_iao_fragment_uhf_populations_sum_to_nelec(basis):
    mol = gto.M(atom="O 0 0 0; H 0 0 1; H 0 1 0", basis=basis, spin=2, verbose=0)
    mf = scf.UHF(mol).run()
    lo_coeff, frag_list, frag_name = iao_fragment(mf, nfrozen=0, frag_type="atom")
    s1e = mf.get_ovlp()
    labels = iao_labels(mol)
    assert lo_coeff[0].shape[1] == len(labels) == 7  # O 1s 2s 2p + 2 H 1s
    dm = mf.make_rdm1()
    pop = sum(np.einsum("pi,pq,qi->i", c, s1e @ d @ s1e, c).sum() for c, d in zip(lo_coeff, dm))
    assert abs(pop - mol.nelectron) < 1e-8
    assert frag_name == ["O0", "H1", "H2"]
    if basis == "6-31g":  # the free-atom route: reference mol must reproduce the labels
        ref = lo.iao.reference_mol(mol, resolve_minao(mol))
        assert ref.ao_labels() == labels
