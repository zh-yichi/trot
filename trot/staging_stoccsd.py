"""
Staging of the stochastically sampled CCSD guide (trial/stoccsd.py) from a pyscf CCSD
object: the Thouless reference mo_t = exp(T1)|HF> and the factor tau of the doubles,
t_{iajb} = sum_g tau_{g,ia} tau_{g,jb}, with the modes below t2_thresh dropped.

The TrialInput is of kind "stoccsd" with data {mo_t, tau, t2_thresh}; it dumps and loads
through the staged file format like any other (the data are plain arrays). The number
of determinants per sample is a run parameter, not staged.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .staging import StagedMfOrCc, TrialInput, _stage_pt2ccsd_input
from .trial.stoccsd import decompose_t2

STOCCSD_KIND = "stoccsd"


def stage_stoccsd_guide(
    cc: Any, *, frozen: int | None = None, t2_thresh: float = 1e-6
) -> TrialInput:
    """
    The restricted stoccsd guide of a pyscf CCSD object. mo_t and the (i,a,j,b) doubles
    come from the pt2CCSD stager (staging._stage_pt2ccsd_input), so the frozen core and
    the orbital layout are exactly those of the pt2CCSD trials; the doubles are then
    factorized (trial.stoccsd.decompose_t2).
    """
    pt2 = _stage_pt2ccsd_input(StagedMfOrCc(cc, frozen))
    tau = decompose_t2(np.asarray(pt2.data["t2"]), thresh=float(t2_thresh))
    data = {
        "mo_t": np.asarray(pt2.data["mo_t"]),
        "tau": tau,
        "t2_thresh": np.array(float(t2_thresh)),
    }
    return TrialInput(
        kind=STOCCSD_KIND,
        data=data,
        frozen=pt2.frozen,
        source_kind=pt2.source_kind,
    )


__all__ = ["STOCCSD_KIND", "stage_stoccsd_guide"]
