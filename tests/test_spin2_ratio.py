"""Pin the spin-2 / spin-0 power ratio of the driving fields.

The draft's per-multipole multipliers for fields sourced by one scalar
potential are L^2/chi^2 (Phi_00, screen Laplacian) and sqrt(L^2 (L^2-2))/chi^2
(Psi_0, trace-free screen Hessian), L^2 = l(l+1). Hence

    C_l^{Psi_0} / C_l^{Phi_00} = (L^2 - 2) / L^2 = (l+2)(l-1) / (l(l+1)).

perFLRW used the inverse of this until 2026-09-03 (found in the B-mode audit,
prototype/bmode_audit/A1/REPORT.md, item 9 of section 4). The ratio is pure
arithmetic, so it is tested without pyccl by importing the helper through a
stub of the pyccl module when pyccl is absent.
"""
from __future__ import annotations

import importlib
import sys
import types

import numpy as np


def _load_helper():
    try:
        import pyccl  # noqa: F401
    except ImportError:  # the helper does not need pyccl; stub the import
        sys.modules.setdefault("pyccl", types.ModuleType("pyccl"))
    mod = importlib.import_module("perFLRW.cosmology")
    return mod.spin2_power_ratio


def test_ratio_matches_squared_multipliers():
    ratio = _load_helper()
    ell = np.arange(2, 3001, dtype=float)
    L2 = ell * (ell + 1.0)
    mult_phi00 = L2
    mult_psi0 = np.sqrt(L2 * (L2 - 2.0))
    np.testing.assert_allclose(ratio(ell), (mult_psi0 / mult_phi00) ** 2, rtol=1e-13)


def test_ratio_is_below_one_and_tends_to_one():
    ratio = _load_helper()
    ell = np.array([2.0, 10.0, 45.0, 1000.0])
    r = ratio(ell)
    assert np.all(r < 1.0)
    assert r[0] == 2.0 / 3.0                      # (4*1)/(2*3)
    assert abs(1.0 - r[2]) < 1.0e-3               # 2/L^2 at l = 45
    assert abs(1.0 - r[3]) < 2.1e-6               # 2/L^2 at l = 1000
