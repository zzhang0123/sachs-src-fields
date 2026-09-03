#!/usr/bin/env python
"""Cross-check: S = T22[xi_+], D = T2-2[xi_-] computed with the manuscript's OWN
analysis3 module (import, no re-implementation) on its ELL grid, for the four
sweeps; plus PyCCL's Order-0 C_l^{kk} for the O0 gate.  Saves
crosscheck_analysis3.npz, read by n2_products_eb.py.

Run with the PyCCL interpreter:
  /opt/homebrew/Caskroom/miniconda/base/envs/PyCCL/bin/python n2_crosscheck_analysis3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
A3 = Path("/Users/zzhang/Documents/MyDrafts/STF_lensing/SFT-lensing-paper-analyses"
          "/sachs_sft/analyses/analysis3")
sys.path.insert(0, str(A3))
import plot_analysis3_cl_decomposition as C  # noqa: E402

RUNS = C.RUNS
SWEEPS = {
    "O0": (C.O0_NPZ, 0),
    "FF": (C.FF_NPZ, 2),
    "FK": (C.FK_NPZ, 2),
    "FK_june": (RUNS / "C_corr_op_K_limber_FK" / "xi_C_corr_op_K_limber_FK.npz", 2),
}


def main() -> int:
    C._selftest_wigner()
    out = {"ELL": C.ELL}
    g_ref = None
    for tag, (path, order) in SWEEPS.items():
        g, grouped = C.load_sweep_order(path, order)
        if g_ref is None:
            g_ref = g
            s22 = C.build_curved_matrix(g, C.ELL, 2, 2)
            s2m2 = C.build_curved_matrix(g, C.ELL, 2, -2)
        assert np.allclose(g, g_ref)
        xip = C._combine(grouped, [((1, 1), +1.0), ((2, 2), +1.0)])
        xim = C._combine(grouped, [((1, 1), +1.0), ((2, 2), -1.0)])
        out[f"S_{tag}"] = C.forward_curved(xip, s22)
        out[f"D_{tag}"] = C.forward_curved(xim, s2m2)
        ee = 0.5 * (out[f"S_{tag}"] + out[f"D_{tag}"])
        bb = 0.5 * (out[f"S_{tag}"] - out[f"D_{tag}"])
        r = bb / ee
        m = (C.ELL >= 50) & (C.ELL <= 1500)
        print(f"{tag:8s} BB/EE: l=60 {r[C.ELL == 60][0]:+.4f}  l=1500 {r[C.ELL == 1500][0]:+.4f}  "
              f"median(50-1500) {np.median(r[m]):+.4f}")
    out["gamma_arcmin"] = g_ref
    ckk = C._pyccl_ckk()
    if ckk is not None:
        out["ckk_pyccl"] = ckk
        print("pyccl C_kk available; O0 S/pyccl at l=60,1500:",
              out["S_O0"][C.ELL == 60][0] / ckk[C.ELL == 60][0],
              out["S_O0"][C.ELL == 1500][0] / ckk[C.ELL == 1500][0])
    np.savez(HERE / "crosscheck_analysis3.npz", **out)
    print(f"-> {HERE / 'crosscheck_analysis3.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
