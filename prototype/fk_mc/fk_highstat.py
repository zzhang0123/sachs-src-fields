"""Re-run ONLY the FK (VR) channel at high statistics, to resolve the small
shear-difference channels xi_- = xi_11 - xi_22 (= zeta_TPP, gamma^4-suppressed)
and xi_{kappa gamma_t} = -xi_01 (parity-forbidden), where the paper predicts the
FK contribution to be ~0. O0 (analytic, exact) and FF are kept from the existing
fk_channels.npz; only fk/fk_se are overwritten.

    python prototype/fk_mc/fk_highstat.py --n-real 300000
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field")
from prototype.fk_mc import _setup  # noqa: E402
mc = _setup.mc

OUT = "/Users/zzhang/projects/SFT/src-field/prototype/fk_mc/outputs"


def run(n_real=300_000, n_lambda=600):
    d = dict(np.load(f"{OUT}/fk_channels.npz"))
    g = d["gamma"]
    base = dict(n_real=n_real, batch_size=4_000, n_lambda=n_lambda,
                sigma_lambda=8.0, use_f_vertex=True, skew_scale=1.0)
    fk = np.zeros((len(g), 3, 3))
    fk_se = np.zeros((len(g), 3, 3))
    print(f"high-stat FK (VR): n_real={n_real}, n_lambda={n_lambda}, {len(g)} gammas")
    print(f"{'gamma':>8} {'xi_+ (11+22)':>14} {'xi_- (11-22)':>14} {'SE(xi_-)':>11} "
          f"{'xi_-/SE':>8} {'kE (01)':>11} {'SE(kE)':>10}")
    t0 = time.time()
    for i, gar in enumerate(g):
        r = mc.simulate_fk_vr(mc.MCConfig(seed=202, **base), gar)
        fk[i] = np.asarray(r.fk); fk_se[i] = np.asarray(r.fk_err)
        xip = fk[i, 1, 1] + fk[i, 2, 2]
        xim = fk[i, 1, 1] - fk[i, 2, 2]
        se_m = np.hypot(fk_se[i, 1, 1], fk_se[i, 2, 2])   # conservative (no CRN credit)
        print(f"{gar:>8.1f} {xip:>14.3e} {xim:>14.3e} {se_m:>11.1e} "
              f"{abs(xim)/se_m if se_m else float('nan'):>8.1f} "
              f"{fk[i,0,1]:>11.3e} {fk_se[i,0,1]:>10.1e}")
    print(f"  done in {time.time()-t0:.0f}s")
    d["fk"] = fk
    d["fk_se"] = fk_se
    d["n_real_fk"] = n_real
    np.savez(f"{OUT}/fk_channels.npz", **d)
    print(f"  updated {OUT}/fk_channels.npz (fk at n_real={n_real}; o0/ff kept)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=300_000)
    p.add_argument("--n-lambda", type=int, default=600)
    a = p.parse_args()
    run(n_real=a.n_real, n_lambda=a.n_lambda)
