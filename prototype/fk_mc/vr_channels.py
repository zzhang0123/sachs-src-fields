"""Physical FK (and O0, FF) channels of the convergence 2-point xi_kappa(gamma),
via the paper's variance-reduced estimators, at physical amplitude.

FK is isolated by the VR estimator (Q pulled outside the ensemble average) -- the
only unbiased way to the FK diagram (see the Chinese note in the session). O0 is
the analytic Order-0; FF is the connected F-vertex channel (F-on/F-off CRN).
Saves xi_kappa(gamma) per channel to outputs/fk_channels.npz for the C_l step.
"""
from __future__ import annotations

import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field")
from prototype.fk_mc import _setup  # noqa: E402
mc, ds = _setup.mc, _setup.ds

OUT = "/Users/zzhang/projects/SFT/src-field/prototype/fk_mc/outputs"


def order0_full(cosg, builder, lam_f, n_gauss=128):
    """Analytic Order-0 (3,3): O0_ab = int Sigma2_ab(cos; la) G(la)^2 dla."""
    from numpy.polynomial.legendre import leggauss
    nodes, w = leggauss(int(n_gauss))
    la = 0.5 * (lam_f - builder.lam_lo) * nodes + 0.5 * (lam_f + builder.lam_lo)
    jw = 0.5 * (lam_f - builder.lam_lo) * w
    sig = np.array([builder.matrix(cosg, float(l)) for l in la])     # (nla,3,3)
    G = ds.order0_window(builder.bg, la, lam_f, n_gauss=n_gauss)      # (nla,)
    return np.einsum("k,kab->ab", jw * G * G, sig)


def run(n_real=40_000, n_lambda=600, gammas=None):
    if gammas is None:
        gammas = np.geomspace(1.0, 5000.0, 26)   # to 83 deg, the analysis-3 range
    base = dict(n_real=n_real, batch_size=4_000, n_lambda=n_lambda,
                sigma_lambda=8.0, use_f_vertex=True, skew_scale=1.0)
    b = ds._builder()
    lam_f = float(mc._bg.LAM_SOURCE_BASELINE)
    ng = len(gammas)
    o0 = np.zeros((ng, 3, 3))
    ff = np.zeros((ng, 3, 3)); ff_se = np.zeros((ng, 3, 3))
    fk = np.zeros((ng, 3, 3)); fk_se = np.zeros((ng, 3, 3))
    print(f"VR channels (full 3x3): n_real={n_real}, n_lambda={n_lambda}, {ng} gammas")
    print(f"{'gamma':>8} {'O0_kk':>11} {'FF_kk':>11} {'FK_kk':>11} {'FK_++':>11} {'|FK_kk/O0|':>10}")
    t0 = time.time()
    for i, g in enumerate(gammas):
        cosg = float(np.cos(np.deg2rad(g / 60.0)))
        o0[i] = order0_full(cosg, b, lam_f, n_gauss=128)
        rff = mc.simulate_ff_crn(mc.MCConfig(seed=101, **base), g)
        ff[i] = np.asarray(rff.ff_conn); ff_se[i] = np.asarray(rff.ff_moment_err)
        rvr = mc.simulate_fk_vr(mc.MCConfig(seed=202, **base), g)
        fk[i] = np.asarray(rvr.fk); fk_se[i] = np.asarray(rvr.fk_err)
        fk_plus = fk[i, 1, 1] + fk[i, 2, 2]
        rat = abs(fk[i, 0, 0] / o0[i, 0, 0]) if o0[i, 0, 0] else float("nan")
        print(f"{g:>8.1f} {o0[i,0,0]:>11.2e} {ff[i,0,0]:>11.2e} {fk[i,0,0]:>11.2e} "
              f"{fk_plus:>11.2e} {rat:>10.1f}")
    print(f"  done in {time.time()-t0:.0f}s")
    import os
    os.makedirs(OUT, exist_ok=True)
    np.savez(f"{OUT}/fk_channels.npz", gamma=np.asarray(gammas), o0=o0,
             ff=ff, ff_se=ff_se, fk=fk, fk_se=fk_se,
             n_real=n_real, n_lambda=n_lambda)
    print(f"  saved {OUT}/fk_channels.npz  (o0/ff/fk shape {o0.shape})")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-real", type=int, default=40_000)
    p.add_argument("--n-lambda", type=int, default=600)
    a = p.parse_args()
    run(n_real=a.n_real, n_lambda=a.n_lambda)
