"""Brute-force FK Monte-Carlo via sachsray (no variance reduction).

For each separation gamma, run two INDEPENDENT ensembles through sachsray's
stable Jacobi: Gaussian (skew_scale=0) and skewed (skew_scale=1, the exact
canoes zeta injected by the demo2 deformation). The non-Gaussian (FK) signal in
the convergence 2-point is

    xi_FK(gamma) = <kappa(n1) kappa(n2)>_skew - <kappa(n1) kappa(n2)>_gauss,

at PHYSICAL amplitude, accepting the (large) brute-force noise floor. sachsray's
linear Jacobi stays finite on the heavy-tailed skewed driving that overflows the
paper's nonlinear Riccati -- the redesign's robustness win.
"""
from __future__ import annotations

import sys
import warnings
from dataclasses import replace

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field")
from prototype.fk_mc import _setup  # noqa: E402
from prototype.fk_mc.route import gen_field, kappa_sachsray  # noqa: E402
mc = _setup.mc


def accumulate_xi(cfg, gar, n_total, batch, seed0):
    """Batched CONNECTED cross-covariance <k0 k1>_c = <k0 k1> - <k0><k1>.

    Returns (xi_connected, se_jackknife, finite_fraction, mean_kappa). Per-batch
    connected estimates give a jackknife SE that is robust to the heavy-tailed
    skewed kappa (the disconnected <k0><k1> carries the Levy-Q mean inflation).
    """
    cosg = float(np.cos(np.deg2rad(gar / 60.0)))
    s0 = s1 = s01 = 0.0
    n = 0
    n_seen = 0
    batch_xi = []
    for b in range(max(1, n_total // batch)):
        cfg_b = replace(cfg, n_real=batch, batch_size=batch, seed=seed0 + b)
        g = gen_field(cfg_b, cosg, batch, seed0 + b)
        k0 = kappa_sachsray(g, 0)
        k1 = kappa_sachsray(g, 1)
        fin = np.isfinite(k0) & np.isfinite(k1)
        k0, k1 = k0[fin], k1[fin]
        nb = k0.size
        if nb > 1:
            batch_xi.append(float(np.mean(k0 * k1) - np.mean(k0) * np.mean(k1)))
        s0 += float(k0.sum()); s1 += float(k1.sum()); s01 += float((k0 * k1).sum())
        n += nb; n_seen += fin.size
    m0, m1 = s0 / max(n, 1), s1 / max(n, 1)
    xi_c = s01 / max(n, 1) - m0 * m1
    bx = np.array(batch_xi)
    se = float(bx.std() / np.sqrt(len(bx))) if len(bx) > 1 else float("nan")
    return xi_c, se, n / max(n_seen, 1), m0


def fk_at_gamma(cfg_base, gar, n_total, batch):
    xi_g, se_g, fin_g, mg = accumulate_xi(replace(cfg_base, skew_scale=0.0),
                                          gar, n_total, batch, seed0=1_000_000)
    xi_s, se_s, fin_s, ms = accumulate_xi(replace(cfg_base, skew_scale=1.0),
                                          gar, n_total, batch, seed0=9_000_000)
    return dict(gamma=gar, xi_gauss=xi_g, se_gauss=se_g, xi_skew=xi_s,
                se_skew=se_s, fk=xi_s - xi_g, fk_se=float(np.hypot(se_g, se_s)),
                finite_gauss=fin_g, finite_skew=fin_s, mean_skew=ms)


def _scan():
    n_total, batch = 16_000, 4_000
    cfg = mc.MCConfig(n_lambda=300, sigma_lambda=8.0, use_f_vertex=True)
    pap = np.load(_setup.PAPER_SACHS_SFT
                  + "/analyses/mc_sachs_2pt/outputs/appendix_mc_curve.npz")
    print(f"brute-force CONNECTED FK via sachsray (n_total={n_total}/ensemble, n_lambda={cfg.n_lambda})")
    print(f"{'gamma':>7} {'xi_gauss_c':>11} {'xi_skew_c':>11} {'FK_conn':>11} {'FK_SE':>10} "
          f"{'FK/SE':>6} {'paper FK':>10} {'brute/pap':>9} {'<k>_skew':>10}")
    for gar in (10.0, 100.0, 300.0, 600.0, 1000.0, 1500.0, 1944.0):
        r = fk_at_gamma(cfg, gar, n_total, batch)
        i = int(np.argmin(np.abs(pap["g_mc"] - gar)))
        pfk = float(pap["fk_mc"][i])
        sn = abs(r["fk"]) / r["fk_se"] if r["fk_se"] and np.isfinite(r["fk_se"]) else float("nan")
        ratio = r["fk"] / pfk if pfk else float("nan")
        print(f"{gar:>7.1f} {r['xi_gauss']:>11.2e} {r['xi_skew']:>11.2e} "
              f"{r['fk']:>11.2e} {r['fk_se']:>10.2e} {sn:>6.1f} "
              f"{pfk:>10.2e} {ratio:>9.2f} {r['mean_skew']:>10.2e}")


if __name__ == "__main__":
    _scan()
