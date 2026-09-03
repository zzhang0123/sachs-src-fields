"""Spatial diagnostic of the zs16 shear B-mode: compare the B-mode map with the
B-mode implied by the rotation map (B_lm = s sqrt(F_l) omega_lm, s = sign of
C^{omega B}) at lmax 1024, look at the latitude profile of their variances and
of the residual, and draw Mollweide maps. Output: outputs/bmap_diag.json,
outputs/fig_bmaps.png."""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import healpy as hp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
sys.path.insert(0, "/Users/zzhang/projects/SFT-WL-B/analysis/wpc_routeb")
from cb2_takahashi_io import read_field  # noqa: E402

PATH = "/Users/zzhang/projects/SFT-WL-B/data/allskymap_nres12r000.zs16.mag.dat"
LMAX = 1024
NSIDE_OUT = 512


def main():
    t0 = time.time()
    g1 = read_field(PATH, "gamma1").astype(np.float64)
    g2 = read_field(PATH, "gamma2").astype(np.float64)
    alm_e, alm_b = hp.map2alm_spin([g1, g2], 2, lmax=LMAX)
    del g1, g2
    w = read_field(PATH, "omega").astype(np.float64)
    alm_w = hp.map2alm(w, lmax=LMAX, iter=0)
    del w
    print(f"transforms {time.time() - t0:.0f}s")
    l = hp.Alm.getlm(LMAX)[0].astype(float)
    F = np.zeros_like(l)
    ok = l >= 2
    F[ok] = (l[ok] + 2) * (l[ok] - 1) / (l[ok] * (l[ok] + 1))
    cl_wb = hp.alm2cl(alm_w, alm_b)
    cl_ww = hp.alm2cl(alm_w)
    cl_bb = hp.alm2cl(alm_b)
    sign = np.sign(np.sum(cl_wb[20:400]))
    alm_bw = sign * np.sqrt(F) * alm_w  # rotation-implied B-mode
    # band-limit everything identically (2 <= l <= LMAX) and drop l < 20 for the maps
    fl = np.ones(LMAX + 1)
    fl[:20] = 0.0
    alm_b_f = hp.almxfl(alm_b, fl)
    alm_bw_f = hp.almxfl(alm_bw, fl)
    b_map = hp.alm2map(alm_b_f, NSIDE_OUT, lmax=LMAX)
    bw_map = hp.alm2map(alm_bw_f, NSIDE_OUT, lmax=LMAX)
    res_map = b_map - bw_map
    e_map = hp.alm2map(hp.almxfl(alm_e, fl), NSIDE_OUT, lmax=LMAX)
    theta, _ = hp.pix2ang(NSIDE_OUT, np.arange(hp.nside2npix(NSIDE_OUT)))
    lat = 90.0 - np.degrees(theta)
    edges = np.array([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90], dtype=float)
    prof = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (lat >= lo) & (lat < hi)
        prof.append(dict(lat=[lo, hi], var_B=float(np.var(b_map[sel])), var_Bomega=float(np.var(bw_map[sel])),
                         var_resid=float(np.var(res_map[sel])), var_E=float(np.var(e_map[sel])),
                         corr_B_Bomega=float(np.corrcoef(b_map[sel], bw_map[sel])[0, 1])))
    r_band = []
    for lo, hi in [(20, 50), (50, 100), (100, 200), (200, 400), (400, 800), (800, 1024)]:
        r_band.append(dict(band=[lo, hi], r_wB=float(cl_wb[lo:hi].mean() / np.sqrt(cl_ww[lo:hi].mean() * cl_bb[lo:hi].mean()))))
    out = dict(lmax=LMAX, nside_out=NSIDE_OUT, sign_wB=float(sign), latitude_profile=prof, r_wB=r_band,
               summary=dict(var_B=float(np.var(b_map)), var_Bomega=float(np.var(bw_map)), var_resid=float(np.var(res_map)),
                           corr_B_Bomega=float(np.corrcoef(b_map, bw_map)[0, 1]), corr_resid_Bomega=float(np.corrcoef(res_map, bw_map)[0, 1])))
    with open(os.path.join(OUT, "bmap_diag.json"), "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))
    fig = plt.figure(figsize=(14, 9))
    vmax = 3 * np.std(b_map)
    hp.mollview(b_map, sub=(2, 2, 1), title=r"shear B-mode map, zs16, $20\leq\ell\leq1024$", min=-vmax, max=vmax, cmap="RdBu_r")
    hp.mollview(bw_map, sub=(2, 2, 2), title=r"rotation-implied B-mode $s\sqrt{F_\ell}\,\omega_{\ell m}$", min=-vmax, max=vmax, cmap="RdBu_r")
    hp.mollview(res_map, sub=(2, 2, 3), title="residual B - B_omega", min=-vmax, max=vmax, cmap="RdBu_r")
    hp.mollview(e_map, sub=(2, 2, 4), title=r"shear E-mode map (same band)", min=-3 * np.std(e_map), max=3 * np.std(e_map), cmap="RdBu_r")
    fig.savefig(os.path.join(OUT, "fig_bmaps.png"), dpi=110)


if __name__ == "__main__":
    main()
