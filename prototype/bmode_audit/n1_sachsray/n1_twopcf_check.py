"""
Item 6 -- 2PCF version of the theorem check.

For the FK-type cross term (order 2 x order 1) measured by n1_theorem_sachsray.py,
build xi_+ = <g+ g+> + <gx gx> and xi_- = <g+ g+> - <gx gx> from the harmonic
cross-spectra (exact Wigner-d sums),

    xi_+(th) = sum_l (2l+1)/(4pi) (C_EE + C_BB) d^l_{22}(th)
    xi_-(th) = sum_l (2l+1)/(4pi) (C_EE - C_BB) d^l_{2,-2}(th)

then invert BOTH with the paper's curved-sky transform (prototype/fk_mc/figure12.py,
ported verbatim from analysis3) and show that T_22^{-1}[xi_+] = T_{2,-2}^{-1}[xi_-],
i.e. the two real-space combinations carry ONE spectrum and BB = 0.  Also fits the
small-angle power law of xi_- (the d^l_{2,-2} kernel gives gamma^4, whatever the
spectrum) versus xi_+ (-> const).

Usage:  python n1_twopcf_check.py --nside 64 [--case ng] [--tag ""]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO, "prototype", "fk_mc"))
import figure12 as f12  # noqa: E402  (wigner_d, build_curved_matrix, forward_curved)

OUT = os.path.join(HERE, "outputs")


def xi_from_cl(cl_plus, cl_minus, theta_rad):
    """xi_+ from (C_EE + C_BB) with d^l_22, xi_- from (C_EE - C_BB) with d^l_{2,-2}."""
    lmax = len(cl_plus) - 1
    ells = list(range(2, lmax + 1))
    x = np.cos(theta_rad)
    d22 = f12.wigner_d(ells, x, 2, 2)
    d2m2 = f12.wigner_d(ells, x, 2, -2)
    xp = np.zeros_like(theta_rad)
    xm = np.zeros_like(theta_rad)
    for L in ells:
        w = (2 * L + 1) / (4 * math.pi)
        xp += w * cl_plus[L] * d22[L]
        xm += w * cl_minus[L] * d2m2[L]
    return xp, xm


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nside", type=int, default=64)
    p.add_argument("--case", default="ng")
    p.add_argument("--tag", default="")
    p.add_argument("--pair", default="12", help="order pair: 12 (FK-type), 22 (FF-type), 11 (O0)")
    p.add_argument("--smooth", type=float, default=0.0,
                   help="running-mean half-width in dex of log(l) applied to the (noisy, seed-averaged) "
                        "C_l before the transform (the ensemble spectrum is smooth); 0 = none")
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(OUT, f"{args.tag}nside{args.nside}_seed*.npz")))
    if not files:
        sys.exit("no npz results")
    CEE = np.mean([np.load(f)[f"{args.case}/C_EE_{args.pair}"] for f in files], axis=0)
    CBB = np.mean([np.load(f)[f"{args.case}/C_BB_{args.pair}"] for f in files], axis=0)
    lmax = len(CEE) - 1
    L_cut = min(int(1.4 * args.nside), lmax // 2)
    if args.smooth > 0:
        def running(c):
            out = c.copy()
            ell = np.arange(len(c))
            for L in range(2, len(c)):
                m = (ell >= 2) & (np.abs(np.log10(ell / L)) <= args.smooth)
                out[L] = np.mean((2 * ell[m] + 1) * c[m]) / np.mean(2 * ell[m] + 1)
            return out
        CEE, CBB = running(CEE), running(CBB)
        CEE[L_cut + 1:] = 0.0
        CBB[L_cut + 1:] = 0.0

    # --- real-space 2PCFs on the paper's grid and on a full-range grid ---
    gam_paper = np.geomspace(1.0, 5000.0, 26)              # arcmin, as fk_mc/figure12
    gam_full = np.geomspace(1.0, 10800.0, 60)              # to 180 deg
    out = {"nside": args.nside, "case": args.case, "pair": args.pair, "n_seeds": len(files), "L_cut": L_cut,
           "smooth_dex": args.smooth, "band_BB_over_EE_input": float(np.sum((2*np.arange(lmax+1)+1)*CBB) / np.sum((2*np.arange(lmax+1)+1)*CEE))}
    for name, gam in (("paper_grid", gam_paper), ("full_grid", gam_full)):
        th = gam * math.pi / 180.0 / 60.0
        xp, xm = xi_from_cl(CEE + CBB, CEE - CBB, th)
        # small-angle power laws (gamma < 10')
        sm = gam < 10.0
        slope_m = np.polyfit(np.log(gam[sm]), np.log(np.abs(xm[sm])), 1)[0]
        slope_p = np.polyfit(np.log(gam[sm]), np.log(np.abs(xp[sm])), 1)[0]
        # invert with the paper's curved-sky operator (both kernels), ell <= L_cut
        ELL = np.array([L for L in sorted({int(round(v)) for v in np.geomspace(3.0, 1500.0, 30)}) if L <= L_cut], float)
        ELL = np.unique(np.concatenate([ELL, np.array([2.0, 4.0, 6.0, 8.0, 12.0, 16.0, 24.0, 40.0, 60.0, 80.0])]))
        ELL = ELL[ELL <= L_cut]
        S22 = f12.build_curved_matrix(gam, ELL, 2, 2)
        S2m2 = f12.build_curved_matrix(gam, ELL, 2, -2)
        dc = name == "paper_grid"   # the paper subtracts xi at the largest gamma
        Cp = f12.forward_curved(xp, S22, dc_subtract=dc)      # = C_EE + C_BB (recovered)
        Cm = f12.forward_curved(xm, S2m2, dc_subtract=dc)     # = C_EE - C_BB (recovered)
        EE_rec, BB_rec = 0.5 * (Cp + Cm), 0.5 * (Cp - Cm)
        li = ELL.astype(int)
        truth_EE, truth_BB = CEE[li], CBB[li]
        out[name] = {
            "gamma_arcmin": gam.tolist(), "xi_plus": xp.tolist(), "xi_minus": xm.tolist(),
            "slope_xi_minus_small_angle": float(slope_m), "slope_xi_plus_small_angle": float(slope_p),
            "ELL": ELL.tolist(), "C_plus_rec": Cp.tolist(), "C_minus_rec": Cm.tolist(),
            "EE_rec": EE_rec.tolist(), "BB_rec": BB_rec.tolist(),
            "EE_true": truth_EE.tolist(), "BB_true": truth_BB.tolist(),
            "rel_diff_Cplus_Cminus": (np.abs(Cp - Cm) / np.maximum(np.abs(Cp + Cm), 1e-300)).tolist(),
            "BB_rec_over_EE_rec": (BB_rec / EE_rec).tolist(),
            "EE_rec_over_EE_true": (EE_rec / truth_EE).tolist(),
        }
        print(f"\n[{name}] pair {args.pair} case {args.case}: xi_- small-angle slope = {slope_m:.2f} "
              f"(kernel prediction 4), xi_+ slope = {slope_p:.2f} (prediction 0)")
        print(" ell   EE_true      EE_rec/EE_true  |C+ - C-|/|C+ + C-|   BB_rec/EE_rec   BB_true/EE_true")
        for k, L in enumerate(li):
            print(f" {L:4d}  {truth_EE[k]:+.3e}   {EE_rec[k]/truth_EE[k]:+.3f}          "
                  f"{out[name]['rel_diff_Cplus_Cminus'][k]:.2e}            {BB_rec[k]/EE_rec[k]:+.2e}       "
                  f"{truth_BB[k]/truth_EE[k]:+.2e}")
    sfx = f"_s{args.smooth:g}" if args.smooth > 0 else ""
    with open(os.path.join(OUT, f"{args.tag}twopcf_{args.case}_{args.pair}_nside{args.nside}{sfx}.json"), "w") as f:
        json.dump(out, f, indent=1)

    # figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = out["full_grid"]
    gam = np.array(d["gamma_arcmin"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    for key, lab, c in (("xi_plus", r"$|\xi_+^{(2,1)}|$ ($d^\ell_{22}$ of $C^{EE}+C^{BB}$)", "C0"),
                        ("xi_minus", r"$|\xi_-^{(2,1)}|$ ($d^\ell_{2,-2}$ of $C^{EE}-C^{BB}$)", "C3")):
        y = np.array(d[key])
        ax.loglog(gam[y > 0], y[y > 0], "-", color=c, label=lab)
        ax.loglog(gam[y < 0], -y[y < 0], "--", color=c)
    sm = gam < 10
    ax.loglog(gam[sm], np.abs(d["xi_minus"][0]) * (gam[sm] / gam[0]) ** 4, "k:", label=r"$\gamma^4$")
    ax.set_xlabel(r"$\gamma$ [arcmin]"); ax.set_ylabel(r"$|\xi|$ (solid +, dashed $-$)")
    ax.set_title(f"{args.case}, order pair ({args.pair[0]},{args.pair[1]}): real-space 2PCFs")
    ax.legend(fontsize=8)
    ax = axes[1]
    ELL = np.array(d["ELL"])
    ax.semilogx(ELL, np.array(d["EE_rec_over_EE_true"]), "C0o-", label=r"$T_{22}^{-1}[\xi_+]$ and $T_{2,-2}^{-1}[\xi_-]$ average / true $C^{EE}$")
    ax.semilogx(ELL, np.array(d["BB_rec_over_EE_rec"]), "C3s-", label=r"recovered $C^{BB}/C^{EE}$ = $(C^+ - C^-)/(C^+ + C^-)$")
    ax.axhspan(0.42, 0.95, color="C1", alpha=0.2, label="paper's claimed FK B/E")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel(r"$\ell$"); ax.set_title("inverse curved-sky transforms (paper's operator)")
    ax.legend(fontsize=8); ax.set_ylim(-0.3, 1.3)
    fig.tight_layout()
    path = os.path.join(OUT, f"{args.tag}twopcf_{args.case}_{args.pair}_nside{args.nside}{sfx}.png")
    fig.savefig(path, dpi=130)
    print("->", path)


if __name__ == "__main__":
    main()
