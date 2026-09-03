"""E/B split of the paper's FK, FF and O0 2PCF sweeps via the figure12 port of the
analysis3 curved-sky transform.  Reports BB/EE for FK and the pure-E residual."""
import sys
import numpy as np
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field/prototype/fk_mc")
from figure12 import wigner_d, build_curved_matrix, forward_curved, combine, ELL

BASE = "/Users/zzhang/Documents/MyDrafts/STF_lensing/SFT-lensing-paper-analyses/sachs_sft/sftwick_outputs/2PCF/"
FILES = {"O0": "C_corr_op_O0/xi_C_corr_op_O0.npz",
         "FF": "C_corr_op_K_limber_FF/xi_C_corr_op_K_limber_FF.npz",
         "FK": "C_corr_op_K_limber_FK_cut15360_permfix/xi_C_corr_op_K_limber_FK_cut15360_permfix.npz",
         "FKjun": "C_corr_op_K_limber_FK/xi_C_corr_op_K_limber_FK.npz"}

def load(name):
    d = np.load(BASE + name, allow_pickle=True)
    x = np.array([np.asarray(v, float) for v in d["x"]]); y = np.array([np.asarray(v, float) for v in d["y"]])
    c = np.clip(np.einsum("ij,ij->i", x, y) / np.linalg.norm(x, axis=1) / np.linalg.norm(y, axis=1), -1, 1)
    g = np.degrees(np.arccos(c)) * 60
    a, b, v = d["a"].astype(int), d["b"].astype(int), np.asarray(d["value"], float)
    gam = np.unique(np.round(g, 6))
    out = np.zeros((gam.size, 3, 3))
    for gi, ai, bi, vi in zip(g, a, b, v):
        k = np.argmin(np.abs(gam - gi)); out[k, ai, bi] = vi; out[k, bi, ai] = vi
    return gam, out

ell = np.asarray(ELL, int) if not callable(ELL) else ELL()
res = {}
for tag, f in FILES.items():
    gam, x = load(f)
    xip = x[:, 1, 1] + x[:, 2, 2]; xim = x[:, 1, 1] - x[:, 2, 2]
    Mp = build_curved_matrix(gam, ell, 2, 2); Mm = build_curved_matrix(gam, ell, 2, -2)
    EpB = forward_curved(xip, Mp); EmB = forward_curved(xim, Mm)
    EE = 0.5 * (EpB + EmB); BB = 0.5 * (EpB - EmB)
    res[tag] = (gam, x, EE, BB)
    print(f"\n== {tag}: gamma {gam[0]:.2f}'..{gam[-1]:.0f}'  ({gam.size} pts)")
    for L in (60, 100, 200, 500, 1000, 1500):
        i = np.argmin(np.abs(ell - L))
        print(f"  ell={ell[i]:5d}  EE={EE[i]:+.3e}  BB={BB[i]:+.3e}  BB/EE={BB[i]/EE[i]:+.3f}")
g, x, EE0, BB0 = res["O0"]; _, _, EEk, BBk = res["FK"]
print("\nFK BB / O0 EE at ell 60,1500:", [f"{BBk[np.argmin(np.abs(ell-L))]/EE0[np.argmin(np.abs(ell-L))]:.2e}" for L in (60, 1500)])
print("FK xi_-/xi_+ at small gamma:", [(f"{gg:.2f}'", f"{(xx[1,1]-xx[2,2])/(xx[1,1]+xx[2,2]):+.3f}") for gg, xx in zip(g[:6], x[:6])])
print("O0 xi_-/xi_+ at small gamma:", [(f"{gg:.2f}'", f"{(xx[1,1]-xx[2,2])/(xx[1,1]+xx[2,2]):+.3f}") for gg, xx in zip(g[:6], res['O0'][1][:6])])
