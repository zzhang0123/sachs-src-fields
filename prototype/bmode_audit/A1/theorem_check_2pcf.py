"""Pure-E consistency test of the paper's 2PCF sweeps and of the prototype MC.

For ANY spin-2 field cross-correlated with a pure-E field, xi_+ and xi_- are
the d^l_{2,2} / d^l_{2,-2} transforms of ONE spectrum, so (flat sky)

    xi_-(theta) = xi_+(theta) + int_0^theta d v  v xi_+(v) (4/theta^2 - 12 v^2/theta^4).

We test each sweep's (xi_+, xi_-) against this relation (needs xi_+ below the
smallest tabulated gamma: extrapolated as constant; the weight from v<gamma_min
is 2 (gamma_min/theta)^2, negligible for theta >~ 10 gamma_min).  We also
reproduce the paper's E/B ratios with the ported curved-sky transform.
"""
import sys, math
import numpy as np
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field")
from prototype.fk_mc import figure12 as F

ROOT = "/Users/zzhang/Documents/MyDrafts/STF_lensing/SFT-lensing-paper-analyses/sachs_sft/sftwick_outputs/2PCF/"
SWEEPS = {
    "O0":          (ROOT + "C_corr_op_O0/xi_C_corr_op_O0.npz", 0),
    "FF":          (ROOT + "C_corr_op_K_limber_FF/xi_C_corr_op_K_limber_FF.npz", 2),
    "FK_15360pf":  (ROOT + "C_corr_op_K_limber_FK_cut15360_permfix/xi_C_corr_op_K_limber_FK_cut15360_permfix.npz", 2),
    "FK_cut1000":  (ROOT + "C_corr_op_K_limber_FK/xi_C_corr_op_K_limber_FK.npz", 2),
}

def gamma_arcmin(xi, yi):
    xi = np.asarray(xi, float) / np.linalg.norm(xi); yi = np.asarray(yi, float) / np.linalg.norm(yi)
    return np.degrees(np.arccos(np.clip(np.dot(xi, yi), -1, 1))) * 60.0

def load(path, order):
    d = np.load(path, allow_pickle=True)
    a, b, o, v, x, y = d["a"], d["b"], d["order"], np.asarray(d["value"], float), d["x"], d["y"]
    out = {}; gref = None
    for pa, pb in sorted({(int(i), int(j)) for i, j in zip(a, b)}):
        m = (a == pa) & (b == pb) & (o == order)
        if not m.any(): continue
        idx = np.flatnonzero(m)
        g = np.array([gamma_arcmin(x[i], y[i]) for i in idx]); s = np.argsort(g)
        if gref is None: gref = g[s]
        out[(pa, pb)] = v[m][s]
    return gref, out

def pureE_xim_from_xip(g_arcmin, xip):
    """Predicted xi_- for a pure-E field from xi_+ (flat sky), const extrapolation below g[0]."""
    th = np.deg2rad(g_arcmin / 60.0)
    # fine grid from 0 to theta_max; xi_+ = xip[0] below th[0]
    tf = np.concatenate([np.linspace(0, th[0], 200, endpoint=False), np.exp(np.linspace(np.log(th[0]), np.log(th[-1]), 20000))])
    xf = np.interp(np.log(np.clip(tf, th[0], None)), np.log(th), xip)
    pred = np.empty_like(th)
    for i, t in enumerate(th):
        m = tf <= t
        v = tf[m]; f = xf[m] * v * (4.0 / t**2 - 12.0 * v**2 / t**4)
        pred[i] = np.interp(t, th, xip) + np.trapezoid(f, v)
    return pred

def report(name, g, xip, xim, xikk=None):
    pred = pureE_xim_from_xip(g, xip)
    print(f"\n=== {name} ===  (gamma range {g[0]:.2f}' .. {g[-1]:.0f}', {len(g)} pts)")
    print(f"{'gamma':>8} {'xi_kk':>11} {'xi_+':>11} {'xi_-':>11} {'xi_-^pureE':>12} {'xi_-/pred':>10} {'xi_-/xi_+':>10}")
    for i in range(len(g)):
        if g[i] < 3.0 and i % 2: continue
        if g[i] > 60 and i % 3: continue
        kk = xikk[i] if xikk is not None else float('nan')
        r = xim[i] / pred[i] if pred[i] else float('nan')
        print(f"{g[i]:>8.2f} {kk:>11.3e} {xip[i]:>11.3e} {xim[i]:>11.3e} {pred[i]:>12.3e} {r:>10.3f} {xim[i]/xip[i]:>10.3f}")
    # curved-sky E/B via the ported transform (paper's recipe)
    s22 = F.build_curved_matrix(g, F.ELL, 2, 2); s2m2 = F.build_curved_matrix(g, F.ELL, 2, -2)
    EpB = F.forward_curved(xip, s22); EmB = F.forward_curved(xim, s2m2)
    EE = 0.5 * (EpB + EmB); BB = 0.5 * (EpB - EmB)
    # also: BB predicted if xi_- were the pure-E partner of xi_+
    EmB_pred = F.forward_curved(pred, s2m2)
    print(f"   curved-sky (paper transform, DC-subtracted): BB/EE at ell=")
    for L in (60, 100, 300, 1000, 1500):
        j = int(np.argmin(np.abs(F.ELL - L)))
        print(f"      ell={int(F.ELL[j]):5d}: EE={EE[j]:+.3e} BB={BB[j]:+.3e} BB/EE={BB[j]/EE[j]:+.3f}   "
              f"[(EE-BB)_pureE/(EE-BB)_sweep={EmB_pred[j]/EmB[j]:+.3f}]")

for name, (path, order) in SWEEPS.items():
    g, grp = load(path, order)
    xip = grp[(1, 1)] + grp[(2, 2)]; xim = grp[(1, 1)] - grp[(2, 2)]
    report(name, g, xip, xim, grp.get((0, 0)))

# prototype MC channels
d = np.load("/Users/zzhang/projects/SFT/src-field/prototype/fk_mc/outputs/fk_channels.npz")
g = d["gamma"]
for ch in ("o0", "ff", "fk"):
    x = d[ch]
    report(f"prototype MC {ch}", g, x[:, 1, 1] + x[:, 2, 2], x[:, 1, 1] - x[:, 2, 2], x[:, 0, 0])
