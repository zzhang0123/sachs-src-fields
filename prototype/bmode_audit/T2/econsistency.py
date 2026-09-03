"""Pure-E consistency test on the paper's production 2PCF sweeps.

For any spin-2 field A cross-correlated with a pure-E field G (parity even):
  xi_+ = sum (2l+1)/4pi C_l d^l_{2,2},  xi_- = sum (2l+1)/4pi C_l d^l_{2,-2}
with ONE C_l.  Hence T22^{-1}[xi_+] == T2,-2^{-1}[xi_-] and BB == 0.
We apply the paper's own transform to O0 (control: must be pure E), FF, FK.
"""
import sys
import numpy as np
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field/prototype/fk_mc")
import figure12 as F

BASE = ("/Users/zzhang/Documents/MyDrafts/STF_lensing/SFT-lensing-paper-analyses/"
        "sachs_sft/sftwick_outputs/2PCF/")
FILES = {
    "O0": BASE + "C_corr_op_O0/xi_C_corr_op_O0.npz",
    "FF": BASE + "C_corr_op_K_limber_FF/xi_C_corr_op_K_limber_FF.npz",
    "FK15360": BASE + "C_corr_op_K_limber_FK_cut15360_permfix/xi_C_corr_op_K_limber_FK_cut15360_permfix.npz",
    "FK1000": BASE + "C_corr_op_K_limber_FK/xi_C_corr_op_K_limber_FK.npz",
}


def gamma_arcmin(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    c = np.clip(np.sum(x * y, axis=-1) / np.linalg.norm(x, axis=-1) / np.linalg.norm(y, axis=-1), -1, 1)
    return np.degrees(np.arccos(c)) * 60.0


def load(path, order):
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    a, b, o, v = d["a"], d["b"], d["order"], d["value"]
    x = np.stack([np.asarray(t, float) for t in d["x"]])
    y = np.stack([np.asarray(t, float) for t in d["y"]])
    g = gamma_arcmin(x, y)
    sel = (o == order)
    out = {}
    for (aa, bb) in [(1, 1), (2, 2), (1, 2), (0, 0)]:
        m = sel & (a == aa) & (b == bb)
        gg, vv = g[m], v[m]
        idx = np.argsort(gg)
        gg, vv = gg[idx], vv[idx]
        # merge duplicates in gamma by mean
        ug, inv = np.unique(np.round(gg, 6), return_inverse=True)
        mv = np.array([vv[inv == i].mean() for i in range(len(ug))])
        out[(aa, bb)] = (ug, mv)
    return keys, out


ELL = F.ELL
for name, path in FILES.items():
    order = 0 if name == "O0" else 2
    keys, xi = load(path, order)
    g = xi[(1, 1)][0]
    assert np.allclose(g, xi[(2, 2)][0])
    xip = xi[(1, 1)][1] + xi[(2, 2)][1]
    xim = xi[(1, 1)][1] - xi[(2, 2)][1]
    s22 = F.build_curved_matrix(g, ELL, 2, 2)
    s2m2 = F.build_curved_matrix(g, ELL, 2, -2)
    P = F.forward_curved(xip, s22)
    M = F.forward_curved(xim, s2m2)
    EE, BB = 0.5 * (P + M), 0.5 * (P - M)
    print(f"\n=== {name}: keys={keys}, n_gamma={len(g)}, gamma=[{g.min():.3g},{g.max():.3g}]'")
    print(" ell    xi+->EE+BB      xi- ->EE-BB      BB/EE")
    for l, p, m, ee, bb in zip(ELL, P, M, EE, BB):
        if l in (10, 20, 60, 100, 200, 500, 1000, 1500) or abs(l - 60) < 8 or abs(l-1500) < 100:
            print(f"{l:5.0f}  {p:+.4e}   {m:+.4e}   {bb/ee:+.3f}")
    # small-gamma slope of xi_- and xi_+
    lo = g < 2.0
    if lo.sum() > 2:
        sl_m = np.polyfit(np.log(g[lo]), np.log(np.abs(xim[lo]) + 1e-300), 1)[0]
        print(f" small-gamma slope: xi_- ~ gamma^{sl_m:.2f};  xi_-/xi_+ at gamma_min = {xim[0]/xip[0]:+.3e}")
    # E-consistency in real space (flat sky): xi_-^E(th) = xi_+(th) + int_0^th d(v) v/th^2 xi_+(v) (4 - 12 v^2/th^2)
    from scipy.interpolate import PchipInterpolator
    from scipy.integrate import quad
    f = PchipInterpolator(np.log(g), xip)
    def xip_of(v):
        return f(np.log(v)) if v >= g[0] else xip[0]
    pred = []
    for th in g:
        val, _ = quad(lambda v: v / th**2 * xip_of(v) * (4 - 12 * v**2 / th**2), 0, th, limit=200)
        pred.append(xip_of(th) + val)
    pred = np.array(pred)
    print(" gamma[']   xi_-(table)     xi_-(pure-E from xi_+)   ratio")
    for k in range(0, len(g), max(1, len(g)//10)):
        print(f" {g[k]:8.3f}  {xim[k]:+.4e}   {pred[k]:+.4e}   {xim[k]/pred[k]:+.3f}")
