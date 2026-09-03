"""Agent T1 audit: is the paper's FK B-mode compatible with a pure-E linear leg?

Three numerical probes on the paper's production 2PCF sweeps (read-only inputs):

  1. Reproduce the harmonic-space E/B split of O0, FF, FK with the paper's
     curved-sky Wigner-d transform (verbatim port in prototype/fk_mc/figure12.py).
  2. Cauchy-Schwarz bound  |C_BB^FK| <= 2 sqrt(C_BB^O0 * C_BB^FF)  against the
     deployed C_BB^FK  (C_BB^O0 is the finite-range transform artefact, C_BB^FF
     stands in for <|gamma_B^(2)|^2>).
  3. Flat-sky pure-E consistency (Schneider, van Waerbeke & Mellier 2002, eq. 27):
       xi_-(theta) = xi_+(theta) + int_0^theta dv (v/theta^2) xi_+(v) [4 - 12 v^2/theta^2]
     which holds iff C_BB = 0.  Needs xi_+ only on [0, theta], i.e. only the
     converged (gamma < 1 deg) part of FK.  Validated on O0 (pure E by construction).

Outputs: outputs/eb_consistency.npz and printed tables.
"""
from __future__ import annotations
import sys, math
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import jv

sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field/prototype/fk_mc")
import figure12 as F  # verbatim port of the paper's curved-sky transform

BASE = ("/Users/zzhang/Documents/MyDrafts/STF_lensing/SFT-lensing-paper-analyses/"
        "sachs_sft/sftwick_outputs/2PCF/")
NPZ = {
    "O0": BASE + "C_corr_op_O0/xi_C_corr_op_O0.npz",
    "FF": BASE + "C_corr_op_K_limber_FF/xi_C_corr_op_K_limber_FF.npz",
    "FK": BASE + "C_corr_op_K_limber_FK_cut15360_permfix/xi_C_corr_op_K_limber_FK_cut15360_permfix.npz",
    "FKjune": BASE + "C_corr_op_K_limber_FK/xi_C_corr_op_K_limber_FK.npz",
}
ORDER = {"O0": 0, "FF": 2, "FK": 2, "FKjune": 2}
OUT = "/Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/T1/outputs"


def gamma_arcmin(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    c = np.dot(x / np.linalg.norm(x), y / np.linalg.norm(y))
    return math.degrees(math.acos(min(1.0, max(-1.0, c)))) * 60.0


def load(path, order):
    """Return gamma[arcmin] (sorted) and dict (a,b) -> xi(gamma) at the given order."""
    d = np.load(path, allow_pickle=True)
    a, b, o, v = d["a"], d["b"], d["order"], np.asarray(d["value"], float)
    x, y = d["x"], d["y"]
    out, g_ref = {}, None
    for pa, pb in sorted({(int(i), int(j)) for i, j in zip(a, b)}):
        m = (a == pa) & (b == pb) & (o == order)
        idx = np.flatnonzero(m)
        g = np.array([gamma_arcmin(x[i], y[i]) for i in idx])
        s = np.argsort(g)
        if g_ref is None:
            g_ref = g[s]
        assert np.allclose(g[s], g_ref)
        out[(pa, pb)] = v[idx][s]
    return g_ref, out


def schneider_minus_from_plus(theta, xip, theta_eval, extrap="const"):
    """Flat-sky pure-E prediction of xi_- from xi_+ on [0, theta_eval].

    xi_+ is interpolated (PCHIP in log theta) on the data range and extrapolated
    below theta[0] either as a constant (xi_+(theta_0)) or linearly in log theta.
    """
    lt = np.log(theta)
    interp = PchipInterpolator(lt, xip)
    slope0 = (xip[1] - xip[0]) / (lt[1] - lt[0])

    def xi_plus(t):
        t = np.asarray(t, float)
        out = np.empty_like(t)
        inside = t >= theta[0]
        out[inside] = interp(np.log(t[inside]))
        if extrap == "const":
            out[~inside] = xip[0]
        else:
            out[~inside] = xip[0] + slope0 * (np.log(np.clip(t[~inside], 1e-12, None)) - lt[0])
        return out

    pred = []
    for th in theta_eval:
        v = np.linspace(0.0, th, 20001)
        integrand = (v / th**2) * xi_plus(v) * (4.0 - 12.0 * v**2 / th**2)
        pred.append(interp(math.log(th)) + np.trapezoid(integrand, v))
    return np.array(pred)


def selftest_schneider():
    """Delta-function spectrum: xi_+ = J0(l0 theta), xi_- must equal J4(l0 theta)."""
    l0 = 800.0
    theta = np.geomspace(1e-5, 0.05, 6000)
    xip = jv(0, l0 * theta)
    te = np.array([0.002, 0.005, 0.01, 0.02, 0.04])
    pred = schneider_minus_from_plus(theta, xip, te, extrap="const")
    truth = jv(4, l0 * te)
    err = np.max(np.abs(pred - truth) / np.max(np.abs(truth)))
    assert err < 2e-3, err
    print(f"[Schneider self-test PASS: J0 -> J4 to {err:.1e}]")


def main():
    import os
    os.makedirs(OUT, exist_ok=True)
    selftest_schneider()
    F.selftest_wigner()

    ELL = F.ELL
    data = {k: load(NPZ[k], ORDER[k]) for k in NPZ}
    g = data["O0"][0]
    for k in data:
        assert np.allclose(data[k][0], g)
    theta = np.radians(g / 60.0)

    s22 = F.build_curved_matrix(g, ELL, 2, 2)
    s2m2 = F.build_curved_matrix(g, ELL, 2, -2)

    res = {}
    print("\n=== 1. harmonic E/B split with the paper's curved-sky transform (z_s=5) ===")
    for k in ("O0", "FF", "FK", "FKjune"):
        xi = data[k][1]
        xip = xi[(1, 1)] + xi[(2, 2)]
        xim = xi[(1, 1)] - xi[(2, 2)]
        pbb = F.forward_curved(xip, s22)      # EE+BB
        mbb = F.forward_curved(xim, s2m2)     # EE-BB
        ee, bb = 0.5 * (pbb + mbb), 0.5 * (pbb - mbb)
        res[k] = dict(xip=xip, xim=xim, ee=ee, bb=bb)
        print(f"  {k:6s} BB/EE:", "  ".join(f"l={int(l)}:{b/e:+.3f}" for l, e, b in zip(ELL, ee, bb)
                                            if int(l) in (3, 10, 60, 150, 300, 800, 1500)))

    print("\n=== 2. Cauchy-Schwarz bound  |C_BB^FK| <= 2 sqrt(C_BB^O0 C_BB^FF)  (all / C_EE^O0) ===")
    ee0 = res["O0"]["ee"]
    print("   ell   f_BB^FK=C_BB^FK/C_EE^O0   |C_BB^O0|/C_EE^O0   C_BB^FF/C_EE^O0   bound/C_EE^O0   claim/bound")
    for i, l in enumerate(ELL):
        if int(l) < 50:
            continue
        fbb = res["FK"]["bb"][i] / ee0[i]
        r0 = abs(res["O0"]["bb"][i]) / ee0[i]
        rff = abs(res["FF"]["bb"][i]) / ee0[i]
        bound = 2.0 * math.sqrt(r0 * rff)
        print(f"  {int(l):5d}   {fbb:+.2e}                 {r0:.2e}            {rff:.2e}         {bound:.2e}      {abs(fbb)/bound:8.1f}")

    print("\n=== 3. flat-sky pure-E consistency: xi_- predicted from xi_+ (Schneider 2002 eq. 27) ===")
    te_arcmin = np.array([2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 60.0])
    te = np.radians(te_arcmin / 60.0)
    table = {}
    for k in ("O0", "FF", "FK", "FKjune"):
        xip, xim = res[k]["xip"], res[k]["xim"]
        pred_c = schneider_minus_from_plus(theta, xip, te, "const")
        pred_l = schneider_minus_from_plus(theta, xip, te, "loglin")
        actual = PchipInterpolator(np.log(theta), xim)(np.log(te))
        plus_at = PchipInterpolator(np.log(theta), xip)(np.log(te))
        table[k] = (pred_c, pred_l, actual, plus_at)
        print(f"  --- {k} ---")
        print("   theta[']   xi_+(actual)   xi_-(actual)   xi_-(pred,const)  xi_-(pred,loglin)   actual/pred   |xi_-/xi_+| act   |xi_-/xi_+| pred")
        for j, t in enumerate(te_arcmin):
            print(f"   {t:6.1f}   {plus_at[j]:+.3e}    {actual[j]:+.3e}    {pred_c[j]:+.3e}      {pred_l[j]:+.3e}       {actual[j]/pred_c[j]:+8.3f}      {abs(actual[j]/plus_at[j]):.2e}        {abs(pred_c[j]/plus_at[j]):.2e}")

    np.savez(OUT + "/eb_consistency.npz", ELL=ELL, gamma_arcmin=g,
             **{f"{k}_{q}": v for k in res for q, v in res[k].items()},
             te_arcmin=te_arcmin,
             **{f"schneider_{k}_{nm}": arr for k in table for nm, arr in
                zip(("pred_const", "pred_loglin", "actual", "plus"), table[k])})
    print(f"\n-> {OUT}/eb_consistency.npz")


if __name__ == "__main__":
    main()
