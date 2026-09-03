"""Cross-spectrum form of the THEOREM test on the full-sky Sachs ray-tracer.

FK diagram == 2 Re <gamma^(>=2)(n1) gamma^(1)*(n2)> at O(zeta).  We build
gamma_lin = linear response (tiny-amplitude run, rescaled; contains the pure-E
response to the whole phi = g + f(g^2-<g^2>)) and gamma_full (all orders), and
measure the cross spectra
    X_EE(f) = C^EE(full, lin) - C^EE(lin, lin),   X_BB(f) = C^BB(full, lin) - C^BB(lin,lin)
    X_OO(f) = C(omega_full, omega_lin)
The f-odd part [X(+f)-X(-f)]/2 at O(A^3 f) is exactly the FK-type term with the
local-NG bispectrum.  Theorem: X_BB_odd == 0 identically (gamma_B^(1) == 0),
paper: X_BB_odd / X_EE_odd = 0.42..0.95.  Several seeds are averaged to beat
the realisation noise of X_EE_odd.
"""
from __future__ import annotations
import sys, time, os
import numpy as np
import healpy as hp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field")
from sachsray import fields, raytrace  # noqa: E402

NSIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 64
NLAM = int(sys.argv[2]) if len(sys.argv) > 2 else 24
A0 = float(sys.argv[3]) if len(sys.argv) > 3 else 2e-4
NSEED = int(sys.argv[4]) if len(sys.argv) > 4 else 4
LMAX = 2 * NSIDE
F0 = 0.6
EPS = 1e-4          # linear-response amplitude ratio
OUT = "/Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/T2/outputs"
os.makedirs(OUT, exist_ok=True)

ell = np.arange(LMAX + 1)
cl_g = np.where(ell >= 2, 1.0 / (ell + 1.0) ** 2, 0.0)
lap = -ell * (ell + 1.0)
eth2 = np.sqrt(np.clip((ell - 1) * ell * (ell + 1) * (ell + 2), 0, None))
ts = np.linspace(0.0, 1.0, NLAM + 1)


def make_g(seed):
    rng = np.random.default_rng(seed)
    out = []
    for k in range(NLAM + 1):
        alm = hp.synalm(cl_g, lmax=LMAX, new=True)
        out.append(hp.alm2map(alm, NSIDE, lmax=LMAX))
    return np.array(out)


def driving(g_maps, A, f):
    dphi, wre, wim = [], [], []
    for k in range(NLAM + 1):
        g = g_maps[k]
        phi = g + f * (g * g - np.mean(g * g))
        alm = hp.map2alm(phi, lmax=LMAX, iter=3)
        dphi.append(hp.alm2map(hp.almxfl(alm, -A * lap), NSIDE, lmax=LMAX))
        q, u = hp.alm2map_spin([hp.almxfl(alm, -A * eth2), np.zeros_like(alm)], NSIDE, 2, LMAX)
        wre.append(q); wim.append(u)
    return np.array(dphi), np.array(wre), np.array(wim)


def trace(g_maps, A, f):
    d, wr, wi = driving(g_maps, A, f)
    fld = fields.driving_from_components(ts, d, wr, wi, None, dtype=jnp.float64)
    out = raytrace.trace_rays(fld, 1.0, chunk=4096, rtol=1e-9, atol=1e-12)
    return {k: np.asarray(v) for k, v in out.items()}


def eb(out):
    E, B = hp.map2alm_spin([out["gamma1"], out["gamma2"]], 2, LMAX)
    return E, B, hp.map2alm(out["omega"], lmax=LMAX, iter=0), hp.map2alm(out["kappa"], lmax=LMAX, iter=0)


BANDS = [(2, 5), (6, 12), (13, 25), (26, 45), (46, 80), (81, LMAX - 4)]


def band(c, lo, hi):
    return float(np.mean(c[lo:hi + 1]))


acc = {}
for A in (A0, 2 * A0):
    for seed in range(NSEED):
        g = make_g(100 + seed)
        f = F0 / g.std()
        X = {}
        for ff in (+f, -f):
            t0 = time.time()
            full = eb(trace(g, A, ff))
            lin = eb(trace(g, A * EPS, ff))
            lin = tuple(a / EPS for a in lin)
            Ef, Bf, Of, Kf = full
            El, Bl, Ol, Kl = lin
            X[ff] = dict(
                EE=hp.alm2cl(Ef, El) - hp.alm2cl(El),
                BB=hp.alm2cl(Bf, Bl) - hp.alm2cl(Bl),
                OO=hp.alm2cl(Of, Ol),
                KK=hp.alm2cl(Kf, Kl) - hp.alm2cl(Kl),
                EE_lin=hp.alm2cl(El), BB_lin=hp.alm2cl(Bl),
                BB_full=hp.alm2cl(Bf), EE_full=hp.alm2cl(Ef), OO_full=hp.alm2cl(Of),
            )
            print(f"A={A:.1e} seed={seed} f={ff:+.2f}: {time.time()-t0:.0f}s", flush=True)
        odd = {k: 0.5 * (X[f][k] - X[-f][k]) for k in X[f]}
        even = {k: 0.5 * (X[f][k] + X[-f][k]) for k in X[f]}
        acc.setdefault(A, []).append((odd, even))

print("\n=== cross-spectrum FK-type term, seed-averaged band means ===")
print("A       band     X_EE_odd/EE_lin  (noise)   X_BB_odd/X_EE_odd  (noise)   X_OO_odd/X_EE_odd   BBlin/EElin   X_KK_odd/KK... X_EE_even/EE_lin")
for A, lst in acc.items():
    for lo, hi in BANDS:
        ee = np.array([band(o["EE"], lo, hi) for o, e in lst])
        bb = np.array([band(o["BB"], lo, hi) for o, e in lst])
        oo = np.array([band(o["OO"], lo, hi) for o, e in lst])
        el = np.array([band(e["EE_lin"], lo, hi) for o, e in lst])
        bl = np.array([band(e["BB_lin"], lo, hi) for o, e in lst])
        eev = np.array([band(e["EE"], lo, hi) for o, e in lst])
        n = len(lst)
        r_ee = ee.mean() / el.mean(); r_ee_err = ee.std(ddof=1) / np.sqrt(n) / el.mean() if n > 1 else np.nan
        r_bb = bb.mean() / ee.mean(); r_bb_err = bb.std(ddof=1) / np.sqrt(n) / abs(ee.mean()) if n > 1 else np.nan
        r_oo = oo.mean() / ee.mean()
        print(f"{A:.1e} [{lo:3d},{hi:3d}]  {r_ee:+.3e} ({r_ee_err:.1e})   {r_bb:+.3e} ({r_bb_err:.1e})   {r_oo:+.3e}     {bl.mean()/el.mean():.1e}    {eev.mean()/el.mean():+.3e}")

print("\n=== amplitude scaling of the seed-averaged odd cross terms (band [13,80]) ===")
As = list(acc.keys())
for key in ("EE", "BB", "OO", "KK"):
    vals = [np.mean([band(o[key], 13, 80) for o, e in acc[A]]) for A in As]
    sl = np.log(abs(vals[1] / vals[0])) / np.log(As[1] / As[0])
    print(f"X_{key}_odd: {['%+.3e' % v for v in vals]}  slope in A = {sl:.2f}  (theorem: EE 3, BB none at 3)")
np.savez(f"{OUT}/theorem_cross_nside{NSIDE}.npz",
         **{f"A{A:.1e}_s{i}_{par}_{k}": v for A, lst in acc.items() for i, (o, e) in enumerate(lst)
            for par, dd in (("odd", o), ("even", e)) for k, v in dd.items()})
