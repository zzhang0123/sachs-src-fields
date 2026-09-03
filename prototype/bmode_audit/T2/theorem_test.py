"""Direct test of the THEOREM with the full-sky Sachs ray-tracer (all orders in
the driving field, unperturbed path, exactly the paper's Sachs system).

Driving: per shell k an independent Gaussian scalar g_k on the sphere; local
non-Gaussianity phi_k = g_k + f (g_k^2 - <g_k^2>) gives a bispectrum linear in f.
Phi00 = -amp * nabla^2 phi (spin 0), Psi0 = -amp * eth^2 phi (spin 2, pure E by
construction).  The tracer integrates J'' = T J per ray.

THEOREM: the part of C_BB odd in f (the K3-linear part, which contains the
paper's FK diagram) vanishes at the order A^4 f at which the E-mode FK lives;
the first f-odd B-mode is O(A^6 f).  The paper's claim is
[C_BB]_odd / [C_EE]_odd = 0.42 .. 0.95 at O(A^4 f).
We measure both, with the SAME Gaussian seed for +f, -f and 0, so the odd part
is an exact difference, and scale the amplitude A to read off the exponents.
"""
from __future__ import annotations
import sys, time
import numpy as np
import healpy as hp
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field")
from sachsray import fields, raytrace  # noqa: E402

NSIDE = int(sys.argv[1]) if len(sys.argv) > 1 else 32
NLAM = int(sys.argv[2]) if len(sys.argv) > 2 else 16
LMAX = 2 * NSIDE
SEED = 7
F0 = 0.6           # local-NG coefficient in units of 1/sigma_g  (f = F0/sigma_g)
A0 = float(sys.argv[3]) if len(sys.argv) > 3 else 6.0   # base amplitude
OUT = "/Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/T2/outputs"
import os; os.makedirs(OUT, exist_ok=True)

ell = np.arange(LMAX + 1)
cl_g = np.where(ell >= 2, 1.0 / (ell + 1.0) ** 2, 0.0)
lap = -ell * (ell + 1.0)                                   # nabla^2 eigenvalue
eth2 = np.sqrt(np.clip((ell - 1) * ell * (ell + 1) * (ell + 2), 0, None))  # eth^2

rng = np.random.default_rng(SEED)
ts = np.linspace(0.0, 1.0, NLAM + 1)          # vacuum background, D_bg = lambda
g_maps = []
for k in range(NLAM + 1):
    alm = hp.synalm(cl_g, lmax=LMAX, new=True)
    g_maps.append(hp.alm2map(alm, NSIDE, lmax=LMAX, verbose=False) if 'verbose' in hp.alm2map.__code__.co_varnames else hp.alm2map(alm, NSIDE, lmax=LMAX))
g_maps = np.array(g_maps)                      # (NLAM+1, npix)
sig_g = g_maps.std()
print(f"nside={NSIDE} npix={12*NSIDE**2} nlam={NLAM+1} sigma_g={sig_g:.4f}")


def driving(A, f):
    """Return (dphi00, wre, wim) each (n_lam, npix) from phi = g + f(g^2-<g^2>)."""
    dphi, wre, wim = [], [], []
    for k in range(NLAM + 1):
        g = g_maps[k]
        phi = g + f * (g * g - np.mean(g * g))
        alm = hp.map2alm(phi, lmax=LMAX, iter=3)
        alm_lap = hp.almxfl(alm, -A * lap)
        alm_eth = hp.almxfl(alm, -A * eth2)
        dphi.append(hp.alm2map(alm_lap, NSIDE, lmax=LMAX))
        q, u = hp.alm2map_spin([alm_eth, np.zeros_like(alm_eth)], NSIDE, 2, LMAX)
        wre.append(q); wim.append(u)
    return np.array(dphi), np.array(wre), np.array(wim)


def trace(A, f):
    d, wr, wi = driving(A, f)
    fld = fields.driving_from_components(ts, d, wr, wi, None, dtype=jnp.float64)
    t0 = time.time()
    out = raytrace.trace_rays(fld, 1.0, chunk=4096, rtol=1e-8, atol=1e-11)
    out = {k: np.asarray(v) for k, v in out.items()}
    print(f"  traced A={A:.3g} f={f:+.3g}: {time.time()-t0:.1f}s  "
          f"kappa_rms={out['kappa'].std():.3e} gamma_rms={np.hypot(out['gamma1'],out['gamma2']).std():.3e} "
          f"omega_rms={out['omega'].std():.3e}")
    return out


def spectra(out):
    E, B = hp.map2alm_spin([out["gamma1"], out["gamma2"]], 2, LMAX)
    cee, cbb = hp.alm2cl(E), hp.alm2cl(B)
    ceb = hp.alm2cl(E, B)
    ck = hp.anafast(out["kappa"], lmax=LMAX)
    com = hp.anafast(out["omega"], lmax=LMAX)
    return dict(EE=cee, BB=cbb, EB=ceb, KK=ck, OO=com)


def band(c, lo, hi):
    return float(np.mean(c[lo:hi + 1]))


BANDS = [(2, 5), (6, 12), (13, 25), (26, 45), (46, LMAX - 4)]
results = {}
for A in (A0, 2 * A0, 4 * A0):
    f = F0 / sig_g
    res = {}
    for ff in (0.0, +f, -f):
        res[ff] = spectra(trace(A, ff))
    # linear-response leakage floor: run at tiny amplitude, rescale
    lin = spectra(trace(A * 1e-3, 0.0))
    results[A] = (res, lin, f)

print("\n=== per-band results (band means of C_l) ===")
print("A       band      EE0        BB0/EE0   BB_lin/EE_lin  EE_odd/EE0   BB_odd/EE_odd   OO_odd/EE_odd  BB_odd/BB0")
for A, (res, lin, f) in results.items():
    ee0, bb0 = res[0.0]["EE"], res[0.0]["BB"]
    eeo = 0.5 * (res[f]["EE"] - res[-f]["EE"])
    bbo = 0.5 * (res[f]["BB"] - res[-f]["BB"])
    ooo = 0.5 * (res[f]["OO"] - res[-f]["OO"])
    for lo, hi in BANDS:
        print(f"{A:5.1f}  [{lo:3d},{hi:3d}]  {band(ee0,lo,hi):.3e}  {band(bb0,lo,hi)/band(ee0,lo,hi):.2e}  "
              f"{band(lin['BB'],lo,hi)/band(lin['EE'],lo,hi):.2e}      {band(eeo,lo,hi)/band(ee0,lo,hi):+.3e}   "
              f"{band(bbo,lo,hi)/band(eeo,lo,hi):+.3e}      {band(ooo,lo,hi)/band(eeo,lo,hi):+.3e}    {band(bbo,lo,hi)/band(bb0,lo,hi):+.3e}")

print("\n=== amplitude scaling (band [6,45]) ===")
As = list(results.keys())
for q in ("EE0", "BB0", "EE_odd", "BB_odd", "OO0", "OO_odd"):
    vals = []
    for A, (res, lin, f) in results.items():
        if q == "EE0": v = band(res[0.0]["EE"], 6, 45)
        elif q == "BB0": v = band(res[0.0]["BB"], 6, 45)
        elif q == "OO0": v = band(res[0.0]["OO"], 6, 45)
        elif q == "EE_odd": v = band(0.5 * (res[f]["EE"] - res[-f]["EE"]), 6, 45)
        elif q == "BB_odd": v = band(0.5 * (res[f]["BB"] - res[-f]["BB"]), 6, 45)
        else: v = band(0.5 * (res[f]["OO"] - res[-f]["OO"]), 6, 45)
        vals.append(v)
    slopes = [np.log(abs(vals[i + 1] / vals[i])) / np.log(As[i + 1] / As[i]) for i in range(len(As) - 1)]
    print(f"{q:7s} values={['%+.3e' % v for v in vals]}  log-slopes in A: {['%.2f' % s for s in slopes]}")

np.savez(f"{OUT}/theorem_test_nside{NSIDE}.npz",
         **{f"A{A:.3g}_f{ff:+.3g}_{k}": v for A, (res, lin, f) in results.items()
            for ff, sp in res.items() for k, v in sp.items()})
