"""Numerical B-mode floor of sachsray vs dtype and solver tolerance, at a
realistic amplitude (std kappa ~ 0.01).  Pure-E spin-2 input (see
sachsray_eb_basis_test.py); any excess BB/EE over the converged float64
rtol=1e-9 run is solver/roundoff error leaking into B."""
import sys, time
import numpy as np, healpy as hp
import jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field")
import sachsray as sr
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/A1")
import warnings; warnings.filterwarnings("ignore")

nside, n_lam, lam_s = 32, 48, 2.0
lmax = 2 * nside; npix = hp.nside2npix(nside)
ell = np.arange(lmax + 1).astype(float); lam = np.linspace(0.0, lam_s, n_lam)
rng = np.random.default_rng(1)
A = 3e-5
clphi = np.zeros(lmax + 1); clphi[2:] = A / (ell[2:] * (ell[2:] + 1.0)) ** 2
c2 = np.sqrt(np.clip((ell + 2) * (ell + 1) * ell * (ell - 1), 0, None))
phi00 = np.zeros((n_lam, npix)); w1 = np.zeros((n_lam, npix)); w2 = np.zeros((n_lam, npix))
for i in range(n_lam):
    alm = hp.synalm(clphi, lmax=lmax, new=True)
    phi00[i] = hp.alm2map(hp.almxfl(alm, -ell * (ell + 1.0)), nside, lmax=lmax)
    Q, U = hp.alm2map_spin([hp.almxfl(alm, -c2), np.zeros_like(alm)], nside, 2, lmax)
    w1[i] = Q; w2[i] = U

def eb(g1, g2):
    aE, aB = hp.map2alm_spin([g1, g2], 2, lmax=lmax)
    cEE = hp.alm2cl(aE); cBB = hp.alm2cl(aB); band = slice(4, 50)
    return np.sum(cBB[band]) / np.sum(cEE[band])

ref = None
print(f"{'dtype':>8} {'rtol':>6} {'std(kappa)':>10} {'BB/EE':>10} {'excess over ref':>16} {'rms(dgamma1)/std':>17}")
for dtype, rtol in ((np.float64, 1e-9), (np.float64, 1e-7), (np.float64, 1e-6), (np.float64, 1e-5),
                    (np.float32, 1e-7), (np.float32, 1e-6), (np.float32, 1e-5)):
    field = sr.driving_from_components(lam, phi00, w1, w2, phi00_bg=None, dtype=dtype)
    out = sr.trace_rays(field, lam_s, chunk=4096, rtol=rtol, atol=rtol * 1e-2)
    k, g1, g2 = (np.asarray(out[x], float) for x in ("kappa", "gamma1", "gamma2"))
    r = eb(g1, g2)
    if ref is None:
        ref = (r, g1.copy())
    print(f"{np.dtype(dtype).name:>8} {rtol:>6.0e} {np.std(k):>10.3e} {r:>10.2e} {r - ref[0]:>+16.2e} {np.sqrt(np.mean((g1-ref[1])**2))/np.std(ref[1]):>17.1e}")
