"""Pin sachsray's screen-basis convention and E/B behaviour with a spin-2 input.

Build per-shell driving fields from ONE scalar potential phi_lm:
    Phi00 = alm2map( -l(l+1) phi_lm )                      (screen Laplacian)
    (W1, W2) = alm2map_spin([ s*sqrt((l+2)!/(l-2)!) phi_lm, 0 ], spin 2)   (eth^2 phi, pure E)
in HEALPix's (e_theta, e_phi) basis, trace with sachsray, and decompose the
output shear (gamma1, gamma2) with map2alm_spin.  A pure-E linear response must
give C_BB << C_EE (residual = second order, scaling with amplitude^2) and
C_kE / C_kk = +-1.  Controls: flip the sign of W2 (U) -> should turn the output
into (mostly) B; flip s.  Also the Born (linear) reference from the same maps.
"""
import sys, time
import numpy as np, healpy as hp
import jax
jax.config.update("jax_enable_x64", True)
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field")
import sachsray as sr

nside, n_lam, lam_s = 32, 48, 2.0
lmax = 2 * nside                       # keep well inside the pixel band limit
npix = hp.nside2npix(nside)
ell = np.arange(lmax + 1).astype(float)
lam = np.linspace(0.0, lam_s, n_lam)
kern = (lam_s - lam) * lam / lam_s     # vacuum lensing efficiency

def build(A, s=+1.0, flipU=False, seed=0):
    rng = np.random.default_rng(seed)
    clphi = np.zeros(lmax + 1); clphi[2:] = A / (ell[2:] * (ell[2:] + 1.0)) ** 2
    phi00 = np.zeros((n_lam, npix)); w1 = np.zeros((n_lam, npix)); w2 = np.zeros((n_lam, npix))
    c2 = np.sqrt(np.clip((ell + 2) * (ell + 1) * ell * (ell - 1), 0, None))
    for i in range(n_lam):
        alm = hp.synalm(clphi, lmax=lmax, new=True)
        phi00[i] = hp.alm2map(hp.almxfl(alm, -ell * (ell + 1.0)), nside, lmax=lmax, verbose=False)
        Q, U = hp.alm2map_spin([hp.almxfl(alm, s * c2), np.zeros_like(alm)], nside, 2, lmax)
        w1[i] = Q; w2[i] = (-U if flipU else U)
    return phi00, w1, w2

def eb(kappa, g1, g2):
    ak = hp.map2alm(kappa, lmax=lmax, iter=3)
    aE, aB = hp.map2alm_spin([g1, g2], 2, lmax=lmax)
    ckk = hp.alm2cl(ak); cEE = hp.alm2cl(aE); cBB = hp.alm2cl(aB); ckE = hp.alm2cl(ak, aE)
    band = slice(4, 50)
    return (np.sum(cBB[band]) / np.sum(cEE[band]), np.sum(ckE[band]) / np.sqrt(np.sum(ckk[band]) * np.sum(cEE[band])),
            np.sum(cEE[band]) / np.sum(ckk[band]))

print(f"nside={nside} lmax={lmax} n_lam={n_lam} npix={npix}")
print(f"{'case':>28} {'std(kappa)':>10} {'BB/EE(ray)':>11} {'BB/EE(Born)':>12} {'corr(k,E)':>10} {'EE/kk':>7}")
for A, s, flipU, label in ((3e-3, +1, False, "A=3e-3, s=+1"),
                           (3e-4, +1, False, "A=3e-4, s=+1 (10x weaker)"),
                           (3e-3, +1, True,  "A=3e-3, U -> -U (control)"),
                           (3e-3, -1, False, "A=3e-3, s=-1 (control)")):
    phi00, w1, w2 = build(A, s, flipU)
    field = sr.driving_from_components(lam, phi00, w1, w2, phi00_bg=None, dtype=np.float64)
    t0 = time.time()
    out = sr.trace_rays(field, lam_s, chunk=4096, rtol=1e-9, atol=1e-11)
    k, g1, g2 = (np.asarray(out[x]) for x in ("kappa", "gamma1", "gamma2"))
    # Born (linear) reference from the same maps: kappa = -int K Phi00, gamma_i = -int K W_i
    kb = -np.trapezoid(kern[:, None] * phi00, lam, axis=0)
    g1b = -np.trapezoid(kern[:, None] * w1, lam, axis=0); g2b = -np.trapezoid(kern[:, None] * w2, lam, axis=0)
    r_ray = eb(k, g1, g2); r_born = eb(kb, g1b, g2b)
    print(f"{label:>28} {np.std(k):>10.2e} {r_ray[0]:>11.2e} {r_born[0]:>12.2e} {r_ray[1]:>+10.4f} {r_ray[2]:>7.3f}   "
          f"[ray-Born: max|dk|/std={np.max(np.abs(k-kb))/np.std(kb):.2e}, max|dg1|/std={np.max(np.abs(g1-g1b))/np.std(g1b):.2e}]  {time.time()-t0:.0f}s")
