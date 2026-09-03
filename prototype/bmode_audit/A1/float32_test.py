"""float32 vs float64 in sachsray at cosmological units (lambda in Mpc, D ~ 2e3).

A B-mode of C_BB/C_EE ~ 1e-3 .. 1e-4 is a shear component ~3% .. 1% of the E
amplitude.  Measure the float32 error on (kappa, gamma1, gamma2, omega) against
float64, per ray, in units of std(gamma) and against a 1e-2 x std(gamma) target.
Vacuum background (D = lambda) with lambda_s = 2313 Mpc, driving fluctuation
amplitude ~1e-7 Mpc^-2 on ~8 Mpc cells (paper: Phi00, |Psi0| ~ 1e-8 Mpc^-2 level).
"""
import sys, time
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field")
import sachsray as sr

lam_s = 2313.0
npix = 512

def run(n_lam, amp, cell, rtol, atol, dtype, seed=0):
    rng = np.random.default_rng(seed)
    lam = np.linspace(0.0, lam_s, n_lam)
    # smooth-in-lambda random fields: sum of ~ (lam_s/cell) sinusoids
    ks = 2 * np.pi * np.arange(1, int(lam_s / cell)) / lam_s
    ph = rng.uniform(0, 2 * np.pi, (3, npix, ks.size))
    a = amp / np.sqrt(ks.size)
    maps = [(a * np.sin(ks[None, None, :] * lam[None, :, None] + ph[c][:, None, :])).sum(-1).T for c in range(3)]
    field = sr.driving_from_components(lam, maps[0], maps[1], maps[2], phi00_bg=None, dtype=dtype)
    out = sr.trace_rays(field, lam_s, chunk=512, rtol=rtol, atol=atol)
    return {k: np.asarray(out[k], float) for k in out}

def compare(n_lam, amp, cell, rtol, atol):
    r64 = run(n_lam, amp, cell, rtol, atol, np.float64)
    r32 = run(n_lam, amp, cell, rtol, atol, np.float32)
    r64t = run(n_lam, amp, cell, 1e-10, 1e-12, np.float64)  # tight reference
    sg = np.std(r64t["gamma1"])
    print(f"n_lam={n_lam} amp={amp:.0e} rtol={rtol:.0e}: std(kappa)={np.std(r64t['kappa']):.2e} std(gamma1)={sg:.2e} std(omega)={np.std(r64t['omega']):.2e}")
    for k in ("kappa", "gamma1", "gamma2", "omega"):
        e32 = r32[k] - r64t[k]; e64 = r64[k] - r64t[k]
        print(f"   {k:7s}: rms err float32 = {np.sqrt(np.mean(e32**2)):.2e} ({np.sqrt(np.mean(e32**2))/sg:.1e} of std(gamma)) "
              f"| float64@same tol = {np.sqrt(np.mean(e64**2)):.2e} ({np.sqrt(np.mean(e64**2))/sg:.1e})  "
              f"| B-target 1e-2*std(gamma) = {1e-2*sg:.1e}")

t0 = time.time()
compare(300, 1e-7, 8.0, 1e-6, 1e-8)
compare(300, 1e-7, 8.0, 1e-5, 1e-7)
compare(300, 1e-7, 8.0, 1e-7, 1e-9)
print(f"{time.time()-t0:.0f}s")
