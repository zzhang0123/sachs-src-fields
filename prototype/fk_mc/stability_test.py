"""Linchpin test: does sachsray's linear Jacobi stay finite on the demo2
non-Gaussian driving that blows up the paper's nonlinear Riccati?

Generates ONE batch of skewed (demo2-deformed, AR(1)) driving realisations at a
separation where the Riccati blows up, then evolves the SAME realisations with
(a) the paper's Riccati `_step` and (b) sachsray's Jacobi, and compares blow-up
rates. Run: python prototype/fk_mc/stability_test.py
"""
from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field")
from prototype.fk_mc import _setup  # noqa: E402
mc, ds, bgmod = _setup.mc, _setup.ds, _setup.bg

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import sachsray as sr  # noqa: E402


def gen_skew_field(cfg, cos_gamma, m, seed):
    """(N, 6, m) skewed AR(1) demo2 driving field f, plus (lam, phi00_bg, IC)."""
    bg_obj, lam, dlam, resp, V, Lf, Q, sig, _ = mc._field_precompute(cfg, cos_gamma)
    N = cfg.n_lambda
    rho = float(np.exp(-dlam / sig))
    s1mr2 = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
    rng = np.random.default_rng(seed)
    f = np.empty((N, 6, m))
    z = np.zeros((6, m))
    for k in range(N):
        innov = Lf[k] @ rng.standard_normal((6, m))
        z = innov if k == 0 else (rho * z + s1mr2 * innov)
        ff = np.einsum("im,jm->ijm", z, z) - V[k][:, :, None]
        f[k] = z + 0.5 * np.einsum("aij,ijm->am", Q[k], ff)
    Dv = np.asarray(bg_obj.D(lam), float)
    phi00_bg = np.asarray(bg_obj._D_spline(lam, 2), float) / Dv   # D''/D = Phi00_bg
    return lam, dlam, resp, f, phi00_bg, Dv, bg_obj


def riccati_blowup(cfg, lam, dlam, resp, f, m):
    """Evolve ray-0 with the paper's nonlinear Riccati; return blow-up fraction."""
    N = cfg.n_lambda
    s = np.zeros((3, m))
    blown = np.zeros(m, dtype=bool)
    for k in range(N):
        fk = f[k, 0:3, :]                          # ray-0 driving fluctuation
        s = resp[k] * s + mc._f_vertex(s) * dlam + fk * dlam
        blown |= np.any(np.abs(s) > cfg.blowup, axis=0)
    return float(blown.mean())


def jacobi_finite(lam, f, phi00_bg, Dv, bg_obj, m):
    """Evolve ray-0 with sachsray's Jacobi; return finite fraction + kappa."""
    ts = jnp.asarray(lam)
    # full driving Phi00 = background + fluctuation; Psi = fluctuation
    drive = np.empty((m, lam.size, 3))
    drive[:, :, 0] = phi00_bg[None, :] + f[:, 0, :].T
    drive[:, :, 1] = f[:, 1, :].T
    drive[:, :, 2] = f[:, 2, :].T
    drive = jnp.asarray(drive)
    # background Jacobi IC at lam[0]: J = D(lam0) I, Jdot = D'(lam0) I
    D0 = float(Dv[0])
    Dp0 = float(bg_obj._D_spline(lam[0], 1))
    y0 = jnp.array([D0, 0.0, 0.0, D0, Dp0, 0.0, 0.0, Dp0])
    lam_eval = jnp.asarray([lam[-1]])
    D_bg = D0  # scalar background distance at source via Jacobi scalar (use Dv[-1])
    D_bg = float(Dv[-1])

    def one(d):
        y = sr.solvers.solve_jacobi_matrix(ts, d, lam_eval, y0=y0,
                                           rtol=1e-6, atol=1e-8, max_steps=200_000)[0]
        J = y[:4].reshape(2, 2)
        A = J / D_bg
        return 1.0 - 0.5 * (A[0, 0] + A[1, 1])    # convergence kappa

    # python loop (robust to per-ray diffrax max_steps errors)
    kap = np.full(m, np.nan)
    one_j = jax.jit(one)
    n_err = 0
    for i in range(m):
        try:
            kap[i] = float(one_j(drive[i]))
        except Exception:
            n_err += 1
    finite = np.isfinite(kap)
    return float(finite.mean()), kap, n_err


def main():
    g = 100.0
    cosg = float(np.cos(np.deg2rad(g / 60.0)))
    m = 400
    cfg = mc.MCConfig(n_real=m, batch_size=m, n_lambda=300, sigma_lambda=8.0,
                      use_f_vertex=True, skew_scale=1.0)
    print(f"gamma={g}'  m={m}  n_lambda={cfg.n_lambda}  sigma_lambda={cfg.sigma_lambda}")
    lam, dlam, resp, f, phi00_bg, Dv, bg_obj = gen_skew_field(cfg, cosg, m, seed=7)
    print(f"  driving |f| stats: median={np.median(np.abs(f)):.2e} "
          f"max={np.max(np.abs(f)):.2e}  phi00_bg(mid)={phi00_bg[len(phi00_bg)//2]:.2e}")

    rb = riccati_blowup(cfg, lam, dlam, resp, f, m)
    print(f"  Riccati  (paper integrator):  blow-up fraction = {rb:.0%}")

    jf, kap, n_err = jacobi_finite(lam, f, phi00_bg, Dv, bg_obj, m)
    print(f"  Jacobi   (sachsray):          finite  fraction = {jf:.0%}  "
          f"(diffrax max_steps errors: {n_err})")
    fin = np.isfinite(kap)
    if fin.any():
        print(f"  sachsray kappa: mean={np.nanmean(kap):.3e} std={np.nanstd(kap):.3e} "
              f"range=[{np.nanmin(kap):.2e}, {np.nanmax(kap):.2e}]")
    print("\n  => " + ("Jacobi STABLE where Riccati blows up (linchpin OK)"
                       if jf > 0.98 and rb > 0.1 else "inconclusive — inspect"))


if __name__ == "__main__":
    main()
