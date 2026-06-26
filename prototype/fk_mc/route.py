"""Route the paper's demo2 driving through sachsray's Jacobi, in the paper's
observable convention kappa = int (theta - theta_sa) dlambda.

The paper integrates the fluctuation Riccati ds=-2 theta_sa s + F ss + f and reads
kappa = int s dlambda.  sachsray integrates the LINEAR Jacobi J''=T(lam)J (stable
on heavy-tailed driving) and we extract the SAME observable by reading the optical
expansion theta(lam)=1/2 tr(Jdot J^-1) along the ray and integrating its
fluctuation theta - theta_sa.  Gaussian (skew=0) must reproduce the paper's
Order-0 / simulate() <kappa kappa>; that is the normalization gate.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "/Users/zzhang/projects/SFT/src-field")
from prototype.fk_mc import _setup  # noqa: E402
mc = _setup.mc

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
from sachsray import solvers as _sol  # noqa: E402


def gen_field(cfg: "mc.MCConfig", cos_gamma: float, m: int, seed: int):
    """Generate (N,6,m) AR(1) demo2 driving f (skew per cfg.skew_scale) + grids."""
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
        f[k] = z + 0.5 * np.einsum("aij,ijm->am", Q[k], ff)  # Q=0 if Gaussian
    Dv = np.asarray(bg_obj.D(lam), float)
    phi00_bg = np.asarray(bg_obj._D_spline(lam, 2), float) / Dv  # D''/D
    theta_sa = np.asarray(bg_obj.theta_sa(lam), float)           # D'/D
    return dict(lam=lam, dlam=dlam, resp=resp, f=f, phi00_bg=phi00_bg,
                theta_sa=theta_sa, Dv=Dv, bg=bg_obj)


# ---- sachsray Jacobi observable: kappa = int (theta - theta_sa) dlambda -------
def kappa_sachsray(g: dict, ray: int, *, rtol=1e-6, atol=1e-8, max_steps=200_000):
    """Per-realisation kappa for a ray, via the stable Jacobi + theta extraction."""
    lam = g["lam"]
    ts = jnp.asarray(lam)
    theta_sa = jnp.asarray(g["theta_sa"])
    base = 3 * ray
    f = g["f"]
    m = f.shape[2]
    drive = np.empty((m, lam.size, 3))
    drive[:, :, 0] = g["phi00_bg"][None, :] + f[:, base + 0, :].T
    drive[:, :, 1] = f[:, base + 1, :].T
    drive[:, :, 2] = f[:, base + 2, :].T
    drive = jnp.asarray(drive)
    D0 = float(g["Dv"][0])
    Dp0 = float(g["bg"]._D_spline(lam[0], 1))
    y0 = jnp.array([D0, 0.0, 0.0, D0, Dp0, 0.0, 0.0, Dp0])

    def one(d):
        ys = _sol.solve_jacobi_matrix(ts, d, ts, y0=y0, rtol=rtol, atol=atol,
                                      max_steps=max_steps)        # (N,8)
        J = ys[:, :4].reshape(-1, 2, 2)
        Jd = ys[:, 4:].reshape(-1, 2, 2)
        S = jnp.einsum("nij,njk->nik", Jd, jnp.linalg.inv(J))
        theta = 0.5 * (S[:, 0, 0] + S[:, 1, 1])
        return jnp.trapezoid(theta - theta_sa, ts)              # kappa = int dtheta

    return np.asarray(jax.jit(jax.vmap(one))(drive))             # (m,)


def kappa_riccati(g: dict, cfg: "mc.MCConfig", ray: int):
    """Per-realisation kappa for a ray via the paper's Riccati (cross-check)."""
    lam, dlam, resp, f = g["lam"], g["dlam"], g["resp"], g["f"]
    N = cfg.n_lambda
    base = 3 * ray
    m = f.shape[2]
    s = np.zeros((3, m))
    kap = np.zeros(m)
    blown = np.zeros(m, dtype=bool)
    for k in range(N):
        fk = f[k, base:base + 3, :]
        s = resp[k] * s + mc._f_vertex(s) * dlam + fk * dlam
        w = 0.5 * dlam if (k == 0 or k == N - 1) else dlam
        kap += w * s[0]
        blown |= np.any(np.abs(s) > cfg.blowup, axis=0)
    kap[blown] = np.nan
    return kap


def _bridge_test():
    import warnings
    warnings.filterwarnings("ignore")
    print("Gaussian normalization gate: sachsray kappa=int dtheta vs paper Riccati & O0")
    print(f"{'gamma':>7} {'O0_analytic':>12} {'paper xi00':>12} {'sachsray xi00':>14} "
          f"{'sr/paper':>9} {'sr/O0':>8}")
    m = 6000
    cfg = mc.MCConfig(n_real=m, batch_size=m, n_lambda=400, sigma_lambda=8.0,
                      use_f_vertex=True, skew_scale=0.0)
    for gar in (10.0, 100.0, 1000.0):
        cosg = float(np.cos(np.deg2rad(gar / 60.0)))
        g = gen_field(cfg, cosg, m, seed=11)
        k0_sr = kappa_sachsray(g, 0); k1_sr = kappa_sachsray(g, 1)
        xi_sr = float(np.nanmean(k0_sr * k1_sr))
        k0_r = kappa_riccati(g, cfg, 0); k1_r = kappa_riccati(g, cfg, 1)
        good = np.isfinite(k0_r) & np.isfinite(k1_r)
        xi_r = float(np.mean(k0_r[good] * k1_r[good]))
        o0 = float(mc.simulate(cfg, gar).o0_analytic)
        print(f"{gar:>7.1f} {o0:>12.4e} {xi_r:>12.4e} {xi_sr:>14.4e} "
              f"{xi_sr/xi_r if xi_r else float('nan'):>9.3f} {xi_sr/o0 if o0 else float('nan'):>8.3f}")


if __name__ == "__main__":
    _bridge_test()
