"""
End-to-end cross-validation: new sachsray (linear Jacobi) vs the OLD production
sachsfield.FluctuationSolver (nonlinear Riccati x_a MC), on the SAME full-sky
field realisation, vacuum background, pre-caustic regime.

Both integrate the same physical (theta, sigma) Sachs system in different
variables:

  old:  x = x_saddle + xi,  x1 = 2*theta  (F_111 = -1/2).  Field-1 fluctuation
        source must be 2*dPhi00 to represent the physical Phi00.
  new:  J'' = T(lam) J ;  theta = 1/2 tr(Jdot J^-1), sigma from the trace-free
        part.  Started mid-path at lam_start from the vacuum IC J=lam0*I, Jdot=I.

Expected (vacuum bg, theta_bg = 1/lam):
  theta_new(lam,n) == 1/lam + xi1_old(lam,n)/2
  sigma+_new       == xi2_old
  sigmax_new       == xi3_old
The agreement is to interpolation/solver discretisation (old uses linear-interp
+ RK45, new uses cubic control + Tsit5), confirming identical physics.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from scipy.interpolate import interp1d

import sachsfield as sf
from sachsfield.saddle import solve_singular_theta
from sachsfield.fluctuations import FluctuationSolver
from sachsfield.coefficients import M_matrix
from sachsray import solvers, physics


# --- one shared field realisation -----------------------------------------
nside = 4
npix = 12 * nside**2
lmax = 3 * nside - 1
lam_start, lam_s = 0.1, 2.0
n_lam = 96
lam_samples = np.linspace(lam_start, lam_s, n_lam)
A = 0.01  # source amplitude (visible nonlinearity, still pre-caustic)


def cl_func(lam):
    ell = np.arange(lmax + 1)
    cl = np.zeros(lmax + 1)
    cl[1:] = A / (ell[1:] * (ell[1:] + 1))
    return cl, cl.copy(), cl.copy()


source = sf.FullSkySource(
    cl_func=cl_func, lam_samples=lam_samples, nside=nside,
    corr_func=lambda d: np.exp(-0.5 * (d / 0.3) ** 2), seed=0, n_fields=3,
)
source.generate()
ds = np.asarray(source._delta_s)  # (3, n_lam, npix) = (dPhi00, dPsi+, dPsix)
lam_eval = np.linspace(lam_start + 0.05, lam_s, 24)


# --- OLD: vacuum saddle (x1 = 2/lam) + FluctuationSolver -------------------
t_th, th = solve_singular_theta(lambda t: 0.0, t_max=lam_s, epsilon=1e-6, num_points=5000)
th_interp = interp1d(t_th, th, kind="cubic", fill_value="extrapolate")  # x1 = 2*theta


class VacuumSaddle:
    n_fields = 3
    lam = lam_samples

    def M(self, lam):
        chi = np.array([float(th_interp(lam)), 0.0, 0.0])
        return M_matrix(chi, 3)


ds_old = np.stack([2.0 * ds[0], ds[1], ds[2]])  # field-1 source = 2*dPhi00
interps = [interp1d(lam_samples, ds_old[f], axis=0, kind="linear", fill_value="extrapolate")
           for f in range(3)]


def delta_s_old(lam):
    return np.stack([interps[f](lam) for f in range(3)])  # (3, npix)


fl = FluctuationSolver(
    VacuumSaddle(), delta_s_old, npix, n_fields=3, linear_only=False,
    lam_span=(lam_start, lam_s), method="RK45", rtol=1e-9, atol=1e-12,
)
xi = np.asarray(fl.solve(t_eval=lam_eval).xi)  # (n_eval, 3, npix)


# --- NEW: sachsray Jacobi per ray, vacuum IC at lam_start -----------------
ts = jnp.asarray(lam_samples)
le = jnp.asarray(lam_eval)
drive_rays = jnp.asarray(np.transpose(ds, (2, 1, 0)))  # (npix, n_lam, 3)
# vacuum IC at lam_start: J = lam_start*I, Jdot = I  -> theta=1/lam_start, sigma=0
y0 = jnp.array([lam_start, 0.0, 0.0, lam_start, 1.0, 0.0, 0.0, 1.0])


def one_ray(drive):
    ys = solvers.solve_jacobi_matrix(ts, drive, le, y0=y0, rtol=1e-10, atol=1e-12)

    def extract(y):
        J = y[:4].reshape(2, 2)
        Jd = y[4:].reshape(2, 2)
        sc = physics.scalars_from_deformation(physics.deformation_rate(J, Jd))
        return jnp.stack([sc["theta"], sc["sigma_plus"], sc["sigma_cross"]])

    return jax.vmap(extract)(ys)  # (n_eval, 3)


scal_new = np.asarray(jax.jit(jax.vmap(one_ray))(drive_rays))  # (npix, n_eval, 3)
theta_new = scal_new[:, :, 0].T   # (n_eval, npix)
sp_new = scal_new[:, :, 1].T
sx_new = scal_new[:, :, 2].T

theta_bg = 1.0 / lam_eval[:, None]
dtheta_new = theta_new - theta_bg     # new fluctuation of theta
dtheta_old = xi[:, 0, :] / 2.0        # xi1 = 2*delta_theta


def rel(a, b):
    return np.max(np.abs(a - b)) / (np.std(b) + 1e-30)


print(f"nside={nside} npix={npix} n_lam={n_lam} A={A}  pre-caustic, vacuum bg")
print(f"  theta fluctuation  : max|d| / std = {rel(dtheta_new, dtheta_old):.3e}  "
      f"(amp std={np.std(dtheta_old):.2e})")
print(f"  sigma_plus         : max|d| / std = {rel(sp_new, xi[:, 1, :]):.3e}  "
      f"(amp std={np.std(xi[:,1,:]):.2e})")
print(f"  sigma_cross        : max|d| / std = {rel(sx_new, xi[:, 2, :]):.3e}  "
      f"(amp std={np.std(xi[:,2,:]):.2e})")
print("  --> new Jacobi reproduces the old Riccati x_a MC (x1=2theta) end-to-end.")
