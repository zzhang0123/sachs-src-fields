"""
End-to-end cross-validation: the new sachsray Jacobi solver must reproduce the
OLD production sachsfield.FluctuationSolver (nonlinear Riccati x_a MC) on the
same full-sky realisation, vacuum background, pre-caustic regime.

Both integrate the same physical (theta, sigma) Sachs system; the old uses
x1 = 2*theta (F_111 = -1/2) so its field-1 fluctuation source is 2*dPhi00, and
its xi1 = 2*delta_theta. Agreement is to interpolation/solver discretisation
(old: linear-interp + RK45; new: cubic control + Tsit5).
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sachsray import solvers, physics  # noqa: E402


def test_jacobi_reproduces_old_fluctuation_mc():
    pytest.importorskip("healpy")
    sf = pytest.importorskip("sachsfield")
    from sachsfield.saddle import solve_singular_theta
    from sachsfield.fluctuations import FluctuationSolver
    from sachsfield.coefficients import M_matrix
    from scipy.interpolate import interp1d

    nside = 4
    npix = 12 * nside**2
    lmax = 3 * nside - 1
    lam_start, lam_s = 0.1, 2.0
    n_lam = 96
    lam_samples = np.linspace(lam_start, lam_s, n_lam)
    A = 0.01

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

    # OLD: vacuum saddle (x1 = 2/lam) + FluctuationSolver, source-1 = 2*dPhi00
    t_th, th = solve_singular_theta(lambda t: 0.0, t_max=lam_s, epsilon=1e-6, num_points=5000)
    th_interp = interp1d(t_th, th, kind="cubic", fill_value="extrapolate")

    class VacuumSaddle:
        n_fields = 3
        lam = lam_samples

        def M(self, lam):
            return M_matrix(np.array([float(th_interp(lam)), 0.0, 0.0]), 3)

    ds_old = np.stack([2.0 * ds[0], ds[1], ds[2]])
    interps = [interp1d(lam_samples, ds_old[f], axis=0, kind="linear", fill_value="extrapolate")
               for f in range(3)]
    fl = FluctuationSolver(
        VacuumSaddle(), lambda lam: np.stack([ip(lam) for ip in interps]), npix,
        n_fields=3, linear_only=False, lam_span=(lam_start, lam_s),
        method="RK45", rtol=1e-9, atol=1e-12,
    )
    xi = np.asarray(fl.solve(t_eval=lam_eval).xi)  # (n_eval, 3, npix)

    # NEW: sachsray Jacobi per ray, vacuum IC at lam_start
    ts = jnp.asarray(lam_samples)
    le = jnp.asarray(lam_eval)
    drive_rays = jnp.asarray(np.transpose(ds, (2, 1, 0)))  # (npix, n_lam, 3)
    y0 = jnp.array([lam_start, 0.0, 0.0, lam_start, 1.0, 0.0, 0.0, 1.0])

    def one_ray(drive):
        ys = solvers.solve_jacobi_matrix(ts, drive, le, y0=y0, rtol=1e-10, atol=1e-12)

        def extract(y):
            J, Jd = y[:4].reshape(2, 2), y[4:].reshape(2, 2)
            sc = physics.scalars_from_deformation(physics.deformation_rate(J, Jd))
            return jnp.stack([sc["theta"], sc["sigma_plus"], sc["sigma_cross"]])

        return jax.vmap(extract)(ys)

    scal_new = np.asarray(jax.jit(jax.vmap(one_ray))(drive_rays))  # (npix, n_eval, 3)
    theta_bg = 1.0 / lam_eval[:, None]
    dtheta_new = scal_new[:, :, 0].T - theta_bg
    dtheta_old = xi[:, 0, :] / 2.0  # xi1 = 2*delta_theta

    def rel(a, b):
        return np.max(np.abs(a - b)) / (np.std(b) + 1e-30)

    # all three optical-scalar fluctuations agree to interpolation precision
    assert rel(dtheta_new, dtheta_old) < 5e-3, "theta fluctuation mismatch"
    assert rel(scal_new[:, :, 1].T, xi[:, 1, :]) < 5e-3, "sigma_plus mismatch"
    assert rel(scal_new[:, :, 2].T, xi[:, 2, :]) < 5e-3, "sigma_cross mismatch"
    # sanity: the fluctuations are non-trivial (the test isn't vacuously passing)
    assert np.std(dtheta_old) > 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
