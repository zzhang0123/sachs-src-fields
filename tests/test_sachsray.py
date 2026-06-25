"""
Validation suite for the sachsray full-sky Sachs ray-tracer.

Pins the physics conventions (especially the Weyl-shear sign in the tidal
matrix) by cross-validating the linear Jacobi formulation against the nonlinear
Riccati formulation -- the boundary-validation methodology: two formulations of
the same physics must agree where both are valid.

Run:  pytest tests/test_sachsray.py -q
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import sachsray as sr  # noqa: E402
from sachsray import physics, solvers  # noqa: E402


# --------------------------------------------------------------------------
# smooth analytic driving field (mild -> no caustic over the test range)
# --------------------------------------------------------------------------
def _drive(ts: np.ndarray) -> np.ndarray:
    phi = -0.05 + 0.02 * np.cos(ts)
    w1 = 0.03 * np.sin(0.7 * ts + 0.3)
    w2 = 0.02 * np.cos(0.5 * ts)
    return np.stack([phi, w1, w2], axis=-1)


# ==========================================================================
# Analytic pins
# ==========================================================================
def test_jacobi_scalar_vacuum():
    """Phi00 = 0 -> D = lam, Ddot = 1."""
    ts = jnp.linspace(0.0, 3.0, 200)
    phi = jnp.zeros_like(ts)
    le = jnp.linspace(1e-3, 3.0, 100)
    ys = solvers.solve_jacobi_scalar(ts, phi, le, rtol=1e-11, atol=1e-13)
    D, Dd = np.asarray(ys[:, 0]), np.asarray(ys[:, 1])
    assert np.allclose(D, np.asarray(le), atol=1e-8)
    assert np.allclose(Dd, 1.0, atol=1e-8)


def test_jacobi_scalar_constant_focusing():
    """Phi00 = -w^2 -> D = sin(w lam)/w (exact)."""
    w = 1.3
    ts = jnp.linspace(0.0, 2.0, 400)
    phi = jnp.full_like(ts, -(w**2))
    le = jnp.linspace(1e-3, 2.0, 200)
    ys = solvers.solve_jacobi_scalar(ts, phi, le, rtol=1e-11, atol=1e-13)
    D = np.asarray(ys[:, 0])
    D_true = np.sin(w * np.asarray(le)) / w
    assert np.max(np.abs(D - D_true)) < 1e-9


def test_jacobi_passes_through_caustic():
    """The Jacobi amplitude stays finite through caustics (D crosses zero)."""
    w = 1.0
    lam_max = 2.0 * np.pi + 0.3  # two caustics at pi, 2pi
    ts = jnp.linspace(0.0, lam_max, 600)
    phi = jnp.full_like(ts, -(w**2))
    le = jnp.linspace(1e-3, lam_max, 400)
    ys = solvers.solve_jacobi_scalar(ts, phi, le, rtol=1e-10, atol=1e-12)
    D = np.asarray(ys[:, 0])
    assert np.all(np.isfinite(D))
    # sign change confirms it went through a zero (caustic), not around it
    assert np.any(D[: len(D) // 2] > 0) and np.any(D < 0)


# ==========================================================================
# Convention-pinning cross-validation: Jacobi <-> Riccati
# ==========================================================================
def test_jacobi_matches_riccati():
    """Optical scalars extracted from the matrix Jacobi map must satisfy the
    Riccati system -- this PINS the Weyl-shear sign in physics.tidal_matrix."""
    lam0, lam1 = 0.0, 2.0
    n = 600
    ts = jnp.linspace(lam0, lam1, n)
    drive = jnp.asarray(_drive(np.asarray(ts)))

    lam_eval = jnp.linspace(0.2, lam1, 60)  # away from the vertex (J invertible)
    yj = solvers.solve_jacobi_matrix(ts, drive, lam_eval, rtol=1e-11, atol=1e-13)

    def extract(y):
        J = y[:4].reshape(2, 2)
        Jd = y[4:].reshape(2, 2)
        S = physics.deformation_rate(J, Jd)
        sc = physics.scalars_from_deformation(S)
        return jnp.stack([sc["theta"], sc["sigma_plus"], sc["sigma_cross"]])

    chi_jac = np.asarray(jax.vmap(extract)(yj))  # (60, 3)

    # integrate the Riccati from the first matched point with the SAME field
    ts2 = jnp.linspace(0.2, lam1, n)
    drive2 = jnp.asarray(_drive(np.asarray(ts2)))
    chi0 = jnp.asarray(chi_jac[0])
    yr = np.asarray(
        solvers.solve_riccati(ts2, drive2, lam_eval, chi0=chi0, rtol=1e-11, atol=1e-13)
    )

    # theta (trace part) and BOTH shear components must agree
    assert np.max(np.abs(chi_jac[:, 0] - yr[:, 0])) < 1e-6, "theta mismatch"
    assert np.max(np.abs(chi_jac[:, 1] - yr[:, 1])) < 1e-6, "sigma_plus mismatch (Weyl sign!)"
    assert np.max(np.abs(chi_jac[:, 2] - yr[:, 2])) < 1e-6, "sigma_cross mismatch (Weyl sign!)"


def test_omega_is_zero_for_twist_free():
    """A symmetric tidal matrix -> no image rotation (omega = 0)."""
    ts = jnp.linspace(0.0, 2.0, 400)
    drive = jnp.asarray(_drive(np.asarray(ts)))
    le = jnp.linspace(0.2, 2.0, 30)
    yj = solvers.solve_jacobi_matrix(ts, drive, le, rtol=1e-10, atol=1e-12)

    def omega(y):
        J = y[:4].reshape(2, 2)
        Jd = y[4:].reshape(2, 2)
        return physics.scalars_from_deformation(physics.deformation_rate(J, Jd))["omega"]

    om = np.asarray(jax.vmap(omega)(yj))
    assert np.max(np.abs(om)) < 1e-9


# ==========================================================================
# Observables
# ==========================================================================
def test_convergence_sign_for_focusing():
    """Pure focusing (Phi00 < 0, no shear) gives positive convergence."""
    eps = 0.05
    lam_s = 2.0
    ts = jnp.linspace(0.0, lam_s, 300)
    drive = jnp.stack([jnp.full_like(ts, -eps), jnp.zeros_like(ts), jnp.zeros_like(ts)], axis=-1)
    le = jnp.asarray([lam_s])
    yj = solvers.solve_jacobi_matrix(ts, drive, le, rtol=1e-10, atol=1e-12)[0]
    J = yj[:4].reshape(2, 2)
    D_bg = lam_s  # phi00_bg = 0 -> D_bg = lam
    obs = physics.observables_from_jacobi(J, D_bg)
    kappa = float(obs["kappa"])
    # analytic: D = sin(sqrt(eps) lam)/sqrt(eps); kappa = 1 - D/lam > 0
    D = np.sin(np.sqrt(eps) * lam_s) / np.sqrt(eps)
    assert kappa > 0
    assert abs(kappa - (1.0 - D / lam_s)) < 1e-6
    assert abs(float(obs["gamma1"])) < 1e-9 and abs(float(obs["gamma2"])) < 1e-9


def test_weyl_sources_shear():
    """A nonzero W1 produces gamma1; W2 produces gamma2."""
    lam_s = 2.0
    ts = jnp.linspace(0.0, lam_s, 300)
    drive = jnp.stack(
        [jnp.full_like(ts, -0.02), jnp.full_like(ts, 0.03), jnp.zeros_like(ts)], axis=-1
    )
    le = jnp.asarray([lam_s])
    yj = solvers.solve_jacobi_matrix(ts, drive, le, rtol=1e-10, atol=1e-12)[0]
    J = yj[:4].reshape(2, 2)
    obs = physics.observables_from_jacobi(J, lam_s)
    assert abs(float(obs["gamma1"])) > 1e-4
    assert abs(float(obs["gamma2"])) < 1e-9  # W2 = 0


# ==========================================================================
# High-level ray-tracer (vmap + chunked)
# ==========================================================================
def test_trace_rays_smoke():
    npix, n_lam, lam_s = 64, 128, 2.0
    ts = jnp.linspace(0.0, lam_s, n_lam)
    rng = np.random.default_rng(0)
    maps = np.zeros((npix, n_lam, 3))
    tt = np.asarray(ts)
    for p in range(npix):
        ph = rng.uniform(0, 2 * np.pi, 3)
        maps[p, :, 0] = -0.05 + 0.01 * np.sin(tt + ph[0])
        maps[p, :, 1] = 0.02 * np.sin(0.7 * tt + ph[1])
        maps[p, :, 2] = 0.02 * np.cos(0.5 * tt + ph[2])
    field = sr.DrivingField(
        ts=ts, maps=jnp.asarray(maps), phi00_bg=jnp.full_like(ts, -0.05)
    )
    out = sr.trace_rays(field, lam_s, chunk=32, rtol=1e-7, atol=1e-9)
    for k in ("kappa", "gamma1", "gamma2", "omega"):
        assert out[k].shape == (npix,)
        assert np.all(np.isfinite(np.asarray(out[k])))


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
