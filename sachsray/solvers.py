"""
diffrax integrators for the Sachs ray-tracer.

Three integrators, all driven by a *given realisation* of the driving field
supplied on a lambda-grid and fed in as a smooth (C2) cubic control path:

* ``solve_jacobi_scalar``  -- linear ``Ddot = Phi00 D`` (shear-free).
* ``solve_jacobi_matrix``  -- linear ``Jddot = T(lam) J`` (full 2x2 map).
* ``solve_riccati``        -- nonlinear physical Sachs (theta, sigma) system;
                              reference / additive-noise-SDE-capable form.

The Jacobi integrators are the robust workhorse (caustic-safe, no vertex
rescaling). All functions are ``vmap``-able over a leading ray axis.
"""

from __future__ import annotations

import diffrax
import jax
import jax.numpy as jnp

from . import physics


def cubic_control(ts: jax.Array, ys: jax.Array) -> diffrax.CubicInterpolation:
    """Smooth (backward-Hermite cubic, C1/C2) control path from samples.

    ``ts``: (n_lam,) increasing. ``ys``: (n_lam, ...) field samples.
    A cubic control gives the adaptive stepper smooth forcing -- unlike the
    legacy ``interp1d(kind='linear')`` whose derivative jumps at every knot.
    """
    coeffs = diffrax.backward_hermite_coefficients(ts, ys)
    return diffrax.CubicInterpolation(ts, coeffs)


def _solve(term, y0, ts_grid, lam_eval, solver, rtol, atol, max_steps):
    return diffrax.diffeqsolve(
        term,
        solver,
        t0=ts_grid[0],
        t1=ts_grid[-1],
        dt0=None,
        y0=y0,
        saveat=diffrax.SaveAt(ts=lam_eval),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
    ).ys


def solve_jacobi_scalar(
    ts_grid: jax.Array,
    phi00_grid: jax.Array,
    lam_eval: jax.Array,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 100_000,
    solver: diffrax.AbstractSolver | None = None,
) -> jax.Array:
    """Integrate ``Ddot = Phi00(lam) D``; state (D, Ddot), IC (0, 1).

    Returns (len(lam_eval), 2) of (D, Ddot).
    """
    solver = solver or diffrax.Tsit5()
    ctrl = cubic_control(ts_grid, phi00_grid)

    def vf(t, y, args):
        D, Dd = y
        return jnp.stack([Dd, ctrl.evaluate(t) * D])

    y0 = jnp.array([0.0, 1.0], dtype=phi00_grid.dtype)
    return _solve(diffrax.ODETerm(vf), y0, ts_grid, lam_eval, solver, rtol, atol, max_steps)


def solve_jacobi_matrix(
    ts_grid: jax.Array,
    drive_grid: jax.Array,
    lam_eval: jax.Array,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 100_000,
    solver: diffrax.AbstractSolver | None = None,
) -> jax.Array:
    """Integrate ``Jddot = T(lam) J``; state (J[4], Jdot[4]) flattened to 8.

    ``drive_grid``: (n_lam, 3) samples of (Phi00, W1, W2). IC J(0)=0, Jdot(0)=I.
    Returns (len(lam_eval), 8): first 4 cols J (row-major), last 4 Jdot.
    """
    solver = solver or diffrax.Tsit5()
    ctrl = cubic_control(ts_grid, drive_grid)

    def vf(t, y, args):
        J = y[:4].reshape(2, 2)
        Jd = y[4:].reshape(2, 2)
        d = ctrl.evaluate(t)
        T = physics.tidal_matrix(d[0], d[1], d[2])
        return jnp.concatenate([Jd.reshape(-1), (T @ J).reshape(-1)])

    J0 = jnp.zeros((2, 2), dtype=drive_grid.dtype)
    Jd0 = jnp.eye(2, dtype=drive_grid.dtype)
    y0 = jnp.concatenate([J0.reshape(-1), Jd0.reshape(-1)])
    return _solve(diffrax.ODETerm(vf), y0, ts_grid, lam_eval, solver, rtol, atol, max_steps)


def solve_riccati(
    ts_grid: jax.Array,
    drive_grid: jax.Array,
    lam_eval: jax.Array,
    *,
    chi0: jax.Array | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 100_000,
    solver: diffrax.AbstractSolver | None = None,
) -> jax.Array:
    """Integrate the physical Sachs (Riccati) system; state (theta, sp, sc).

    ``drive_grid``: (n_lam, 3) of (Phi00, W1, W2). Needs a finite ``chi0``
    (the vertex theta ~ 1/lam is singular -- this is why the Riccati form is
    fragile and the Jacobi form is preferred). Returns (len(lam_eval), 3).
    """
    solver = solver or diffrax.Tsit5()
    ctrl = cubic_control(ts_grid, drive_grid)

    def vf(t, y, args):
        return physics.riccati_rhs(y, ctrl.evaluate(t))

    if chi0 is None:
        chi0 = jnp.zeros(3, dtype=drive_grid.dtype)
    return _solve(diffrax.ODETerm(vf), chi0, ts_grid, lam_eval, solver, rtol, atol, max_steps)
