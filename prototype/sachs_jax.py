"""
Prototype JAX/diffrax core for a full-sky Sachs ray-tracer.

This is a decision-support prototype (see prototype/bench.py) for the
``sachsfield`` redesign. It implements the *physical* Sachs optical system in
two equivalent formulations and integrates them with diffrax so that:

* every ray is an independent ODE (``vmap`` over rays, GPU-ready),
* the (possibly non-Gaussian, possibly rough) driving fields are supplied as
  *given realisations* on a lambda-grid and fed in as a smooth control path,
* memory is bounded by processing rays in chunks and saving only the
  observables at the source plane.

Physics (draft STF_lensing, appendix.tex:68-200)
------------------------------------------------
Sachs scalar (Riccati) system, with NP spin coefficients rho (spin-0) and
sigma (spin-2) and driving fields Phi00 (Ricci focusing, real) and
Psi0 (Weyl shear, complex)::

    drho/dlam   = rho^2 + sigma*conj(sigma) - Phi00
    dsigma/dlam = (rho + conj(rho)) sigma   + Psi0

Equivalent linear Jacobi form (the robust object)::

    Jdot..(lam) = T(lam) @ J(lam),   J(0) = 0,  Jdot(0) = I

with the 2x2 optical tidal matrix built from the driving fields::

    T = [[Phi00 + Re Psi0,        Im Psi0],
         [       Im Psi0,  Phi00 - Re Psi0]]

The scalar (shear-free) case ``D'' = Phi00 D`` has the closed-form analytic
solution used for validation; e.g. constant ``Phi00 = -w^2`` gives
``D = sin(w*lam)/w`` with a caustic at ``lam = pi/w`` (where the Riccati rho
= -Ddot/D diverges but D passes smoothly through zero).
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp
import diffrax


# --------------------------------------------------------------------------
# Driving field -> smooth (C2) control path
# --------------------------------------------------------------------------
def cubic_control(ts: jax.Array, ys: jax.Array) -> diffrax.CubicInterpolation:
    """Build a C2 cubic-spline control path from a sampled driving field.

    Parameters
    ----------
    ts : (n_lam,) array of lambda samples (strictly increasing).
    ys : (n_lam, ...) array of field values at the samples.

    Returns
    -------
    diffrax.CubicInterpolation
        Smooth interpolant; ``.evaluate(t)`` returns the field at ``t``.

    Notes
    -----
    Smoothness matters: a piecewise-*linear* interpolant (the current
    ``scipy.interpolate.interp1d(kind='linear')`` in sources.py) is only C0,
    so its derivative jumps at every knot and an adaptive RK stepper wastes
    effort / loses accuracy there. A backward-Hermite cubic is C1/C2 and the
    stepper sees smooth forcing.
    """
    coeffs = diffrax.backward_hermite_coefficients(ts, ys)
    return diffrax.CubicInterpolation(ts, coeffs)


# --------------------------------------------------------------------------
# Jacobi (linear) formulation -- the robust workhorse
# --------------------------------------------------------------------------
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
    """Integrate the scalar Jacobi equation ``D'' = Phi00(lam) D``.

    State ``y = (D, Ddot)``, initial ``D(0)=0, Ddot(0)=1``.

    Returns
    -------
    (len(lam_eval), 2) array of ``(D, Ddot)`` at the requested lambdas.
    """
    solver = solver or diffrax.Tsit5()
    ctrl = cubic_control(ts_grid, phi00_grid)

    def vf(t, y, args):
        D, Dd = y
        phi = ctrl.evaluate(t)
        return jnp.stack([Dd, phi * D])

    t0 = ts_grid[0]
    t1 = ts_grid[-1]
    y0 = jnp.array([0.0, 1.0], dtype=phi00_grid.dtype)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf),
        solver,
        t0=t0,
        t1=t1,
        dt0=None,
        y0=y0,
        saveat=diffrax.SaveAt(ts=lam_eval),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
    )
    return sol.ys


def _tidal_matrix(phi00: jax.Array, psi_re: jax.Array, psi_im: jax.Array) -> jax.Array:
    """Optical tidal matrix T(lam) (2x2) from the driving fields."""
    return jnp.array(
        [[phi00 + psi_re, psi_im], [psi_im, phi00 - psi_re]],
        dtype=phi00.dtype,
    )


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
    """Integrate the 2x2 Jacobi matrix equation ``Jddot = T(lam) J``.

    Parameters
    ----------
    drive_grid : (n_lam, 3) array of ``(Phi00, Re Psi0, Im Psi0)`` samples.

    State is ``(J[2x2], Jdot[2x2])`` flattened to 8 reals. Initial condition
    ``J(0)=0, Jdot(0)=I``.

    Returns
    -------
    (len(lam_eval), 8) array; first 4 columns are ``J`` (row-major), last 4
    are ``Jdot``.
    """
    solver = solver or diffrax.Tsit5()
    ctrl = cubic_control(ts_grid, drive_grid)  # vector-valued control

    def vf(t, y, args):
        J = y[:4].reshape(2, 2)
        Jd = y[4:].reshape(2, 2)
        d = ctrl.evaluate(t)
        T = _tidal_matrix(d[0], d[1], d[2])
        Jdd = T @ J
        return jnp.concatenate([Jd.reshape(-1), Jdd.reshape(-1)])

    t0 = ts_grid[0]
    t1 = ts_grid[-1]
    J0 = jnp.zeros((2, 2), dtype=drive_grid.dtype)
    Jd0 = jnp.eye(2, dtype=drive_grid.dtype)
    y0 = jnp.concatenate([J0.reshape(-1), Jd0.reshape(-1)])
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf),
        solver,
        t0=t0,
        t1=t1,
        dt0=None,
        y0=y0,
        saveat=diffrax.SaveAt(ts=lam_eval),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
    )
    return sol.ys


# --------------------------------------------------------------------------
# Riccati (nonlinear) formulation -- additive driving, fragile at caustics
# --------------------------------------------------------------------------
def solve_riccati_scalar(
    ts_grid: jax.Array,
    phi00_grid: jax.Array,
    lam_eval: jax.Array,
    *,
    rho0: float,
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = 100_000,
    solver: diffrax.AbstractSolver | None = None,
) -> jax.Array:
    """Integrate the shear-free Riccati equation ``drho/dlam = rho^2 - Phi00``.

    Needs a finite starting rho0 (the observer vertex rho ~ -1/lam is singular,
    which is exactly why this form is fragile). Returns rho at lam_eval.
    """
    solver = solver or diffrax.Tsit5()
    ctrl = cubic_control(ts_grid, phi00_grid)

    def vf(t, y, args):
        rho = y[0]
        phi = ctrl.evaluate(t)
        return jnp.stack([rho * rho - phi])

    t0 = ts_grid[0]
    t1 = ts_grid[-1]
    y0 = jnp.array([rho0], dtype=phi00_grid.dtype)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(vf),
        solver,
        t0=t0,
        t1=t1,
        dt0=None,
        y0=y0,
        saveat=diffrax.SaveAt(ts=lam_eval),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
    )
    return sol.ys[:, 0]


# --------------------------------------------------------------------------
# Observables from the Jacobi matrix at the source plane
# --------------------------------------------------------------------------
def observables_from_jacobi(J: jax.Array, D_bg: jax.Array) -> dict:
    """Weak-lensing observables from the 2x2 Jacobi map.

    The distortion (magnification) matrix is ``A = J / D_bg`` relative to the
    isotropic background angular-diameter distance ``D_bg``. Then
    ``A = [[1-kappa-gamma1, -gamma2+omega], [-gamma2-omega, 1-kappa+gamma1]]``.
    """
    A = J / D_bg
    kappa = 1.0 - 0.5 * (A[0, 0] + A[1, 1])
    gamma1 = 0.5 * (A[1, 1] - A[0, 0])
    gamma2 = -0.5 * (A[0, 1] + A[1, 0])
    omega = 0.5 * (A[0, 1] - A[1, 0])
    return {"kappa": kappa, "gamma1": gamma1, "gamma2": gamma2, "omega": omega}


# --------------------------------------------------------------------------
# vmap over rays + chunked driver (memory management)
# --------------------------------------------------------------------------
def solve_rays_chunked(
    single_ray: Callable[[jax.Array], jax.Array],
    drive_rays: jax.Array,
    *,
    chunk: int = 4096,
) -> jax.Array:
    """Map ``single_ray`` over a leading ray axis in memory-bounded chunks.

    Parameters
    ----------
    single_ray : callable mapping one ray's driving array -> its output.
        Must be ``vmap``-able over the leading axis of ``drive_rays``.
    drive_rays : (n_rays, n_lam, ...) array of per-ray driving realisations.
    chunk : rays processed per vmap call. Peak memory scales with ``chunk``,
        NOT ``n_rays`` -- this is the lever for nside>=256 full-sky maps.

    Returns
    -------
    (n_rays, ...) stacked outputs.
    """
    batched = jax.jit(jax.vmap(single_ray))
    n = drive_rays.shape[0]
    out = []
    for start in range(0, n, chunk):
        block = drive_rays[start : start + chunk]
        res = batched(block)
        out.append(jax.device_get(res))  # pull to host, free device buffer
        del res
    return jnp.concatenate([jnp.asarray(o) for o in out], axis=0)
