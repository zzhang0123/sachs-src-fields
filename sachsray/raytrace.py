"""
High-level full-sky Sachs ray-tracer: driving fields in, lensing maps out.

The user supplies driving-field realisations on a shared lambda-grid (any
statistics -- Gaussian or non-Gaussian) and gets per-pixel weak-lensing
observables (kappa, gamma1, gamma2, omega) at the source plane.

Memory management
-----------------
Rays are integrated in chunks (``vmap`` over a ray-block, jitted once). Peak
device memory scales with ``chunk``, NOT the map size, so nside >= 256 maps are
streamed: results are pulled to host and device buffers freed between chunks.
Only the source-plane Jacobi state is saved (no dense per-lambda trajectories).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from . import physics, solvers


@dataclass(frozen=True)
class DrivingField:
    """A realisation of the driving fields on a shared lambda-grid.

    Attributes
    ----------
    ts : (n_lam,) increasing affine-parameter grid.
    maps : (npix, n_lam, 3) per-ray samples of (Phi00, W1, W2).
    phi00_bg : (n_lam,) background/mean Phi00 for the reference distance D_bg.
    """

    ts: jax.Array
    maps: jax.Array
    phi00_bg: jax.Array

    @property
    def npix(self) -> int:
        return int(self.maps.shape[0])


def background_distance(field: DrivingField, lam_source: jax.Array, **kw) -> jax.Array:
    """Scalar background Jacobi amplitude D_bg(lam_source) from phi00_bg."""
    ys = solvers.solve_jacobi_scalar(field.ts, field.phi00_bg, lam_source, **kw)
    return ys[:, 0]  # (len(lam_source),)


def trace_rays(
    field: DrivingField,
    lam_source: float,
    *,
    chunk: int = 8192,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    solver=None,
) -> dict:
    """Trace all rays to ``lam_source`` and return observable maps.

    Returns a dict of (npix,) arrays: kappa, gamma1, gamma2, omega.
    """
    ts = field.ts
    lam_eval = jnp.asarray([lam_source], dtype=ts.dtype)
    D_bg = background_distance(field, lam_eval, rtol=rtol, atol=atol)[0]

    def single_ray(drive):  # (n_lam, 3) -> dict of scalars
        y = solvers.solve_jacobi_matrix(
            ts, drive, lam_eval, rtol=rtol, atol=atol, solver=solver
        )[0]
        J = y[:4].reshape(2, 2)
        return physics.observables_from_jacobi(J, D_bg)

    batched = jax.jit(jax.vmap(single_ray))
    npix = field.npix
    acc: dict[str, list] = {"kappa": [], "gamma1": [], "gamma2": [], "omega": []}
    for start in range(0, npix, chunk):
        block = field.maps[start : start + chunk]
        out = batched(block)
        for k in acc:
            acc[k].append(jax.device_get(out[k]))  # to host, free device
        del out
    return {k: jnp.concatenate([jnp.asarray(v) for v in acc[k]]) for k in acc}
