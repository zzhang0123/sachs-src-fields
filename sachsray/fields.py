"""
Build ``DrivingField`` objects from source-field realisations.

Field generation is intentionally decoupled from the solver: the ray-tracer
consumes a realisation on a (lambda, pixel) grid and does not care how it was
produced (Gaussian via healpy, non-Gaussian from an external simulation, etc.).
These helpers assemble the physical driving array

    Phi00(lambda, n) = Phi00_bg(lambda) + delta_Phi00(lambda, n)
    Psi0(lambda, n)  =          0        + delta_Psi0(lambda, n)

(the Weyl shear has zero background) into the (npix, n_lam, 3) layout that
``raytrace.trace_rays`` expects. The line-of-sight smoothing is handled inside
the solver (cubic control path), so only the sampled grid is needed here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .raytrace import DrivingField


def driving_from_components(
    ts: np.ndarray,
    delta_phi00: np.ndarray,
    delta_psi_re: np.ndarray,
    delta_psi_im: np.ndarray,
    phi00_bg: np.ndarray | None = None,
    *,
    dtype=jnp.float32,
) -> DrivingField:
    """Assemble a ``DrivingField`` from per-channel fluctuation maps + background.

    Parameters
    ----------
    ts : (n_lam,) increasing lambda grid.
    delta_phi00, delta_psi_re, delta_psi_im : (n_lam, npix) zero-mean fluctuation
        maps for Phi00, Re Psi0, Im Psi0.
    phi00_bg : (n_lam,) background (mean) Phi00 along the ray (e.g. from
        ``perFLRW`` -- negative for focusing). Defaults to zeros (vacuum
        background, D_bg = lambda).
    dtype : array dtype (default float32, the production memory mode).

    Returns
    -------
    DrivingField with ``maps`` of shape (npix, n_lam, 3).
    """
    ts = np.asarray(ts)
    n_lam = ts.shape[0]
    dphi = np.asarray(delta_phi00)
    dpre = np.asarray(delta_psi_re)
    dpim = np.asarray(delta_psi_im)
    for name, arr in (("delta_phi00", dphi), ("delta_psi_re", dpre), ("delta_psi_im", dpim)):
        if arr.ndim != 2 or arr.shape[0] != n_lam:
            raise ValueError(
                f"{name} must have shape (n_lam={n_lam}, npix); got {arr.shape}"
            )
    npix = dphi.shape[1]
    if not (dpre.shape[1] == npix and dpim.shape[1] == npix):
        raise ValueError("all fluctuation maps must share the same npix")

    if phi00_bg is None:
        phi00_bg = np.zeros(n_lam)
    phi00_bg = np.asarray(phi00_bg)
    if phi00_bg.shape != (n_lam,):
        raise ValueError(f"phi00_bg must have shape ({n_lam},); got {phi00_bg.shape}")

    phi00_total = phi00_bg[:, None] + dphi              # (n_lam, npix)
    maps = np.stack([phi00_total, dpre, dpim], axis=-1)  # (n_lam, npix, 3)
    maps = np.transpose(maps, (1, 0, 2))                 # (npix, n_lam, 3)

    return DrivingField(
        ts=jnp.asarray(ts, dtype=dtype),
        maps=jnp.asarray(maps, dtype=dtype),
        phi00_bg=jnp.asarray(phi00_bg, dtype=dtype),
    )


def driving_from_source(source, phi00_bg: np.ndarray | None = None, *, dtype=jnp.float32) -> DrivingField:
    """Bridge a generated sachsfield source object to a ``DrivingField``.

    ``source`` is duck-typed: it must expose ``lam_samples`` (n_lam,) and, after
    generation, ``_delta_s`` of shape (3, n_lam, npix) -- satisfied by
    ``sachsfield.FullSkySource`` and ``FlatSkySource``. This keeps the bridge
    decoupled (no hard import of the legacy package).

    Parameters
    ----------
    source : object with ``.lam_samples``, ``._delta_s``, ``.generate()``,
        ``._generated`` (the (3, n_lam, npix) fluctuation maps for
        Phi00, Re Psi0, Im Psi0).
    phi00_bg : (n_lam,) background Phi00 on ``source.lam_samples`` (e.g.
        ``cosmo.background_phi00_of_lambda(source.lam_samples)``). Default zeros.
    """
    if not getattr(source, "_generated", False):
        source.generate()
    ds = np.asarray(source._delta_s)  # (3, n_lam, npix)
    if ds.ndim != 3 or ds.shape[0] != 3:
        raise ValueError(f"source._delta_s must be (3, n_lam, npix); got {ds.shape}")
    return driving_from_components(
        source.lam_samples, ds[0], ds[1], ds[2], phi00_bg, dtype=dtype,
    )


def driving_from_potential_alms(
    ts: np.ndarray,
    alms: np.ndarray,
    nside: int,
    *,
    lmax: int,
    scale: float = 1.0,
    phi00_bg: np.ndarray | None = None,
    dtype=jnp.float32,
) -> DrivingField:
    """Build a pure-E ``DrivingField`` from per-shell scalar-potential alms.

    Both driving fields are derived from ONE scalar ``S`` per shell so that the
    Weyl shear is the trace-free screen Hessian (a spin-2 ``eth eth S`` field,
    pure E for every realisation) and the Ricci focusing is the screen
    Laplacian::

        Phi00 = scale * Lap S      (spin-0 alm multiplied by -l(l+1))
        Psi0  = scale * eth eth S  (E-alm multiplied by -sqrt((l+2)!/(l-2)!), B = 0)

    in the HEALPix ``(e_theta, e_phi)`` basis returned by ``healpy.alm2map_spin``.
    The SAME sign on both multipliers is what makes the traced linear shear
    reproduce the convergence (``kappa_E = +gamma_E``) with no B-mode; flipping
    the sign of ``W2`` (or of one multiplier) yields B/E ~ 1. Pinned by the
    2026-09-03 audit (``prototype/bmode_audit/A1/sachsray_eb_basis_test.py``,
    ``prototype/bmode_audit/n1_sachsray``) and by ``tests/test_fields_pure_e.py``.

    Use this instead of the legacy ``sachsfield.FullSkySource``, which draws
    Re Psi0 and Im Psi0 as independent scalar maps and therefore has E = B.

    Parameters
    ----------
    ts : (n_lam,) increasing affine-parameter grid.
    alms : (n_lam, n_alm) complex healpy alms of the potential on each shell,
        healpy ordering for ``lmax``.
    nside : HEALPix resolution of the output maps.
    lmax : band limit of ``alms``.
    scale : overall multiplier applied to both fields (e.g. 1/2 for T = Hess/2).
    phi00_bg : (n_lam,) background Phi00 (default zeros, vacuum).
    """
    import healpy as hp

    ts = np.asarray(ts)
    alms = np.asarray(alms)
    if alms.ndim != 2 or alms.shape[0] != ts.shape[0]:
        raise ValueError(f"alms must have shape (n_lam={ts.shape[0]}, n_alm); got {alms.shape}")
    ell = np.arange(lmax + 1, dtype=float)
    f_lap = -scale * ell * (ell + 1.0)
    f_e = np.zeros(lmax + 1)
    f_e[2:] = -scale * np.sqrt((ell[2:] + 2.0) * (ell[2:] + 1.0) * ell[2:] * (ell[2:] - 1.0))
    phi00, w1, w2 = [], [], []
    for a in alms:
        phi00.append(hp.alm2map(hp.almxfl(a, f_lap), nside, lmax=lmax))
        q, u = hp.alm2map_spin([hp.almxfl(a, f_e), np.zeros_like(a)], nside, 2, lmax)
        w1.append(q)
        w2.append(u)
    return driving_from_components(
        ts, np.stack(phi00), np.stack(w1), np.stack(w2), phi00_bg, dtype=dtype,
    )
