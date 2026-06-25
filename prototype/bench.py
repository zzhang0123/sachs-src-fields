"""
Decision-support benchmark for the sachsfield -> full-sky Sachs ray-tracer
redesign. Run from the repo root:

    python prototype/bench.py

It answers three questions with numbers:

  EXP 1  Correctness + caustic robustness
         Does the diffrax Jacobi solver reproduce the closed-form analytic
         solution, and does it survive a caustic where the Riccati/Sachs form
         (the current x_a formulation) blows up?

  EXP 2  Robustness to rough (discontinuous-derivative) driving
         The current code feeds the driving field via linear interpolation
         (C0, kinks at every knot). How much accuracy does that cost an
         adaptive stepper vs a smooth cubic control path?

  EXP 3  Full-sky scaling + memory management
         vmap independent per-ray integration in float32, processed in chunks
         so peak memory scales with the chunk size, not the map size.

Everything is sized small on purpose (CPU, single ray-block) -- the point is
to measure ratios and project, not to run a full nside>=256 map here.
"""

from __future__ import annotations

import gc
import resource
import time

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)  # accuracy experiments in float64
import jax.numpy as jnp  # noqa: E402
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.interpolate import interp1d  # noqa: E402

import sachs_jax as sj  # noqa: E402


def _maxrss_mb() -> float:
    """Peak resident set size in MB (macOS reports ru_maxrss in bytes)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _banner(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


# ==========================================================================
# EXP 1 -- analytic pin + caustic robustness
# ==========================================================================
def exp1_caustic() -> None:
    _banner("EXP 1  Analytic pin (D = sin(w*lam)/w) + caustic robustness")
    w = 1.0
    caustic = np.pi / w  # D -> 0, rho = -Ddot/D -> infinity here
    lam_max = 2.0 * np.pi / w + 0.3  # integrate THROUGH two caustics
    n_lam = 400
    ts = jnp.linspace(0.0, lam_max, n_lam)
    phi00 = jnp.full((n_lam,), -(w**2))  # constant Ricci focusing

    lam_eval = jnp.linspace(1e-3, lam_max, 600)
    ys = sj.solve_jacobi_scalar(ts, phi00, lam_eval, rtol=1e-10, atol=1e-12)
    D = np.asarray(ys[:, 0])
    le = np.asarray(lam_eval)
    D_true = np.sin(w * le) / w

    rel = np.abs(D - D_true) / (np.abs(D_true) + 1e-12)
    # restrict error metric to away-from-zero points (D_true ~ 0 at caustics)
    away = np.abs(D_true) > 1e-2
    print(f"  Jacobi (diffrax) vs analytic sin(w*lam)/w:")
    print(f"    max abs error over full range [0, {lam_max:.2f}] : {np.max(np.abs(D - D_true)):.3e}")
    print(f"    max rel error (away from caustics)              : {np.max(rel[away]):.3e}")
    print(f"    integrated THROUGH caustics at lam = {caustic:.3f}, {2*caustic:.3f}  -> finite: {np.all(np.isfinite(D))}")

    # Now the Riccati form, started from a regular point before the caustic.
    lam_start = 0.4 * caustic
    rho0 = float(-w / np.tan(w * lam_start))  # analytic rho = -w cot(w lam)
    ts2 = jnp.linspace(lam_start, lam_max, n_lam)
    phi2 = jnp.full((n_lam,), -(w**2))
    le2 = jnp.linspace(lam_start + 1e-3, lam_max, 600)
    try:
        rho = np.asarray(
            sj.solve_riccati_scalar(ts2, phi2, le2, rho0=rho0, rtol=1e-9, atol=1e-11)
        )
        finite_frac = np.mean(np.isfinite(rho) & (np.abs(rho) < 1e6))
        # where does it first explode?
        le2n = np.asarray(le2)
        bad = np.where(~(np.isfinite(rho) & (np.abs(rho) < 1e3)))[0]
        first_bad = le2n[bad[0]] if bad.size else None
        print(f"  Riccati (rho=-Ddot/D), same physics, started pre-caustic:")
        print(f"    fraction of points finite & |rho|<1e6 : {finite_frac:.2f}")
        print(f"    first blow-up near lam               : {first_bad}  (caustic at {caustic:.3f})")
    except Exception as e:  # diffrax may hit max_steps at the singularity
        print(f"  Riccati solve FAILED at the caustic: {type(e).__name__}: {e}")

    print("  --> Jacobi is exact and caustic-robust; Riccati diverges at the caustic.")


# ==========================================================================
# EXP 2 -- rough forcing: linear-interp (current) vs cubic control
# ==========================================================================
def _rough_phi00(le: np.ndarray, seed: int = 0) -> np.ndarray:
    """A deliberately rough (broadband, white-ish along lambda) focusing field."""
    rng = np.random.default_rng(seed)
    base = -1.0
    # sum of many Fourier modes with flat-ish amplitude -> rough, ~C0 when
    # sampled and linearly interpolated
    ks = np.arange(1, 60)
    amp = 0.4 / np.sqrt(ks)
    phase = rng.uniform(0, 2 * np.pi, ks.size)
    osc = (amp[None, :] * np.sin(ks[None, :] * le[:, None] + phase[None, :])).sum(1)
    return base + osc


def exp2_rough() -> None:
    _banner("EXP 2  Rough driving: linear-interp + scipy-RK45 (current) vs cubic control")
    lam_max = 2.0
    # a FINE reference grid + tight solve = ground truth
    n_fine = 8000
    ts_fine = np.linspace(0.0, lam_max, n_fine)
    phi_fine = _rough_phi00(ts_fine)
    lam_eval = jnp.linspace(1e-3, lam_max, 400)

    ys_ref = sj.solve_jacobi_scalar(
        jnp.asarray(ts_fine), jnp.asarray(phi_fine), lam_eval, rtol=1e-12, atol=1e-13
    )
    D_ref = np.asarray(ys_ref[:, 0])

    print(f"  {'n_grid':>7} | {'cubic-control (diffrax Jacobi)':>32} | {'linear-interp RK45 (scipy, current)':>36}")
    print("  " + "-" * 84)
    for n_grid in (64, 128, 256, 512):
        ts_g = np.linspace(0.0, lam_max, n_grid)
        phi_g = _rough_phi00(ts_g)  # same field, coarser sampling

        # (a) cubic control + diffrax Jacobi
        ys = sj.solve_jacobi_scalar(
            jnp.asarray(ts_g), jnp.asarray(phi_g), lam_eval, rtol=1e-9, atol=1e-11
        )
        D_cub = np.asarray(ys[:, 0])
        err_cub = np.max(np.abs(D_cub - D_ref)) / np.max(np.abs(D_ref))

        # (b) linear-interp forcing + scipy RK45 on the SAME linear Jacobi eq,
        #     mimicking sources.py interp1d(kind='linear') + solve_ivp.
        phi_lin = interp1d(ts_g, phi_g, kind="linear", fill_value="extrapolate")

        def rhs(t, y):
            return [y[1], float(phi_lin(t)) * y[0]]

        sol = solve_ivp(
            rhs, (0.0, lam_max), [0.0, 1.0], method="RK45",
            t_eval=np.asarray(lam_eval), rtol=1e-9, atol=1e-11,
        )
        D_lin = sol.y[0]
        err_lin = np.max(np.abs(D_lin - D_ref)) / np.max(np.abs(D_ref))

        print(f"  {n_grid:>7} | rel.err {err_cub:>10.3e}  ({'OK' if err_cub<1e-3 else 'WARN'})        "
              f"| rel.err {err_lin:>10.3e}  nfev={sol.nfev:<6} ({'OK' if err_lin<1e-3 else 'WARN'})")
    print("  --> linear interpolation caps accuracy at the grid spacing regardless of solver tol;")
    print("      cubic control converges far faster on the same samples.")


# ==========================================================================
# EXP 3 -- full-sky scaling + memory
# ==========================================================================
def exp3_memory() -> None:
    _banner("EXP 3  vmap over rays, float32, chunked  (memory bounded by chunk)")
    # NOTE: float32 production mode. Each ray gets an independent rough driving
    # realisation (Phi00, RePsi, ImPsi) on a shared lambda grid.
    n_lam = 256
    lam_max = 2.0
    ts = jnp.linspace(0.0, lam_max, n_lam).astype(jnp.float32)
    lam_source = jnp.asarray([lam_max], dtype=jnp.float32)

    def single_ray(drive):  # drive: (n_lam, 3) -> (8,) Jacobi state at source
        return sj.solve_jacobi_matrix(ts, drive, lam_source, rtol=1e-5, atol=1e-6)[0]

    n_rays = 1 << 14  # 16384 rays (~ nside=37 patch); small on purpose
    rng = np.random.default_rng(0)
    drive = np.zeros((n_rays, n_lam, 3), dtype=np.float32)
    le = np.asarray(ts)
    for j, amp in enumerate((1.0, 0.5, 0.5)):  # Phi00, RePsi, ImPsi scales
        ks = np.arange(1, 40)
        a = amp * 0.3 / np.sqrt(ks)
        ph = rng.uniform(0, 2 * np.pi, (n_rays, ks.size))
        drive[:, :, j] = (a[None, None, :] * np.sin(
            ks[None, None, :] * le[None, :, None] + ph[:, None, :]
        )).sum(-1) + (-1.0 if j == 0 else 0.0)
    drive = jnp.asarray(drive)

    bytes_per_ray_drive = n_lam * 3 * 4
    print(f"  n_rays={n_rays}, n_lam={n_lam}, dtype=float32")
    print(f"  driving array resident: {drive.nbytes/1e6:.1f} MB "
          f"({bytes_per_ray_drive} B/ray)")

    for chunk in (1024, 4096, 16384):
        gc.collect()
        rss0 = _maxrss_mb()
        t0 = time.perf_counter()
        out = sj.solve_rays_chunked(single_ray, drive, chunk=chunk)
        out.block_until_ready() if hasattr(out, "block_until_ready") else None
        dt = time.perf_counter() - t0
        rss1 = _maxrss_mb()
        # sanity: observables finite for all rays
        finite = bool(np.all(np.isfinite(np.asarray(out))))
        print(f"  chunk={chunk:>6}: time={dt:6.2f}s  peakRSS={rss1:7.1f}MB "
              f"(d+{rss1-rss0:5.1f})  all-finite={finite}")
        del out
    # projection
    npix_256 = 12 * 256 * 256
    full_drive_gb = npix_256 * n_lam * 3 * 4 / 1e9
    print(f"  PROJECTION nside=256 (npix={npix_256}): full driving array would be "
          f"{full_drive_gb:.1f} GB in float32")
    print(f"    -> stream in chunks of e.g. 65536 rays: working set ~"
          f"{65536*n_lam*3*4/1e6:.0f} MB; generate field per-chunk, never hold all of it.")


def main() -> None:
    print("JAX backend:", jax.default_backend(), "| devices:", jax.devices())
    exp1_caustic()
    exp2_rough()
    exp3_memory()
    print("\nDone.")


if __name__ == "__main__":
    main()
