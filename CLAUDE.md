# CLAUDE.md — sachsfield development context

## What this project is

`sachsfield` is a Python package for numerically evolving coupled scalar field systems from the Sachs optical equations (gravitational lensing). It solves the ODE `x_dot_a = F_{abc} x_b x_c + s_a` for 3 or 4 real scalar fields on the sphere (or a flat-sky patch), decomposed into a direction-independent saddle point and direction-dependent fluctuations.

## Architecture

```
sachsfield/
  coefficients.py   -> quadratic(), M_matrix() — hardcoded F tensor, vectorized over pixels
  saddle.py         -> SaddlePointSolver, SaddlePointResult — rescaled-variable ODE
  sources.py        -> FullSkySource (healpy), FlatSkySource (2D FFT) — lambda-correlated Gaussian fields
  fluctuations.py   -> FluctuationSolver, FluctuationResult — pixel-parallel ODE
  solver.py         -> SachsFieldSolver, FullResult — high-level orchestrator
  utils.py          -> animate_evolution, pixel_pdf, angular_power_spectrum, summary_statistics, evolution_summary
```

## Critical implementation details

### Rescaled variable for saddle point
The saddle-point solver does NOT integrate `chi_1` directly. It integrates `y_1 = lambda * chi_1`, which stays O(1) near the singular point `lambda -> 0`. The rescaled ODE is:
```
dy_1/dlam = y_1(2 - y_1) / (2*lam) + lam * (remaining terms)
dy_a/dlam = -y_1 * y_a / lam + sbar_a    (a >= 2)
```
The `SaddlePointResult` stores and exposes chi (original variables), with conversion `chi_1 = y_1 / lam` done internally. Dense output interpolation is also in rescaled coords, converted on the fly.

### Asymptotic consistency issue
With `F_111 = -1/2`, the self-consistent asymptotic is `chi_1 = 2/lambda` (y1_init=2), NOT `-1/lambda` as originally stated by the user. This was verified analytically: `d/dlam(2/lam) = -2/lam^2 = -1/2 * (2/lam)^2`. The `y1_init` parameter exists so the user can adjust if their F convention differs. This is an **unresolved ambiguity** that may need revisiting.

### Source generation: per-mode Cholesky
Lambda correlations are implemented by building a covariance matrix `K[i,j] = sqrt(C_ell(lam_i)) * sqrt(C_ell(lam_j)) * rho(|lam_i - lam_j|)` for each ell mode, then Cholesky-decomposing and multiplying by white noise. This is exact but has cost O(lmax * n_lam^3). Pre-computing the rho matrix outside the ell loop avoids redundant work.

### Fluctuation ODE size
The state vector is `(n_fields * npix)`. For nside=64, that's ~150k equations. The RHS is vectorized: `M @ xi` is a `(3,3) @ (3, npix)` matmul, `quadratic(xi)` uses element-wise numpy ops. No per-pixel Python loops.

## perFLRW module

```
perFLRW/
  cosmology.py  -> FLRWCosmology — wraps PyCCL for Sachs source generation
```

`FLRWCosmology` provides:
- `lambda_of_z(z)`, `z_of_lambda(lam)` — affine parameter ↔ redshift via precomputed interpolation tables. Uses `dλ/dz = -1/((1+z)² H(z) E_co)`.
- `background_phi00(z)` — Ricci focusing term `Φ₀₀ = 4πG (ρ+P) (E_co/a)²`. Uses standard Sachs convention (4πG, not 8πG).
- `sachs_cls(z, lmax, delta_z)` — angular power spectra via PyCCL custom tracer with narrow Gaussian selection. `C_l^{Ψ₀}` derived from `C_l^{Φ₀₀}` via spin-2 correction `ℓ(ℓ+1)/((ℓ+2)(ℓ-1))`.
- `sachs_cls_cross(z_array, lmax, delta_z, field)` — full z-z cross-spectrum matrix `C_ℓ(z_i, z_j)`. Builds a PyCCL tracer per shell and computes all N(N+1)/2 unique pairs. Cost is O(N²) in PyCCL `angular_cl` calls, so keep `len(z_array)` ≤ ~50.
- `sbar_func(n_fields)`, `cl_func(z_array, lmax)` — return callables compatible with `sachsfield.SachsFieldSolver` and `sachsfield.FullSkySource`.

Dark energy EOS uses CPL parameterization: `w(a) = w0 + wa*(1-a)`, read from cosmo params. Field 3 (Ψ₀ shear component 2) has same C_l as field 2 by statistical isotropy.

### Cross-redshift C_l structure
`sachs_cls_cross` returns `(ell, cl_matrix)` where `cl_matrix[i, j, k]` = `C_{ell[k]}(z_i, z_j)`. In Limber approximation, shells at different comoving distances only correlate if their Gaussian selection windows overlap — so the matrix is sharply diagonal for `delta_z ≪ Δz_grid`. The correlation width in z shrinks at higher ℓ (smaller physical scales have shorter correlation lengths along the line of sight).

## Dependencies
- `numpy`, `scipy` (solve_ivp, interp1d) — core (sachsfield)
- `healpy` — full-sky source generation and analysis only (sachsfield)
- `pyccl` — required by perFLRW only; sachsfield itself has no pyccl dependency

## Known limitations / future work
- Source generation is slow for large lmax due to per-ell Cholesky (could batch ell values with identical C_ell profiles)
- No JAX/GPU backend yet — numpy only
- Flat-sky source normalization (`C_ell / dx^2`) should be validated against analytic expectations
- The `y1_init` ambiguity needs resolution with the user's actual physical conventions
- No cross-correlations between source fields (only C_l^{11}, C_l^{22}, C_l^{33} are supported)
- No adaptive lambda sampling for source generation; uniform grid only

## Testing
Run from `src-field/`:
```python
import sachsfield as sf
import numpy as np

# Saddle point verification (zero source)
sp = sf.SaddlePointSolver(lambda l: np.zeros(3), (-1e-4, -5.0), n_fields=3, y1_init=2.0, rtol=1e-10, atol=1e-12)
r = sp.solve()
assert np.allclose(r.chi[0], 2.0 / r.lam, rtol=1e-8)  # chi_1 = 2/lam
assert np.allclose(r.chi[1], 0, atol=1e-12)              # chi_2 = 0
```

## File locations
- Package: `sachsfield/`
- Cosmology module: `perFLRW/`
- Demo notebook: `notebooks/demo.ipynb` (sections 1–8: sachsfield, section 9: perFLRW pipeline + z-z C_l visualization)
- Existing exploratory notebook: `test.ipynb` (early pyccl trials, partially superseded by perFLRW)
- README: `README.md`
