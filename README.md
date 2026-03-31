# sachsfield

A Python package for evolving coupled scalar field systems arising from the Sachs optical equations in gravitational lensing.

## Physics Overview

### The dynamical system

Four real scalar fields `x_a(n, lambda)` (a = 1,2,3,4) evolve along an affine parameter `lambda` on directions `n` on the sky:

```
x_dot_a = F_{abc} x_b x_c + s_a
```

where dot denotes `d/d_lambda`, repeated Latin indices are summed over {1,2,3,4}, and `s_a` is the source (driving) term with `s_4 = 0` always.

### Coupling tensor F

The nonzero entries of `F_{abc}` are:

| abc | 111  | 122 | 133 | 144 | 212 | 313 | 414 |
|-----|------|-----|-----|-----|-----|-----|-----|
| F   | -1/2 | -2  | -2  | 2   | -1  | -1  | -1  |

### Saddle-point + fluctuation decomposition

The system is decomposed as `x_a = chi_a + xi_a`:

**Saddle point** `chi_a(lambda)` (direction-independent):
```
chi_dot_a = F_{abc} chi_b chi_c + sbar_a(lambda)
```
where `sbar_a` is the sky-averaged source. Initial conditions: `chi_1 -> y1_init / lambda` as `lambda -> 0^-`, `chi_{2,3,4} = 0`.

> **Note on the asymptotic:** With `F_111 = -1/2`, the self-consistent asymptotic solution of the homogeneous equation `chi_dot = -1/2 chi^2` is `chi_1 = 2/lambda` (i.e., `y1_init = 2`). Verify: `d/dlam(2/lam) = -2/lam^2` and `-1/2 * (2/lam)^2 = -2/lam^2`. The parameter `y1_init` is adjustable if a different F convention is used.

**Fluctuations** `xi_a(lambda, n)` (direction-dependent):
```
xi_dot_a = M_{ab}(lambda) xi_b + F_{abc} xi_b xi_c + delta_s_a(lambda, n)
```
where `M_{ab} = (F_{abc} + F_{acb}) chi_c` is the linearization matrix and `delta_s_a = s_a - sbar_a`. Initial condition: `xi_a = 0`.

The linearization matrix has the explicit form:
```
        [ -c1   -4c2  -4c3   4c4 ]
M(chi)= [ -c2   -c1    0      0  ]     (4-field)
        [ -c3    0    -c1     0  ]
        [ -c4    0     0     -c1 ]
```

### The 4th field

Since `s_4 = 0`, `chi_4(0) = 0`, and the fluctuation equation gives `xi_dot_4 = -(chi_1 + xi_1) xi_4`, the 4th field is identically zero. The package supports `n_fields=3` to drop it entirely (25% fewer ODE equations).

## Installation

```bash
cd /path/to/src-field
pip install -e .
```

Dependencies: `numpy`, `scipy`, `healpy`.

## Package Structure

```
sachsfield/
  __init__.py          # Public API exports
  coefficients.py      # F tensor, M matrix, quadratic nonlinearity
  saddle.py            # Saddle-point ODE solver (rescaled variables)
  sources.py           # Source field generation (full-sky + flat-sky)
  fluctuations.py      # Fluctuation ODE solver on pixel arrays
  solver.py            # High-level orchestrator
```

## Module Reference

### `coefficients.py`

```python
quadratic(x, n_fields=4)
```
Compute `F_{abc} x_b x_c` for all `a`. Input `x` has shape `(n_fields,)` or `(n_fields, npix)`. Fully vectorized over pixels.

```python
M_matrix(chi, n_fields=4)
```
Return the `(n_fields, n_fields)` linearization matrix `M_{ab} = (F_{abc} + F_{acb}) chi_c`.

### `saddle.py`

#### `SaddlePointSolver`

```python
solver = SaddlePointSolver(
    sbar_func,          # callable(lam) -> ndarray(n_fields,)
    lam_span,           # (lam_start, lam_end), both negative, lam_end more negative
    n_fields=3,         # 3 or 4
    y1_init=2.0,        # rescaled IC: y1 = lam * chi_1 at lam_start
    method='Radau',     # scipy solve_ivp method
    dense_output=True,
    rtol=1e-8, atol=1e-10,  # passed to solve_ivp
)
result = solver.solve(t_eval=None)
```

**Rescaled variable trick:** Internally integrates `y_1 = lambda * chi_1` instead of `chi_1` directly. This keeps the state vector O(1) near the singular point `lambda -> 0`, avoiding numerical stiffness. The rescaled ODE is:
```
dy_1/dlam = y_1(2 - y_1) / (2 lam) + lam * (F_{1bc>1} chi_b chi_c + sbar_1)
dy_a/dlam = -y_1 y_a / lam + sbar_a     (a >= 2)
```
with IC `y_1 = y1_init`, `y_{rest} = 0`. The fixed point `y_1 = 2` (for `F_111 = -1/2`) is a stable equilibrium of the rescaled equation, so integration is well-conditioned.

#### `SaddlePointResult`

```python
result.lam              # ndarray, shape (N,)
result.chi              # ndarray, shape (n_fields, N) -- in original chi coords
result(lam)             # interpolate chi at arbitrary lambda (dense output)
result.M(lam)           # linearization matrix at given lambda
```

### `sources.py`

Two backends with identical interface, differing only in sky geometry.

#### `FullSkySource` (healpy spherical harmonics)

```python
source = FullSkySource(
    cl_func,            # callable(lam) -> (cl_11, cl_22, cl_33), arrays indexed by ell
    lam_samples,        # 1D array of lambda values for source generation
    nside,              # healpy nside
    corr_func=None,     # callable(|dlam|) -> rho in [0,1], or None for uncorrelated
    seed=None,
    n_fields=3,
)
source.generate()       # pre-generate all maps (call explicitly or auto on first use)
```

#### `FlatSkySource` (2D FFT rectangular patch)

```python
source = FlatSkySource(
    cl_func,            # same as FullSkySource
    lam_samples,
    nx, ny,             # grid dimensions
    dx,                 # pixel angular size in radians
    corr_func=None,
    seed=None,
    n_fields=3,
)
source.generate()
```

#### Source generation algorithm

For each source field `a in {1,2,3}` and each harmonic mode `(ell, m)` (or Fourier mode `k`):

1. Build the `(n_lam, n_lam)` covariance matrix:
   ```
   K[i,j] = sqrt(C_ell(lam_i)) * sqrt(C_ell(lam_j)) * rho(|lam_i - lam_j|)
   ```
2. Cholesky decompose `K = L L^T`.
3. Draw iid Gaussians `z ~ N(0,1)` and compute correlated coefficients `a_lm = L z`.
4. Inverse SHT (or inverse FFT) to get maps at each lambda sample.

This ensures:
- Correct angular power spectrum `C_ell(lambda)` at each lambda.
- Prescribed correlation structure `rho(|Delta lambda|)` between different lambda values.
- Field 4 source is always zero.

#### Interpolators

```python
sbar = source.sbar_interpolator()      # callable(lam) -> ndarray(n_fields,)
delta_s = source.delta_s_interpolator() # callable(lam) -> ndarray(n_fields, npix)
```

Both use `scipy.interpolate.interp1d` (linear) along the lambda axis.

### `fluctuations.py`

#### `FluctuationSolver`

```python
solver = FluctuationSolver(
    saddle_result,      # SaddlePointResult
    delta_s_func,       # callable(lam) -> ndarray(n_fields, npix)
    npix,               # number of pixels
    n_fields=3,
    linear_only=False,  # drop quadratic xi*xi term if True
    lam_span=None,      # override; default from saddle_result
    method='RK45',
    rtol=1e-6, atol=1e-8,
)
result = solver.solve(t_eval=lam_array)
```

The state vector has `n_fields * npix` components. The RHS is fully vectorized:
- `M @ xi`: `(n_fields, n_fields) @ (n_fields, npix)` matrix multiply, uniform across pixels.
- `quadratic(xi)`: element-wise arithmetic, vectorized over npix.
- `delta_s(lam)`: interpolated source.

#### `FluctuationResult`

```python
result.lam                                      # shape (N_out,)
result.xi                                       # shape (N_out, n_fields, npix)
result.get_maps(lam_index, field=None)           # extract maps
result.power_spectra_fullsky(lam_index, nside)   # list of C_ell per field
result.power_spectra_flatsky(lam_index, nx, ny, dx)  # (ell_centers, [C_ell per field])
```

### `solver.py`

#### `SachsFieldSolver` (high-level)

```python
solver = SachsFieldSolver(
    source,             # FullSkySource or FlatSkySource
    lam_span,           # (lam_start, lam_end)
    sbar_func=None,     # override; default from source.sbar_interpolator()
    n_fields=3,
    linear_only=False,
    saddle_kwargs={},   # extra args for SaddlePointSolver
    fluct_kwargs={},    # extra args for FluctuationSolver
)
result = solver.solve(t_eval=lam_array)
```

#### `FullResult`

```python
result.saddle                   # SaddlePointResult
result.fluctuation              # FluctuationResult
result.total_field(lam_index)   # chi + xi, shape (n_fields, npix)
```

### `utils.py`

Visualization and summary statistics.

#### Animation

```python
anim = sf.animate_evolution(
    result,
    mode='fullsky',         # or 'flatsky'
    nside=32,               # for fullsky
    # nx=64, ny=64, dx=dx,  # for flatsky
    fields=[0, 1, 2],       # which fields to show
    show_saddle=False,       # True for total field, False for xi only
    interval=200,            # ms between frames
)
# Display in notebook:
from IPython.display import HTML
HTML(anim.to_jshtml())
# Or save:
anim.save('evolution.mp4', writer='ffmpeg', dpi=150)
```

#### Pixel PDF

```python
maps = result.fluctuation.xi[-1]  # shape (n_fields, npix)
hist_data, ax = sf.pixel_pdf(maps, n_bins=50, log_counts=False)
# hist_data[i] = (bin_centers, counts, bin_edges) for field i
```

#### Angular Power Spectrum

```python
# Full-sky
ells, cls = sf.angular_power_spectrum(maps, mode='fullsky', nside=32)

# Flat-sky
ells, cls = sf.angular_power_spectrum(maps, mode='flatsky', nx=64, ny=64, dx=dx)

# Plot
sf.plot_power_spectra(ells, cls)
```

#### Combined Summary

```python
fig, stats = sf.summary_statistics(
    result, lam_index=-1,
    mode='fullsky', nside=32,
    show_saddle=False,
)
# stats contains: 'mean', 'std', 'skew', 'kurtosis', 'ells', 'cls', 'pdf'
```

#### Evolution of Statistics

```python
fig, time_stats = sf.evolution_summary(
    result, mode='fullsky', nside=32, show_saddle=False,
)
# time_stats contains: 'lam', 'rms', 'skew', 'kurtosis' (each a dict per field)
```

## Usage Examples

### Minimal: saddle point only

```python
import numpy as np
import sachsfield as sf

def sbar(lam):
    return np.array([0.1, 0.0, 0.0])

solver = sf.SaddlePointSolver(sbar, lam_span=(-1e-4, -5.0), n_fields=3)
result = solver.solve()

# Interpolate at any lambda
chi = result(-2.0)  # ndarray shape (3,)
```

### Full evolution with sources

```python
import numpy as np
import sachsfield as sf

# Power spectra as function of lambda
def cl_func(lam):
    ell = np.arange(97)
    cl = 1e-4 / (ell + 1)**2
    cl[0] = 0
    return cl, 0.5*cl, 0.3*cl

# Generate lambda-correlated source fields
source = sf.FullSkySource(
    cl_func,
    lam_samples=np.linspace(-0.01, -3.0, 30),
    nside=32,
    corr_func=lambda dl: np.exp(-dl**2 / 0.5),
    seed=42,
    n_fields=3,
)

# Solve everything
solver = sf.SachsFieldSolver(source, lam_span=(-1e-3, -3.0), n_fields=3)
t_eval = np.linspace(-1e-3, -3.0, 20)
result = solver.solve(t_eval=t_eval)

# Get total field x = chi + xi
total = result.total_field(-1)  # shape (3, npix), at final lambda
```

### Flat-sky patch

```python
source = sf.FlatSkySource(
    cl_func,
    lam_samples=np.linspace(-0.01, -3.0, 30),
    nx=128, ny=128,
    dx=np.radians(10.0) / 128,  # 10-degree patch
    seed=42,
    n_fields=3,
)
solver = sf.SachsFieldSolver(source, lam_span=(-1e-3, -3.0), n_fields=3)
result = solver.solve(t_eval=t_eval)
```

## Performance

| nside | npix   | n_fields | ODE size   | Typical wall time (linear, 20 output steps) |
|-------|--------|----------|------------|----------------------------------------------|
| 8     | 768    | 3        | 2,304      | < 1 s                                        |
| 32    | 12,288 | 3        | 36,864     | ~10 s                                        |
| 64    | 49,152 | 3        | 147,456    | ~minutes                                     |

The source generation step (Cholesky per ell mode) is also significant for large `lmax` and many lambda samples.

## Conventions

- **Indexing:** Fields are 0-indexed internally: `a in {0, 1, 2}` for `n_fields=3` or `{0, 1, 2, 3}` for `n_fields=4`. This maps to the physics notation `x_1, x_2, x_3, x_4`.
- **Lambda direction:** Evolution runs from `lam_start` (near 0, negative) to `lam_end` (more negative). Both values are negative.
- **Units:** The package is unit-agnostic. All quantities are in whatever units the user provides for `sbar_func`, `cl_func`, and `lam_span`.
- **Source normalization:** `cl_func(lam)` returns angular power spectra `C_ell^{aa}` with standard normalization (variance = sum of `(2ell+1) C_ell / (4 pi)`). The flat-sky source uses `P(k) = C_ell / dx^2` internally.

---

## perFLRW — Cosmological Source Generation

The `perFLRW` module wraps [PyCCL](https://github.com/LSSTDESC/CCL) to derive physical source terms for the Sachs equations from an FLRW cosmology.

### Installation

Requires `pyccl` (not needed for `sachsfield` core):
```bash
conda install -c conda-forge pyccl
```

### Quick Start

```python
from perFLRW import FLRWCosmology
import numpy as np

flrw = FLRWCosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, sigma8=0.8, n_s=0.96)
```

### `FLRWCosmology` API

#### Constructor

```python
FLRWCosmology(Omega_c, Omega_b, h, sigma8, n_s, E_co=1.0,
              z_max=10.0, n_table=2000, **ccl_kwargs)

# Or wrap an existing pyccl.Cosmology:
FLRWCosmology.from_ccl(cosmo, E_co=1.0)
```

- `E_co`: comoving photon energy (constant along null geodesic). Appears in both the affine parameter mapping and Phi_00.
- `**ccl_kwargs`: passed to `pyccl.Cosmology` (e.g., `w0`, `wa`, `transfer_function`).

#### Affine parameter mapping

```python
lam = flrw.lambda_of_z(z)     # lambda(z), negative for z > 0
z   = flrw.z_of_lambda(lam)   # z(lambda), inverse
```

Computed from `dlambda/dz = -1 / ((1+z)^2 H(z) E_co)` via cumulative trapezoidal integration on a dense grid, stored as cubic interpolation tables.

#### Background Phi_00

```python
phi00 = flrw.background_phi00(z)              # Phi_00(z), units 1/Mpc^2
phi00 = flrw.background_phi00_of_lambda(lam)  # same via lambda
```

Uses the standard Sachs convention:
```
Phi_00 = 4 pi G (rho + P)_total (E_co / a)^2
```

Components: matter (P=0), radiation (P=rho/3), dark energy (P=w(a)*rho, CPL parameterization).

#### Angular power spectra

```python
ell, cl_phi00, cl_psi0 = flrw.sachs_cls(z, lmax=2000, delta_z=0.005)
ell, cl_phi00, cl_psi0 = flrw.sachs_cls_of_lambda(lam, lmax=2000)
```

Uses a custom PyCCL tracer with a narrow Gaussian selection function. The Weyl shear spectrum is derived via the spin-2 correction: `C_l^{Psi_0} = C_l^{Phi_00} * l(l+1) / ((l+2)(l-1))`.

#### Cross-redshift power spectrum matrix

```python
ell, cl_matrix = flrw.sachs_cls_cross(z_array, lmax=200, delta_z=0.01, field='phi00')
# cl_matrix.shape = (n_z, n_z, n_ell)
# cl_matrix[i, j, k] = C_{ell[k]}(z_i, z_j)  (symmetric)
```

Computes the full cross-spectrum between every pair of redshift shells. Useful for:
- Visualizing the z-z correlation structure (how much LSS is shared between shells)
- Constructing the covariance matrix for correlated source generation
- Understanding the "noisiness" of the Limber approximation

The `field` parameter selects `'phi00'` (Ricci) or `'psi0'` (Weyl shear).

```python
# Correlation coefficient (normalize out amplitude)
diag = np.diag(cl_matrix[:, :, k])
r = cl_matrix[:, :, k] / np.sqrt(np.outer(diag, diag))
```

#### sachsfield-compatible callables

```python
# Background source for saddle point
sbar = flrw.sbar_func(n_fields=3)   # sbar(lam) -> ndarray(3,)

# Power spectra for source generation
cl_func = flrw.cl_func(z_array=np.linspace(0.01, 2, 20), lmax=96)
# cl_func(lam) -> (cl_11, cl_22, cl_33)
```

These plug directly into `sachsfield`:
```python
import sachsfield as sf

source = sf.FullSkySource(cl_func, lam_samples, nside, seed=42, n_fields=3)
solver = sf.SachsFieldSolver(source, lam_span=(lam_start, lam_end),
                              sbar_func=sbar, n_fields=3)
result = solver.solve(t_eval=t_eval)
```
