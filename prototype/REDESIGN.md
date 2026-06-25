# sachsfield → full-sky Sachs ray-tracer: redesign blueprint

## Context

`sachsfield` is today a **brute-force stochastic Sachs solver** used to *validate*
the diagrammatic predictions in the STF_lensing draft (Appendix A "Workflow
validation": a direct Monte-Carlo of the stochastic Sachs equation reproduces the
analytic FF/FK channels). It is framed as an abstract scalar-field system
(`ẋ_a = F_abc x_b x_c + s_a`, saddle + per-pixel fluctuation, scipy `solve_ivp`).

The goal is to **reposition it as a numerical full-sky Sachs ray-tracer**: given
realisations of the Ricci-focusing field `Φ₀₀` and Weyl-shear field `Ψ₀`, integrate
the Sachs evolution per ray and read off lensing observables (κ, γ, rotation).
Three forces shape the redesign:

1. **Robustness to rough driving.** Driving fields can be near-discontinuous along λ.
2. **Non-Gaussian inputs.** Fields may be non-Gaussian (the FK / three-point channel),
   so they **cannot** be reduced to a drift+diffusion SDE — realisations must be
   integrated directly.
3. **Scale.** GPU, nside ≥ 256, large Monte-Carlo ensembles, bounded memory.

## Key decision: integrate the **linear Jacobi** form, not the Riccati form

The current `x_a` fields are NP optical scalars obeying the **Riccati** system
(`dρ/dλ = ρ² + σσ̄ − Φ₀₀`, `dσ/dλ = 2Re(ρ)σ + Ψ₀`). The equivalent **linear Jacobi**
form integrates the deformation matrix `𝒥` with `𝒥̈ = T(λ) 𝒥`, `𝒥(0)=0`, `𝒥̇(0)=I`,
where `T = [[Φ₀₀+ReΨ₀, ImΨ₀],[ImΨ₀, Φ₀₀−ReΨ₀]]`.

| | Riccati (current `x_a`) | Jacobi (proposed) |
|---|---|---|
| Linearity | nonlinear | **linear** in `𝒥` |
| Caustics | ρ → ∞ (blows up) | `D` crosses zero smoothly |
| Observer vertex | 1/λ singularity → needs `y₁=λχ₁` rescaling | regular IC, **no rescaling** |
| Noise entry | additive (SDE-friendly) | multiplicative |
| Rough forcing | nonlinear × near-singular = fragile | linear × bounded coeff = robust |

**Prototype evidence** (`prototype/bench.py`, validated against the draft's closed-form
`D = a·χ`, here `sin(ωλ)/ω`):

- Jacobi (diffrax) matches analytic to **3e-11**, integrates **through** caustics at
  λ=π, 2π. The Riccati form **fails at the first caustic** (solver max_steps at ρ→∞).
- On rough forcing, a **C2 cubic control** path is **7–30× more accurate** than the
  current `interp1d(kind='linear')` (C0) on identical samples, and the gap widens with
  grid refinement. The current scipy RK45 also spends growing `nfev` fighting the kinks.

## Architecture

```
┌─ FIELD GENERATION (numpy / healpy; pluggable, decoupled from the solver) ─┐
│  Produce Φ₀₀(λ, n), Ψ₀(λ, n) realisations on a (λ-grid × pixel) array.    │
│  Gaussian (current Cholesky path) OR non-Gaussian (external sims, lognormal,│
│  3-pt-seeded). The solver does not care how they were made.               │
└───────────────────────────────┬───────────────────────────────────────────┘
                                │  per-ray driving array (n_lam, 3)
                                ▼
┌─ SOLVER CORE (JAX / diffrax; GPU-ready, differentiable) ──────────────────┐
│  PRIMARY  controlled-ODE: ODETerm with a CubicInterpolation control of the │
│           driving field; Tsit5 (or Kvaerno5 if stiff). Handles ANY         │
│           realisation, Gaussian or not. This is the workhorse.             │
│  OPTIONAL additive-noise SDE: MultiTerm(ODETerm(drift), ControlTerm(...,    │
│           VirtualBrownianTree)) + ShARK. Gaussian-Markov ONLY; memory-light │
│           on-the-fly noise. Documented as a niche fast path, not default.   │
│                                                                            │
│  vmap over rays  ·  chunked over pixel-blocks  ·  SaveAt source plane only  │
└───────────────────────────────┬───────────────────────────────────────────┘
                                ▼
        observables_from_jacobi(𝒥, D_bg) → κ, γ1, γ2, ω  (per pixel)
```

### Memory plan (the "注意内存管理" requirement)
- **Peak scales with chunk size, not map size.** `solve_rays_chunked` vmaps a
  ray-block, pulls results to host, frees device buffers, repeats.
- **float32** in production (halves footprint vs the current float64), float64 only
  for validation.
- **Save only the source-plane Jacobi state** (`SaveAt(t1=True)`), not full
  trajectories — avoids the current `(n_fields, npix, N_t)` dense blow-up.
- **Generate the driving field per chunk** for large ensembles; never hold the full
  `(npix, n_lam, 3)` array (2.4 GB at nside=256, float32) if it can be streamed.
- For Gaussian ensembles, the SDE path with `VirtualBrownianTree` regenerates noise
  from a seed — no stored field at all.

### API repositioning (physical inputs/outputs)
- Inputs: `Φ₀₀` (real, spin-0) and `Ψ₀` (complex, spin-2) as realisations or callables.
- Output: Jacobi map → `κ = 1 − ½tr(A)`, shear `γ1,γ2`, rotation `ω`, with `A = 𝒥/D_bg`.
- Keep the saddle/`x_a` solver intact as a **validation reference** so the draft's
  crosscheck notebooks keep running unchanged; the new ray-tracer lives alongside.

## Migration path (hybrid, prototype-validated)
1. ✅ **Prototype** (`prototype/`): diffrax Jacobi core + benchmark vs scipy. DONE —
   confirms diffrax works, Jacobi is exact & caustic-robust, cubic ≫ linear, vmap+chunk runs.
2. ✅ **Package the core** → `sachsray/` (scalar + 2×2 matrix Jacobi, cubic-control driving,
   chunked vmap driver, observables). 8/8 unit tests.
3. ✅ **Field-generation adapter** (`sachsray/fields.py`): `driving_from_components`
   (generic, non-Gaussian-ready) + `driving_from_source` (duck-typed bridge to
   `sachsfield.FullSkySource`/`FlatSkySource`). Assembles `Φ₀₀=Φ̄₀₀+δΦ₀₀`. End-to-end
   test generates a real correlated full-sky realisation (healpy) → κ/γ/ω maps.
4. ◑ **Cross-validate**: ✅ Riccati↔Jacobi convention checks; ✅ **Born-limit** — the full
   pipeline reproduces `κ = −∫(λ_s−λ')λ'/λ_s · δΦ₀₀ dλ'` to <5% (`tests/test_fields.py`).
   Pending: end-to-end vs the existing `x_a` MC two-point and the draft's FF/FK channels.
5. ⬜ **Scale test** on GPU at nside ≥ 256 with chunked streaming; pin peak memory
   (CPU-only here — code is device-agnostic/float32-ready).

## Open questions to resolve during step 2–4
- Exact sign/normalisation of the Weyl term in `T` and the `A = 𝒥/D_bg` observable
  dictionary — pin against the draft's 3×3 Jacobi-map appendix (`append: jacobian blocks`).
- Whether the redshift/time (3×3) block is needed, or the 2×2 transverse block suffices
  for the target observables.
- Stiff vs non-stiff solver choice under realistic cosmological `Φ₀₀` amplitudes.
```
```
## Status (built + validated this session)

`sachsray/` package implemented to the draft's published convention and validated:
- `physics.py` — conventions (F-tensor, tidal matrix, optical-scalar extraction, observables).
- `solvers.py` — diffrax integrators (Jacobi scalar/matrix, Riccati), cubic control.
- `raytrace.py` — `DrivingField` + chunked `vmap` `trace_rays` (kappa/gamma/omega maps).
- `tests/test_sachsray.py` — **8/8 pass**: analytic pins (D=λ, D=sin(ωλ)/ω to <1e-9),
  caustic pass-through, observable signs, chunked ray-tracer smoke, and the
  **Riccati↔Jacobi cross-validation** that pins the Weyl-shear sign to <1e-6.

### The `x1 = 2*theta` convention (NOT a bug in the validation — verified)
The old `coefficients.py` uses F_111 = -1/2, i.e. it integrates **x1 = 2*theta**
(`prototype/crosscheck_old_new.py`: theta_code = 2*theta_phys). This is a deliberate,
consistent field redefinition, and the draft's validation is **clean**:

`twopoint_crosscheck.ipynb` (the source of Fig. val mc) uses F_111 = -0.5 on BOTH the
analytic sft-wick side (`F_code`) and the MC side (`coefficients.py` via FluctuationSolver),
with a synthetic background `sbar_1` via `solve_singular_theta` (which bakes in the same
x1=2theta convention, theta->2/lam). The MC drift `A = -theta_code I = -2 theta_phys I`
and the analytic propagator `R = exp(-int theta_code) = (Dbar'/Dbar)^2` BOTH match the
draft's published formulas exactly -- the factor of 2 is precisely the squared-propagator
factor, correctly placed. A self-consistency check with a shared convention is unaffected
by that convention. **perFLRW is not used in any validation notebook.**

Two real-but-minor, validation-independent items remain:
1. *Presentational*: the paper Table tab:Fabc writes F_111 = -1 (physical theta), but the
   code + all notebooks use F_111 = -0.5 (x1=2theta). Equivalent physics; worth one
   reconciling sentence in the paper.
2. *perFLRW sign+factor (SETTLED)*: `perFLRW.background_phi00` returns
   `+4piG(rho+P)(E/a)^2 > 0`, but the paper's eq:driving bg gives
   `Phi00^(0) = -(E/a)^2 (H^2-H') = -4piG(rho+P)/a^2 < 0` (reproduced from R^(0)_munu k^mu k^nu;
   focusing => Phi00<0). So `background_phi00` has the WRONG SIGN. Moreover the old
   x1=2theta saddle needs `s1 = 2*Phi00` (from u''=1/2 s1 u; numerically: s1=2Phi00
   reproduces D=a*chi to 1.7e-3, perFLRW's value gives 12% error). Net: perFLRW is off by
   **-2x** for the old saddle. Tests: `tests/test_perflrw_background.py` (pyccl-free toy +
   pyccl-gated sign assert) and `prototype/crosscheck_saddle_sign.py`. Affects only
   absolute predictions via the perFLRW path (demo/evolution_cosmo), NOT the MC validation.
   **FIX APPLIED (2026-06-25):** `background_phi00` now returns the negative value, and
   `sbar_func` now sets `out[0] = 2*Phi00` (x1=2theta convention) -> net saddle source
   s1 = -8piG(rho+P)/a^2 = 2*Phi00 (correct). `sachsray` (theta convention, s1=Phi00)
   needs only the sign and reproduces D=a*chi to <1e-6.
   ALSO FIXED: the fluctuation-tracer prefactor (now `_phi00_delta_prefactor`, used by
   `_make_phi00_tracer`) was +4piG, corrected to NEGATIVE. From first principles
   (eq:Phi00 scalar, Phi=Psi): the two-derivative terms reduce to Phi00 ~ -(E/a)^2 grad^2 Phi,
   and Poisson grad^2 Phi = 4piG a^2 rho_m delta gives Phi00 ~ -4piG rho_m (E/a)^2 delta < 0
   (= -A delta_m, cosmology.tex:438). Auto-C_l and z-z cross-C_l are UNCHANGED (prefactor
   enters squared / twice); only sign-sensitive uses (Phi00xPsi0, three-point/FK) are now
   physically correct. The FK three-point itself is built by canoes, not perFLRW.

`sachsray` sidesteps all of this by using the physical theta directly (F_111=-1, s1=Phi00).

## Next (optional)
- GPU run at nside≥256 with chunked streaming; pin peak memory.
- Field-generation adapter wrapping `sources.py`/`perFLRW` output into `DrivingField`
  (with the corrected `s1=2*Phi00` or the clean theta-convention).
- Additive-noise SDE path (diffrax ShARK) as the Gaussian-only fast mode.

## Files
- `sachsray/` — the new package (physics, solvers, raytrace).
- `tests/test_sachsray.py` — validation suite.
- `prototype/sachs_jax.py` — original diffrax prototype (Jacobi/Riccati, bench).
- `prototype/bench.py` — robustness/memory benchmark experiments.
- `prototype/crosscheck_old_new.py` — old↔new convention cross-check (factor-2 demo).
