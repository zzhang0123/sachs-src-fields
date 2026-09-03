
# N1 -- direct numerical test of the "no FK B-mode" theorem with sachsray

Agent N1, 2026-09-03. Code and outputs: `/Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/n1_sachsray/` (`n1_theorem_sachsray.py`, `n1_twopcf_check.py`, `outputs/`). Nothing outside this directory was modified; the paper directory, SFT-WL-B and canoes were only read. Nothing committed.

## 1. What was tested

**Theorem.** Expand the shear in powers of the driving field, `gamma = g1 + g2 + g3 + ...` (`gn` homogeneous of degree n). If the linear response `g1` is pure E for every realisation, `g1_B = 0` identically, so the order-3 (one-cumulant, FK-type) B-mode cross term `<g2_B g1_B>` vanishes identically for ANY statistics of the driving field. The first B-mode is 4th order: `<g2_B g2_B>` (FF/post-Born-like) plus the connected four-point piece. Equivalently `xi_+^{(2,1)}` and `xi_-^{(2,1)}` are the `d^l_{22}` and `d^l_{2,-2}` transforms of ONE spectrum, and the `gamma^4` small-angle behaviour of `xi_-` is the kernel, not a B-mode.

**Claim under test** (`sections/conclusion.tex` l.49-81, Fig. `cl_EB_polarization`): FK feeds both polarisations, `C_BB^FK/C_EE^FK` from 0.95 (`l=60`) to 0.42 (`l=1500`).

**Method.** Exact nonlinear Sachs/Jacobi system (`sachsray`, `J'' = T J`, float64, rtol 1e-9, atol 1e-12) on the full HEALPix sky for a scalar-sourced driving field; perturbative orders isolated numerically by scaling the field with `eps = +1, -1, +1/2, -1/2` (same realisation) and Richardson extraction of `g1..g4`; E/B by `healpy.map2alm_spin` (3 iterations); order-(a,b) cross spectra by `hp.alm2cl`. Repeated for a Gaussian potential and for a skewed (non-Gaussian) potential -- the FK-relevant case, where `<g2 g1>` is sourced by the three-point cumulant of the driving field exactly as in the paper's FK diagram (unperturbed path, no deflection; all F-vertex channels and all K3 spin channels TTT/TTP/TPP/PPP are automatically included by the solver). Independent of the paper's vertex tables, zeta channels and 2PCF->C_l transform.

## 2. Setup and conventions

* Potential `psi(lambda, n)`: Gaussian, `C_l^psi = 4A/(l(l+1))^3 exp(-(l/nside)^6)` for `2 <= l <= L_cut = min(1.4 nside, lmax/2)` (89 at nside 64, 179 at nside 128; quadratic products then alias-free at `lmax = 3 nside - 1`), AR(1)-correlated in lambda (`lc = 0.25`, `lambda in [0,2]`, 96 samples); amplitude set so the Born convergence has rms 0.030 (second-order shear ~ 1e-3 of first order in power).
* Tidal matrix = screen Hessian of psi: with sachsray's `T = [[Phi00+W1, W2],[W2, Phi00-W1]]`, `Phi00 = tr T/2 = (1/2) Lap psi` (alm x `-l(l+1)/2`) and `W1 + iW2 = (1/2) eth eth psi`, pure E: `alm_E = -(1/2) sqrt((l+2)!/(l-2)!) psi_lm`, `alm_B = 0`. The multiplier ratio `l(l+1)` vs `sqrt(l(l+1)(l(l+1)-2))` is exactly the paper's `L^2` vs `sqrt(L^2(L^2-2))`.
* Vacuum background (`phi00_bg = 0`, `D_bg = lambda`), source at `lambda_s = 2`. Born: `kappa1 = -int K Phi00`, `gamma1 = -int K (W1+iW2)`, `K = (lambda_s-lambda)lambda/lambda_s`, used as an independent check of `g1`.
* NG variant: `psi -> psi + q (psi^2 - <psi^2>)` per shell, `q sigma_psi = 0.15` (measured skewness of psi 0.75-1.14, of Phi00 -0.8..-1.0), low-passed back to `l <= L_cut`; `Phi00`, `Psi0` rebuilt from the deformed potential, so `Psi0` is still exactly `(1/2) eth eth psi_NG` (pure E for every realisation).
* Order separation: `odd = (g(+1)-g(-1))/2 = g1+g3+...`, `even = (g(+1)+g(-1))/2 = g2+g4+...`; with the `+-1/2` traces `g1 = (8 odd_h - odd_1)/3`, `g3 = 4(odd_1 - 2 odd_h)/3`, `g2 = (16 even_h - even_1)/3`, `g4 = 4(even_1 - 4 even_h)/3` (each exact to two orders higher). For the NG field the orders are powers of the NG field itself, so no mixing: `<g2 g1>` is the genuine one-cumulant (FK-type) term. (Even if one used only the eps = +-1 pair, the odd map's B-mode is still pure-E-sourced at leading order; the only B in `<even_B odd_B>` is then the 5th-order `<g2_B g3_B>`, see 3.3.)
* Band sums `S = sum_l (2l+1) C_l`; ratios `r0 = S_BB^(1,1)/S_EE^(1,1)`, `r1 = S_BB^(2,1)/S_EE^(2,1)`, `r2 = S_BB^(2,1)/sqrt(S_BB^(2,2) S_EE^(1,1))`, `r3 = S_BB^(2,2)/S_EE^(2,2)`, `rho_EE^(2,1) = S_EE^(2,1)/sqrt(S_EE^(2,2) S_EE^(1,1))`.

### 2.1 Screen basis / sign convention (task item 3)

* **sachsray defines no screen basis.** `physics.tidal_matrix` takes `(Phi00, W1, W2)` as components of `T` in whatever orthonormal dyad the input maps use, integrates `J'' = TJ` in that dyad, and `observables_from_jacobi` reads `gamma1 = -(A00-A11)/2`, `gamma2 = -(A01+A10)/2` in the same dyad; at linear order literally `gamma1 + i gamma2 = -int K (W1 + iW2)`. Hence if `(W1,W2)` is a HEALPix `(Q,U)` pair in `(e_theta, e_phi)`, `(gamma1,gamma2)` is a HEALPix `(Q,U)` pair in the same basis. No `e_theta`/`screen basis` definition exists anywhere in `sachsray/` or the legacy `sachsfield/` (grep): the basis is inherited from the inputs.
* **healpy sign** (`convention_check()` in the script, on `psi = sin^2(theta)cos(2phi)`, `3cos^2(theta)-1`, l <= 2): `hp.alm2map_spin([sqrt((l+2)!/(l-2)!) psi_lm, 0], spin=2)` returns **minus** the trace-free covariant Hessian, `(Q,U) = -(H_tt - H_pp, 2H_tp)` (residual 2.6e-11 for the minus sign vs 15.8 for plus); `almxfl(psi, -l(l+1))` is exactly `H_tt + H_pp` (2e-11). This is the CMB convention `2a_lm = -(E+iB)`. Hence `alm_E = -(1/2) sqrt(...) psi_lm` for `T = Hess psi`. The sign only fixes the sign of E (pure E either way).
* **Why `FullSkySource` was not used:** `sachsfield/sources.py` draws `Phi00`, `Psi_+`, `Psi_x` as three INDEPENDENT scalar Gaussian maps; a spin-2 pair built from two independent scalars has equal E and B power and is uncorrelated with `Phi00` -- not a scalar-sourced field.
* **Gate (linear map B/E):** identity convention 1e-11 .. 5e-9 in every band <= L_cut; `gamma2 -> -gamma2`: 0.78-1.1; swap `(gamma1,gamma2)`: 0.93-1.3; `gamma1 -> -gamma1`: 0.78-1.1 (all seeds, both nside). The identity convention is the correct one and is the one justified above, not a fit.

## 3. Results

Diagnostics (all seeds, both cases): corr(`g1` kappa, Born kappa) = corr(`g1` gamma, Born gamma) = 1.000000, rms ratio 1.00002, max|g1 - Born| = 2e-5 (0.15% of rms); `C_EE^(1,1) / [(l+2)(l-1)/(l(l+1)) C_kk^(1,1)] = 1.000000 +- 1e-6`. rms shear by order (nside 64, seed 1, Gaussian): g1 1.35e-2, g2 3.32e-4, g3 4.51e-6, g4 3.75e-8; rotation omega: 8e-11 (g1), 2.4e-4 (g2), 2.9e-6 (g3) -- clean geometric separation, rotation only from order 2 as expected.

### 3.1 nside 64, 4 seeds (mean +- seed scatter), bands within the band limit (L_cut = 89)

Non-Gaussian potential (the FK test):

| band | Nmodes | r0 = BB11/EE11 | **r1 = BB21/EE21** | r2 | r3 = BB22/EE22 | rho_EE^(2,1) | EE21/EE11 | EE22/EE11 | BB22/EE11 |
|---|---|---|---|---|---|---|---|---|---|
| 2-9 | 96 | 2.2e-11 | **-2.0e-6 +- 3.0e-6** | -4e-8 +- 7e-7 | 0.70 +- 0.17 | +0.06 +- 0.24 | +5.6e-4 +- 2.4e-3 | 9.4e-5 | 6.5e-5 |
| 10-29 | 800 | 2.0e-10 | **+2.2e-6 +- 2.3e-6** | -2e-7 +- 3e-7 | 0.41 +- 0.08 | -0.129 +- 0.084 | -2.2e-3 +- 1.6e-3 | 2.7e-4 | 1.1e-4 |
| 30-89 | 7200 | 2.4e-9 | **-3.2e-6 +- 7.0e-6** | +5e-7 +- 8e-7 | 0.226 +- 0.014 | -0.107 +- 0.055 | -2.3e-3 +- 1.2e-3 | 4.5e-4 | 1.0e-4 |

Gaussian potential (control; `<even,odd>` vanishes by symmetry so r1 is 0/0 and only r2, rho are meaningful):

| band | r0 | r1 | r2 | r3 | rho_EE^(2,1) | EE22/EE11 | BB22/EE11 |
|---|---|---|---|---|---|---|---|
| 2-9 | 2.9e-11 | -8e-7 +- 4e-6 | -4e-8 +- 1e-6 | 0.62 +- 0.21 | +0.17 +- 0.16 | 9.9e-5 | 6.1e-5 |
| 10-29 | 2.3e-10 | +4e-6 +- 1e-5 | -1e-7 +- 4e-7 | 0.40 +- 0.06 | -0.010 +- 0.045 | 2.3e-4 | 9.2e-5 |
| 30-89 | 2.7e-9 | -6e-5 +- 5e-5 | +2e-7 +- 1e-6 | 0.230 +- 0.013 | +0.005 +- 0.016 | 3.9e-4 | 8.9e-5 |

(Bands 90-99 and 100-178 contain no linear power by construction; every (1,1)-normalised ratio there is meaningless -- see `tables_nside64.md` if needed; r3 there is 0.018 and 0.007.)

### 3.2 nside 128, seed 1 (L_cut = 179)

NG: 

| band | Nmodes | r0 | **r1** | r2 | r3 | rho_EE^(2,1) | EE21/EE11 | null sigma(r1) |
|---|---|---|---|---|---|---|---|---|
| 2-9 | 96 | 4.6e-12 | **-1.6e-6** | +3.9e-7 | 0.90 | -0.238 | -1.8e-3 | 9e-7 |
| 10-29 | 800 | 4.5e-11 | **+1.6e-6** | -2.0e-7 | 0.45 | -0.083 | -1.2e-3 | 1.9e-6 |
| 30-99 | 9100 | 4.6e-10 | **+8.2e-7** | -1.2e-7 | 0.33 | -0.083 | -1.55e-3 | 1.5e-6 |
| 100-179 | 22400 | 8.1e-9 | **+4.7e-5** | -7.6e-6 | 0.19 | -0.070 | -1.7e-3 | 3.5e-6 |

Gaussian control: rho_EE^(2,1) = -0.030, -0.016, +0.008, -0.008 (consistent with 1/sqrt(Nmodes) = 0.10, 0.035, 0.010, 0.007); r1 = -7.6e-6, +5.9e-6, +5.6e-6, +2.8e-5 (0/0-type); r3 = 0.80, 0.43, 0.35, 0.18.

The FK-type EE cross term is detected at ~10 sigma in the 100-179 band (rho = -0.070 with noise 0.007) and in the 4-seed nside-64 mean (-0.107 +- 0.028 at 30-89). Its BB counterpart is at the leakage floor: |r1| <= 5e-5 everywhere, typically 1e-6, against the claimed 0.42-0.95. `C_EB^(2,1)/C_EE^(2,1)` is likewise at noise.

### 3.3 Raw antithetic (eps = +-1 only) cross and the 5th-order identity

`r1_raw = S_BB^(even x odd)/S_EE^(even x odd)`: NG nside 128 = 5.3e-5 (2-9), 1.5e-4 (10-29), 1.7e-4 (30-99), 2.0e-4 (100-179); the separately extracted `<g2_B g3_B>/<g2_E g1_E>` = 5.4e-5, 1.5e-4, 1.7e-4, 1.6e-4 -- identical to 2-3 digits in every band (also at nside 64). `<g4_B g1_B>/<g2_E g1_E>` < 1e-8. So all the B in the raw even x odd cross is the 5th-order piece through `g3` (which carries B), and the order-3 piece is zero: exactly the theorem's structure.

### 3.4 2PCF version (item 6), paper's own curved-sky operator

From the measured order-(2,1) NG spectra: `xi_-(gamma) ~ gamma^4.00` and `xi_+ ~ gamma^0.00` for gamma < 10' (fit), i.e. the gamma^4 is the `d^l_{2,-2}` kernel of a BB = 0 spectrum. Inverting both with `figure12.build_curved_matrix/forward_curved` (0.1-dex running mean on the noisy seed-averaged C_l, gamma 1'-180 deg, no DC subtraction): for the pure-E O0 spectrum `T22^-1[xi_+]` and `T_{2,-2}^-1[xi_-]` agree to |C+ - C-|/|C+ + C-| = 7e-5..2.1e-2 per l (EE recovered to 0.2-3.6%); for the NG (2,1) spectrum to 1e-3..9e-2 for l <= 60 (EE recovered to 2-15% except near zero crossings). The operator therefore has an intrinsic per-l "BB/EE" floor of order 1e-2 (up to 0.1 at zero crossings) even for a strictly pure-E input; the harmonic-space truth is <= 1e-5. On the paper's 1'-5000' grid with DC subtraction the inversion fails for these low-l-dominated spectra (EE recovered 0.3x-3.5x), which is outside that grid's design regime (l ~ 100-1500) and is noted only as a fragility, not as a statement about the paper's own use.

## 4. Conclusion

With the exact nonlinear Sachs solver, a scalar-sourced (pure-E) driving field, and a non-Gaussian potential with O(1) skewness, the one-cumulant (FK-type) cross term is present in EE (`rho ~ -0.1`, `-0.2%` of O0 for this synthetic bispectrum) and absent in BB to better than 1e-5 of its EE value (1e-6 typical). The theorem holds as stated: `g1_B = 0` per realisation forces `C_BB^FK = 0` identically; the first B-mode is the 4th-order `<g2_B g2_B>` (FF/post-Born-type, B/E = 0.2-0.9 here, 6e-5..1.2e-4 of O0) plus, for NG fields, the connected four-point term. The paper's FK B/E = 0.42-0.95, the Letter's `f_BB ~ 4e-3..8e-3` premise, and the `gamma^4` clause cannot come from the Sachs dynamics for any scalar-sourced field; they must be artifacts of the zeta_XYZ spin channels (the difference channel must equal the pure-E-consistent value) and/or the 2PCF -> C_l transform. An N-body ray-trace should show only the FF/post-Born-level (Krause-Hirata-type) B-mode.

## 5. Files

* `n1_theorem_sachsray.py` (run/aggregate), `n1_twopcf_check.py`
* `outputs/nside64_seed{1..4}.{npz,json}`, `outputs/nside128_seed1.{npz,json}` (all order-pair C_l's for E, B, kappa; float32 maps of g1..g4; per-band ratios, gate variants, Born checks)
* `outputs/summary_nside{64,128}.json`, `outputs/tables_nside{64,128}.md`, `outputs/figure_nside{64,128}.png`
* `outputs/twopcf_gauss_11_nside64_s0.1.{json,png}`, `outputs/twopcf_ng_12_nside64[_s0.1].{json,png}`
* `outputs/log_*.txt` (runtime: 350 s per nside-64 seed with 4 in parallel; 806 s for nside 128)

Caveats: REPORT.md itself could not be written by this agent (harness rule); nside-128 seed 2 is still running in the background (re-run `aggregate --nside 128` once `outputs/nside128_seed2.json` exists); rows above L_cut in the tables are to be ignored; the synthetic amplitude of FK is not a prediction of the physical one.