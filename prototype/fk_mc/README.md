> **SUPERSEDED / CORRECTION (2026-09-03).** The FK results in this folder
> (`outputs/fk_channels.npz` fk/fk_se, `outputs/figure12_mc.png`, the "Result in one
> line", "FK dominates O0 by 10-80x above 200'", "up to ~7x C_l", the per-panel
> max|FK/(O0+FF)| = 46/78/47/5, and the Figure-12 "modulus pair" story) are
> **RETRACTED**. Full audit: `prototype/bmode_audit/REPORT.md` (master) and
> `prototype/bmode_audit/A1/REPORT.md`. Three independent defects, each established
> in the paper's own 2026-08 audit (`SFT-lensing-paper-analyses/sachs_sft/analyses/
> mc_fk_complete/NOTES.md` sections 3.1-3.3; `docs/fk_audit_2026-08/SUMMARY.md`):
>
> 1. **Wrong vertex.** The run bound the June `equal_time_limber` kappa3 callable
>    (ell_max = 1000, cosine grid frozen at the (1,1,1) corner). Its fold is 4.5x
>    low at 0.5' and FLAT to ~600', where the manuscript's ell_max = 15360,
>    permutation-aware vertex decays by three orders of magnitude and changes sign
>    near 3 degrees. "FK dominates O0 above 200'" is that frozen corner read against
>    a decaying Order-0; with the corrected table |FK/O0| is 0.02 at 5 deg and 0.1
>    at 10 deg, and FK is declared unconverged there anyway.
> 2. **Incomplete estimator.** `simulate_fk_vr` deforms only the observable leg and
>    captures a share T1/(T1+T2) = 0.25 (0.5'), 0.17 (2'), 0.03 (8') of the FK
>    diagram, negative beyond ~12' (-0.16 at 30').
> 3. **Mis-calibrated deformation.** Q was solved against the nominal node
>    covariance V = Sigma2/(2 sigma); the "Levy-peaked Q" of finding 1 is this
>    mis-calibration hitting non-PSD nodes of the Sigma2 table (x3.9 inflation at
>    sigma_lambda = 8, n_lambda = 1000; grid-unstable; this run used n_lambda = 600).
>    The correct calibration is solve_Q(B[k], zeta[k]).
>
> The near-agreement with the June fold at ~1' was the product of (2) and (3).
>
> **Spin structure.** The claim "FK feeds EE+BB and cancels in EE-BB, i.e.
> Delta C_EE^FK = Delta C_BB^FK" is wrong. The FK diagram is
> 2 Re <gamma^(2)(n1) gamma^(1)*(n2)>; the linear response gamma^(1) is pure E
> realisation by realisation, so **C_BB^FK = 0 identically** and xi_-^FK is fixed by
> xi_+^FK through the d^l_{2,-2} kernel (its gamma^4 small-angle law is that kernel,
> not a suppression). The MC's xi_- ~ 0 was the frozen-corner coincident-leg identity
> (the cut1000 fold has xi_- = 0 to 1e-16), and a flat xi_+ has a vanishing pure-E
> xi_- anyway, so the run could not distinguish B = E from B = 0. The first non-zero
> shear B-mode is fourth order (Gaussian FF + connected K4). "kappa E
> parity-forbidden" is also wrong (kappa-B and EB are; kappa-E is the standard
> cross-spectrum).
>
> **What survives:** the sachsray Jacobi robustness and Gaussian bridge gates
> (`stability_test.py`, `route.py`), the verbatim curved-sky transform in
> `figure12.py`, and the Order-0 convention gate. The FF channel here is also
> ~0.5x the paper's fold at 1' with an unphysical constant xi_- floor (outdated).

# FK (non-Gaussian) Monte-Carlo → C_ℓ contribution

Brings the paper's **FK (three-point / non-Gaussian)** channel into the sachsray
pipeline, and delivers its contribution to the convergence angular power spectrum.

## Result in one line

Above ~200′ the Gaussian Order-0 collapses (and flips sign) while the FK channel
stays ~2×10⁻⁶, so **the non-Gaussian FK channel dominates ξ_κ(γ) by 10–80×** at
large separation — and contributes up to ~7× the Gaussian `C_ℓ^κ` at some ℓ.

## The three findings

1. **Brute-force FK is infeasible** at every angular scale. The demo2 deformation
   tensor `Q = solve_Q(Σ₂, ζ) ~ ζ/Σ₂²` is *Levy-peaked* (huge at near-singular σ_×
   directions), so the brute "skewed-variance − Gaussian-variance" is dominated by
   spurious higher moments by 10⁴–10⁷×. (`brute_fk.py`)
2. **sachsray's linear Jacobi is robust.** Routing the heavy-tailed driving through
   `sachsray` is 100% finite, where the paper's nonlinear Riccati overflows on
   29–46% of realisations. The κ=∫δθ observable matches the paper to ~2% on
   Gaussian fields. (`stability_test.py`, `route.py`)
3. **VR is necessary and correct.** Isolating the FK *diagram* requires computing
   only the 3-point contribution — `Q` pulled outside the ensemble average (the
   paper's `simulate_fk_vr`): unbiased, ×10³ lower variance, the only feasible route.

## Files

| file | what |
|---|---|
| `_setup.py` | path wiring to the paper MC engine + canoes (cached tables; no pyccl) |
| `stability_test.py` | sachsray-Jacobi 100% finite vs Riccati 29–46% blow-up |
| `route.py` | bridge: sachsray κ=∫δθ ≡ paper κ=∫s to ~2% (Gaussian gate) |
| `brute_fk.py` | brute-force infeasibility (Levy-Q), all γ |
| `vr_channels.py` | physical **full 3×3** O0/FF/FK channels via the paper's VR → `outputs/fk_channels.npz` |
| `figure12.py` | **MC version of Figure 12** — the 4 angular power spectra (`κκ`, `EE±BB`, `κE`) via the paper's curved-sky Wigner-d transform (ported verbatim) |
| `cl_contribution.py` | simpler flat-sky Hankel ξ_κ(γ) → C_ℓ (κ-only) |
| `../../notebooks/demo_fk_cl.ipynb` | the full Figure 12 (4 panels, O0/FF/FK) + the Order-0 convention gate |

## Figure 12 (the complete angular-power-spectrum figure)

`figure12.py` builds `C_ℓ^{κκ}`, `C_ℓ^{EE}+C_ℓ^{BB}`, `C_ℓ^{EE}−C_ℓ^{BB}`, `C_ℓ^{κE}`,
each split O0 / O0+FF / O0+FF+FK, from the MC's full 3×3 ξ_ab. Observable extraction:
`ξ_κκ=ξ₀₀`, `ξ_+=ξ₁₁+ξ₂₂` (m,n=2,2), `ξ_-=ξ₁₁−ξ₂₂` (2,−2), `ξ_κγt=−ξ₀₁` (2,0).
Verified: at Order-0, `EE+BB ≈ EE−BB ≈ κκ` (B=0, `C_EE=C_κκ`) — the convention gate. FK
feeds the **modulus pair** (`κκ`, `EE+BB`) as a low-ℓ excess and cancels in the rest.

## Run

```bash
python prototype/fk_mc/vr_channels.py --n-real 60000   # the channels (minutes)
python prototype/fk_mc/cl_contribution.py              # the C_ell figure
```

## Caveat

`C_ℓ` is the small-angle Hankel transform of a ~20-point ξ(γ); reliable for
ℓ ≳ 1/θ_max. A genuine *brute-force total-NG* C_ℓ (no Levy-Q) needs a 2nd-order-PT
field generator — a separate, larger build.
