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
