
# T3 — The FK term in the paper's own diagrammatics: can a B-mode appear at order K3?

**Verdict: refuted.** The theorem holds inside the paper's formalism; `C_BB^FK = 0` identically. The paper's B-mode (0.95 → 0.42 of E over ℓ = 60–1500, i.e. `C_BB^FK / C_EE^O0 ≈ 6.4e-3`) is reproduced exactly from the deployed sweep and traced to one defect: the deployed ζ_Bmod channel equals ζ_TTT (pair phase dropped in the HIGH branch), fixed in canoes **after** the paper's table and sweep were produced.

Scripts (all read-only on the paper; outputs quoted below): `/Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/T3/{check_table_econsistency.py, check_fk_eb.py, kernel_identity_test.py}`.

## (a) What the FK term is (path_int.tex)

* Reference action `S_0` (l.160–161) carries the linear dynamics and `W^(2)`; `S_int` (l.162–164) carries `V_F = -i∫F_{ijk} X̃_i X_j X_k` and `-Σ_{k≥3} W^(k)[iX̃]`. No `W^(1)`: the driving field is centred.
* Propagators (l.203–216, 230–259): `⟨XX⟩_0 = C = R ζ^(2) R`, `⟨X̃X⟩_0 = -iR`, `⟨X̃X̃⟩_0 = 0`; `R_{ij}(n,λ;n',λ') = δ_{ij} δ(n'-n) Θ(λ'-λ) e^{-2∫θ^(sa)}` (l.230–243) — carries a **directional δ-function**.
* Vertices (l.349–381): `V_F` local (one response leg, two physical legs, tensor `F_{ijk}`, Table Fabc: `F_111=F_122=F_133=-1`, `F_212=F_313=-2`); `V_{K3} = -(i³/3!)∫ X̃_a X̃_b X̃_c ζ^(3)_{abc}` non-local.
* Second order (l.444–478): `½⟨X_a X_b (V_F² + 2 V_{K3} V_F)⟩_0`; surviving topologies FF and FK; KK forbidden.

Wick-contracting `2V_{K3}V_F`: K's three `X̃` must absorb three of the four `X` legs `{z1, z2, F_j, F_k}` through `R`, F's `X̃` takes the fourth. Two placements survive (K→{F_j,F_k,z2}, F's X̃→z1, and the mirror); the third (K→{z1,z2,F_j}, F's X̃→F_k) is an equal-point self-contraction `R(z;z)=Θ(0)=0` (Itô/causal; the MC's Euler scheme evaluates F at the old state; "closed response cycles vanish", l.342, 562). All four `X̃` are used on `R` lines, so **4 R lines, 0 C lines** — as the audit's symbolic re-expansion found (`input_ready_b_to_zeta_spec.md:35–45`, "2 FK terms × 6 pairs × 40 gammas").

Explicit form (net phase real, `(-i)^4 · (+i/6) · 3!`):

```
FK_ab(z1,z2) = ∫dz dz'_1 dz'_2 dz'_3  R_{ai}(z1;z) F_{ijk} R_{jm}(z;z'_1) R_{kn}(z;z'_2) R_{bp}(z2;z'_3) ζ^(3)_{mnp}(z'_1,z'_2,z'_3)  + (z1,a ↔ z2,b)
```

The δ-functions in R pin `z'_1, z'_2` to direction `n1` (the F-vertex side) and `z'_3` to `n2`; the equal-shell collapse (appendix eq. `appendix equal shell approx`) makes it `ζ_{mnp}(γ,λ')` on the `(1, cosγ, cosγ)` family (insights.tex:186 footnote). This is exactly

```
FK(n1,n2) = ⟨X^(2)(n1) X^(1)(n2)⟩ + (1↔2),   X^(1) = Rφ,   X^(2) = R F (Rφ)(Rφ),
```

at first order in ζ^(3) (the Gaussian part of ⟨φφφ⟩ vanishes for a centred field). **Confirmed.** `⟨X^(1)⟩ = 0`, so the moment-vs-covariance issue of the FF plateau (insights.tex:112–117 footnote) does not touch FK: the disconnected piece `⟨X^(2)⟩⟨X^(1)⟩ = 0`. The response-propagator leg (`γ^(1)`) sits at the point opposite the F vertex, symmetrised over the two points.

## (b) Spin structure of the shear entries

For `a = 2` (σ₊ → γ₊): `F_{2jk} ≠ 0` only for `(j,k) = (1,2),(2,1)`, so `γ₊^(2)(n1) ∝ -2 R[(Rφ)_1 (Rφ)_2]` — one **Φ00 leg and one Ψ₊ leg, both at n1**. The K3 leg at `n2` carries the component of the external field there. Hence

* `(1,1)`: `ζ_{Φ00 Ψ₊ Ψ₊}(n1, n1, n2)`; `(2,2)`: `ζ_{Φ00 Ψ× Ψ×}(n1, n1, n2)`; `(1,2)`: `ζ_{Φ00 Ψ₊ Ψ×} = 0` by parity.
* `ξ₊^FK ∝ ⟨Φ00(n1)[Ψ₊(n1)Ψ₊(n2) + Ψ×(n1)Ψ×(n2)]⟩ = Re⟨Φ00(n1) Ψ0(n1) Ψ0*(n2)⟩` — the spin-(2,−2) pairing, `ζ_Bmod` on `(1,c,c)`.
* `ξ₋^FK ∝ Re⟨Φ00(n1) Ψ0(n1) Ψ0(n2)⟩` — total spin 4, `ζ_TPP` on `(1,c,c)`.

The perm-aware callable fills exactly this (`perm_aware_kappa3_callable.py:215–224`: `K011 = (TPP+Bmod)/2`, `K022 = (Bmod−TPP)/2`; `rebuild/probe_eb_structure.py` verifies `K011+K022 = Bmod → ξ₊`, `K011−K022 = TPP → ξ₋`).

So the conclusion's wording (conclusion.tex:56–61) "the Ricci-focusing scalar correlated with the **squared modulus** of the Weyl-shear potential … blind to the orientation of the shear" is **wrong for the shear–shear entries**: the two Ψ legs are at different points. `|Ψ0|²` at one point occurs only in the κκ entry (`F_122, F_133`: both Ψ legs at n1, Φ00 at n2 — the permuted `(c,1,c)` geometry), where the modulus language is correct.

## (c) How ζ_TPP and ζ_Bmod are built; is E-consistency enforced?

Appendix (`append: driving-field spectra`, appendix.tex:346–556):
* eq. `appendix zeta squeezed` (l.446–451): `ζ_XYZ(γ,χ) = Σ 𝒢 P_{ℓ3}(cosγ) B^{XYZ}` with a footnote (l.457–462) that the scalar 3j is only a parity selector and "the spin weight is carried separately by the spin-weighted-harmonic contraction". The `P_{ℓ3}` in the displayed equation is a simplification; the code's LOW branch (`canoes/nuell/correlation/_spin_aware_three_pt.py:566–712`) uses `W3j(ℓ1ℓ2ℓ3; −s1, m2, s1−m2)` times `_{s2}Y`, `_{s3}Y` with the canonical frame n1-at-pole, i.e. a `d^{ℓ3}_{m,−s}` structure; the HIGH branch (`canoes/sachs/kappa3.py:1747–1800`, `_kappa3_limber_alpha_kernel`) is the flat-sky orientation average `2π i^S e^{iS arg Q} J_S(|Q|) e^{i(s1 arg l1 + s3 φ)}`. The spec table (`input_ready_b_to_zeta_spec.md:55–64`) lists the collapsed-family kernels: TTT `P_{ℓ3}`, TPP `d_{2,−2}`, Bmod `d_{2,+2}`.
* "spin-2 subtlety" (l.492–526): `ζ_TPP = ⟨Φ00(Ψ₊²−Ψ×²)⟩` (unconjugated, difference), `ζ_B = ⟨Φ00|Ψ0|²⟩` (conjugated, sum), reconstruction eq. `appendix modulus reconstruction` (l.520–524). Note: the appendix writes both as single-point products — the same modulus misreading as (b).

**On the collapsed geometry the HIGH kernels are (verified to 1e-16 against canoes' function, `kernel_identity_test.py`):** TTT `2πJ0(vγ)`, Bmod `2πJ0(vγ)cos2φ`, TPP `2πJ4(vγ)cos2φ`, with φ the angle between the coincident-leg and offset-leg Fourier modes. Therefore, with `C(v) ≡ 2πv∫u du dφ B(u,v,w)cos2φ`, `Bmod = ∫dv C(v) J0(vγ)` and `TPP = ∫dv C(v) J4(vγ)` — **the pure-E identity `T22^{-1}[ξ₊] = T_{2,−2}^{-1}[ξ₋]` holds shell by shell, automatically, for any bispectrum and any ℓ-window, if the phases are right.** Nothing in the pipeline enforces it: six channels computed independently → `K_abc` reconstruction → sft-wick fold (linear in ζ, R depends only on λ) → `analysis3/plot_cl_EB_polarization.py` transforms ξ₊ with `d_{22}` and ξ₋ with `d_{2,−2}` separately. So it *can* be violated, and it was:

* `canoes` commit **a95c0d6 (2026-09-02 14:55)** "apply the residual spin phase when S = 0 by cancellation": the kernel returned `2πJ0(|Q|)` for `S=0` before applying `spin_extra = exp(i(s1 arg l1 + s3 φ))`; for Bmod's `(0,2,−2)` this drops `exp(−2iφ)`, so "the high-ℓ ζ_Bmod was … bit-identical to ζ_TTT". Blast radius: only Bmod. Converged fixed Bmod/TTT on shell 8, `(c,1,c)`: 0.483/0.575/0.674/0.859/1.022 at 0.5′/1′/2′/5′/12′ (rezeta 0.482 at 0.5′).
* Deployed table `table_permclosed.npz` built **2026-08-26 09:46**; FK sweep `xi_C_corr_op_K_limber_FK_cut15360_permfix.npz` **2026-08-26 09:48**; `figures/cl_EB_polarization.pdf` 2026-09-02 17:52 (redeploy from the same sweep, commit c32395c). No post-fix table exists in the reproduction package or SFT-WL-B.
* Measured on the deployed table (`check_table_econsistency.py`, source shell λ = 2327 Mpc, `(1,c,c)` family): Bmod/TTT = 1.0004 (0.6′), 1.0005 (1.1′), 1.0013 (4.2′), 1.0041 (10.8′), 1.0073 (15.4′); median over all rows/shells 1.0001. The residual is the LOW-branch (ℓ ≤ 60) share, which does carry the phase.

**Consequence:** ξ₊^FK is the J0 transform of the phase-free `C'(v) = 2πv∫u du dφ B`, ξ₋^FK the J4 transform of the correct `C(v)`; the pipeline's "B-mode" is `(C'−C)/2`. Toy tree bispectrum: `C/C'` = 0.23, 0.33, 0.52, 0.55 at v = 100, 300, 700, 1200 → implied BB/EE = 0.63, 0.50, 0.32, 0.29, the same shape as the paper's 0.95 → 0.42.

## Reproduction of the paper's numbers (`check_fk_eb.py`, figure12 port of the analysis3 transform)

| ℓ | O0 BB/EE (transform floor) | FF BB/EE | FK BB/EE |
|---|---|---|---|
| 60 | +3e-4 | 0.023 | **0.951** |
| 218 | −1e-3 | 0.031 | 0.724 |
| 514 | −5e-3 | 0.038 | 0.583 |
| 977 | −0.014 | 0.036 | 0.487 |
| 1500 | −0.040 | 0.019 | **0.420** |

`C_BB^FK / C_EE^O0` = 6.4e-3 (ℓ=60), 6.5e-3 (ℓ=1500) at z_s = 5 — the Letter's f_BB. The superseded June (cut1000) sweep gives BB/EE ≈ 1.0–1.9 with sign flips, i.e. the earlier "C_EE = C_BB" claim was the same artifact at a lower cutoff.

## (d) MC validation

`mc_fk_complete/SPEC.md:54–66`: the observable tested is the **κκ entry (a,b) = (0,0)** only. `NOTES.md:47–59`: "Shared vertex table … What is still NOT tested is the vertex table itself"; §5d: FK depends on `(ζ, D)` alone and both sides read the same `zeta6`. The κκ FK never touches `K022 − K011` and the MC would have agreed with any ζ_Bmod, right or wrong. It could not have caught the spin defect; neither could the "workflow-independent quadrature" (`fk_kernel_crosscheck.py`), which reads the same table.

## (e) γ⁴ and commit e63d45b

`git show e63d45b`: only `sections/insights.tex` (+4/−2, l.206–213) changed — "The γ⁴ scaling sets in only below an arcminute" replaced by "the spin-4 and spin-2 kernels send Order-0 to zero with it, so below an arcminute the ratio climbs". The commit message correctly identifies γ⁴ as the `J4`/`d^ℓ_{2,−2}` kernel (Order-0 slope 3.96, FK 3.61). But `conclusion.tex:64–68` (edited 7e57a1e, 2026-08-29, replacing "splits equally … only as γ→0") still says the difference "survives, suppressed as γ⁴ below an arcminute rather than absent", and `appendix.tex:538` still labels ζ_TPP, ζ_PPP "suppressed as γ⁴ as γ→0" as if a property of the channel. The phrase "kinematic signature of spin 4" is not in the current text. The γ⁴ law says nothing about B-modes: for a pure-E cross-spectrum ξ₋ *must* go as γ⁴ while BB = 0.

## Verdict and corrected statement

Within its own formalism the paper's FK B-mode cannot be nonzero. The E-consistency identity is not enforced anywhere; it is a property the vertex table must satisfy and the deployed one does not, because ζ_Bmod = ζ_TTT for ℓ > 60. The paper should state:

> `ΔC_ℓ^{BB,FK} = 0`. `ξ₊^FK` and `ξ₋^FK` are the `d^ℓ_{22}` and `d^ℓ_{2,−2}` transforms of the single spectrum `ΔC_ℓ^{EE,FK}`; the γ⁴ law of `ξ₋^FK` is the spin-4 kernel, shared with Order-0. The leading shear B-mode is fourth order in the driving field: the Gaussian FF term (two F vertices; unperturbed-path post-Born analogue, ≈1/30 of its E-mode) plus the connected four-point cumulant through two F insertions (F F K4, allowed by the selection rule with k_F = 2, n = 4, p = 2 — Order 3 of the expansion).

Collateral, outside this lens but load-bearing: with the fixed Bmod ≈ 0.48 TTT at 0.5′, the κκ and ξ₊ FK amplitudes (`G_00 = −2(TTT+Bmod)`) also change by up to ~25 % at small γ, so the 1.3–1.7 % headline needs the rebuild too; and any rebuilt table should adopt `BB^FK = 0` (to quadrature accuracy; the deployed 96×64 rule has a calibrated placement error of 1.5 % at 0.5′ and ×1.98 at 12′) as an acceptance gate. The Letter's premise ("exceeds every known B-mode by 8–16×") rests entirely on the artifact.