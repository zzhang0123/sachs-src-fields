
# N2 — products-level E/B audit of the paper's deployed 2PCF sweeps

Agent N2, 2026-09-03. Everything here is computed from the manuscript's OWN production 2PCF sweeps with the manuscript's OWN curved-sky (Wigner-d) transform; nothing was re-derived from the vertex or the propagators. All files live in `/Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/n2_products/`. (This report was returned in the structured output rather than written to REPORT.md, per the harness rule; the parent may save it there.)

## Verdict in one paragraph

The paper's FK E/B numbers are reproduced exactly (C_BB/C_EE = 0.9511 at l=60, 0.4203 at l=1500, median 0.6287 over 50-1500; FF median 0.0321 = 1/31.2). The transform is not the problem: its own floor on the pure-E Order-0 term is |BB/EE| <= 4e-2 (1e-3 once the missing [0, 0.5'] piece is added), and O0 passes the E-consistency test D/S = 1 to 0.1-8%. The FK term fails it by a factor 2.4-40: D/S = 0.025 (l=60), 0.24 (l=300), 0.41 (l=1500). The deployed FK xi_- is too SMALL for pure-E consistency with the deployed FK xi_+ by 2.5x (10'), 3.8x (30'), 5.0x (60'), with two independent routes (curved-sky harmonic back-transform and the flat-sky Crittenden/Schneider integral) agreeing to 7% and both returning 0.98-1.00 on the O0 control. The l=1500 value 0.42 is ROBUST to every large-gamma treatment (0.38-0.54), the DC convention and the small-angle floor (0.455): it is intrinsic to the deployed xi_-^FK / xi_+^FK at arcminute scales, i.e. to the vertex's spin-channel construction, not to the transform. The l=60 value 0.95 is NOT robust (0.53-1.17 across treatments) and the O0 control shows that any BB/EE at l=60 built from a 2PCF converged only to 1 deg is meaningless (pure-E O0 truncated at 1 deg gives 1.000). Between the two vertex versions the same pipeline moves BB/EE from 1.04-1.81 (June cut1000, where xi_11 = xi_22 to 1e-11 so BB = EE identically) to 0.42-0.95 (Aug cut15360 permfix): the split is set by the vertex spin channels. Finally the paper's "gamma^4 below an arcminute" for xi_-^FK is exhibited by the pure-E O0 term as well (fitted exponents O0 3.85, FF 3.71, FK 2.44, FK_june 2.000): it is the band-limited d^l_{2,-2} kernel, not a B-mode signature. All of this is what the THEOREM predicts.

## 0. Setup, provenance, commands

Inputs (read-only, byte-identical to the manuscript's production files, root `/Users/zzhang/Documents/MyDrafts/STF_lensing/SFT-lensing-paper-analyses/sachs_sft/sftwick_outputs/2PCF/`):

| tag | file | order rows | vertex |
|---|---|---|---|
| O0 | `C_corr_op_O0/xi_C_corr_op_O0.npz` | 0 | none (corr_op table, `ell_max = 5000`, `ell_taper_frac = 0.2`; `callables/C_propagator/corr_op/*.meta.json` line 51) |
| FF | `C_corr_op_K_limber_FF/xi_C_corr_op_K_limber_FF.npz` | 2 | F x F |
| FK | `C_corr_op_K_limber_FK_cut15360_permfix/xi_C_corr_op_K_limber_FK_cut15360_permfix.npz` | 2 | perm-aware, ell_max = 15360 (the manuscript's) |
| FK_june | `C_corr_op_K_limber_FK/xi_C_corr_op_K_limber_FK.npz` | 2 | June cut1000, sorting callable (superseded) |

Each file has 240 rows = 6 component pairs x 40 separations; gamma = geomspace(0.5', 5000') (0.5'..83.3 deg), t_final = 2313.03 Mpc (z_s = 5). Grouping is a verbatim port of `analysis3/plot_analysis3_cl_decomposition.py::load_sweep_order` and `_gamma_arcmin` (plus an assertion that every pair shares the grid). Observables: xi_kk = xi_00, xi_+ = xi_11 + xi_22, xi_- = xi_11 - xi_22, xi_12, xi_01, xi_02.

Transform: `prototype/fk_mc/figure12.py` (verbatim port of analysis3's `wigner_d`, `build_curved_matrix`, `forward_curved`, `ELL`): C_l = 2 pi int d(cos theta) xi d^l_{mn}, PCHIP on a 20000-point log-theta grid over the data range only, DC subtraction xi - xi(83 deg). S = T_{2,2}[xi_+] = EE+BB, D = T_{2,-2}[xi_-] = EE-BB. The paper's ELL grid (30 integer multipoles 3..1500) is used for the reproduction; ell = 300 was added for the quoted ratios (not on the paper's grid; nearest 270 and 335).

Commands:

```
cd /Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/n2_products
/opt/homebrew/Caskroom/miniconda/base/envs/PyCCL/bin/python n2_crosscheck_analysis3.py   # imports analysis3 itself + pyccl gate
/opt/homebrew/Caskroom/miniconda/base/bin/python n2_products_eb.py                        # everything else (~5 s)
```

Outputs: `results.npz`, `results.json`, `tables.md` (full tables), `crosscheck_analysis3.npz`, `fig_2pcf.png`, `fig_eb_DS.png`, `fig_robustness.png`, `fig_pureE_prediction.png`, `fig_dense_DS.png`.

## 1. The 2PCF sweeps (subset; full tables in `tables.md` section 1)

Parity nulls: max|xi_12| = 1e-19 (O0), 4.7e-15 (FF), 0 (FK, FK_june); max|xi_02| = 0 / 5e-32 / 0 / 0.

| gamma ['] | O0 xi_+ | O0 xi_- | O0 xi_-/xi_+ | FK xi_+ | FK xi_- | FK xi_-/xi_+ | FK_june xi_+ | FK_june xi_- |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.500 | +8.418e-4 | +2.421e-8 | 2.9e-5 | +1.959e-5 | +1.680e-8 | 8.6e-4 | +4.2878e-6 | -9.9e-17 |
| 1.015 | +8.080e-4 | +3.940e-7 | 4.9e-4 | +1.616e-5 | +1.774e-7 | 1.1e-2 | +4.2878e-6 | -4.1e-16 |
| 2.062 | +6.943e-4 | +5.578e-6 | 8.0e-3 | +1.240e-5 | +4.795e-7 | 3.9e-2 | +4.2877e-6 | -1.7e-15 |
| 4.188 | +4.819e-4 | +4.402e-5 | 9.1e-2 | +8.325e-6 | +5.557e-7 | 6.7e-2 | +4.2877e-6 | -6.9e-15 |
| 8.506 | +3.106e-4 | +5.812e-5 | 0.187 | +4.810e-6 | +5.445e-7 | 0.113 | +4.2875e-6 | -2.9e-14 |
| 17.276 | +1.635e-4 | +6.404e-5 | 0.392 | +2.264e-6 | +4.052e-7 | 0.179 | +4.2869e-6 | -1.2e-13 |
| 35.085 | +6.462e-5 | +5.491e-5 | 0.850 | +7.970e-7 | +2.361e-7 | 0.296 | +4.2840e-6 | -4.9e-13 |
| 71.255 | +1.736e-5 | +3.814e-5 | 2.20 | +1.725e-7 | +1.096e-7 | 0.635 | +4.2724e-6 | -2.0e-12 |
| 144.713 | +2.089e-6 | +1.992e-5 | 9.5 | +5.088e-9 | +4.353e-8 | 8.6 | +4.2247e-6 | -8.2e-12 |
| 293.901 | -4.241e-7 | +8.068e-6 | -19 | -8.245e-9 | +1.654e-8 | -2.0 | +4.0358e-6 | -3.3e-11 |
| 1212.23 | -5.121e-8 | +5.174e-7 | -10 | -2.845e-8 | -2.054e-9 | 0.07 | +1.6853e-6 | -4.0e-10 |
| 5000.00 | -4.749e-10 | +1.394e-8 | -29 | +9.723e-9 | +2.583e-9 | 0.27 | -5.5615e-9 | -1.7e-9 |

Three things to read off:

* The deployed O0 xi_- is itself band-limited: xi_-/xi_+ = 2.9e-5 at 0.5', 8e-3 at 2', rising ~gamma^4 (a Born-level LCDM shear at z_s = 5 would have ~0.1 at 1'). This is the `ell_max = 5000` cutoff of the corr_op table (1/5000 rad = 0.69'): below ~1/ell_max the d^l_{2,-2} ~ (l theta)^4/384 kernel gives xi_- ~ theta^4 for ANY spectrum. The pure-E Order-0 term therefore shows the paper's "gamma^4 below an arcminute" too.
* FK's xi_-/xi_+ at 8'-35' is 0.11-0.30, 2-3x below O0's 0.19-0.85, although FK's xi_+ is the STEEPER 2PCF (falls 25x from 0.5' to 35' vs 13x for O0); more small-scale power should give a larger, not smaller, xi_-/xi_+ for a pure-E field (quantified in section 6).
* FK_june has xi_+ = xi_kk = 4.2878e-6 flat to 5 digits from 0.5' to 35' and xi_11/xi_22 = 1.0000: its xi_- is -1e-16 x gamma^2.000 (machine-level), i.e. xi_-^FK = 0 exactly, which is the earlier paper version's "C_EE^FK = C_BB^FK".

## 2. Reproduction of the paper's E/B numbers

On the paper's exact ELL grid (`results.json: step2_paper_grid_*`):

| term | BB/EE l=60 | l=1500 | median 50-1500 | paper |
|---|---:|---:|---:|---|
| FK (cut15360 permfix) | **+0.9511** | **+0.4203** | **+0.6287** | 0.95 -> 0.42, "about two thirds", docstring median 0.63 |
| FF | +0.0227 | +0.0188 | +0.0321 (EE/BB = 31.2) | "some thirty times", docstring ~31 |
| O0 (gate = transform floor) | +0.0003 | -0.0404 | -0.0020 | "C_BB has no Order-0 component" |
| FK_june (superseded) | +1.0423 | +1.1474 | +1.1271 | (earlier draft: BB = EE) |

Cross-checks:
* The analysis3 module itself, imported in the PyCCL interpreter (`n2_crosscheck_analysis3.py`), gives identical S, D: max relative difference vs the port 2.7e-11 (S) and 3.3e-12 (D) over all four sweeps (`results.json: step2_crosscheck_vs_analysis3`).
* O0 vs PyCCL C_l^{kk} (linear P(k), same cosmology, CMBLensingTracer z_s = 5): S/pyccl = 0.911-0.990 over 50-1500 (0.911 at l=1500), D/pyccl = 0.985-1.003 (0.988 at 1500), C_kk/pyccl = 0.914-0.999. The -4% O0 "B-mode" at l=1500 is entirely the missing [0, 0.5'] part of the S integrand (d_{2,2}(0) = 1, d_{2,-2}(0) = 0, so the truncation biases S but not D): extending the 2PCF below 0.5' (xi_+ constant, xi_- ~ gamma^3.92) brings O0's BB/EE to +0.001 / +0.000 / +0.001 at 60/300/1500 (section 4, last rows).
* With ell = 300 added: FK BB/EE = +0.6180, FF +0.0344, O0 -0.0016 (paper-grid neighbours FK: 0.649 at 270, 0.609 at 335).
* The Letter's f_BB = C_BB^FK / C_EE^O0 (z_s = 5) from these products: 6.42e-3 (l=60), 6.33e-3 (300), 6.47e-3 (1500); C_EE^FK / C_EE^O0 = 6.7e-3 / 1.02e-2 / 1.54e-2. The Letter quotes 6.33e-3..8.28e-3 at 60 and 3.85e-3..5.03e-3 at 1500 (its lo/hi bracket); the deployed z_s = 5 product sits at the low end at l=60 and above the bracket at l=1500.

## 3. E-consistency in ell space: D/S = T_{2,-2}[xi_-] / T_{2,2}[xi_+] (pure E: 1)

| ell | O0 | FF | FK | FK_june |
|---:|---:|---:|---:|---:|
| 60 | 0.999 | 0.956 | 0.025 | -0.021 |
| 93 | 1.001 | 0.951 | 0.130 | -0.033 |
| 142 | 1.001 | 0.946 | 0.124 | 0.079 |
| 218 | 1.002 | 0.940 | 0.160 | -8.8 |
| 300 | 1.003 | 0.933 | 0.236 | -0.287 |
| 415 | 1.005 | 0.932 | 0.243 | -4.4 |
| 637 | 1.009 | 0.926 | 0.263 | -0.306 |
| 977 | 1.020 | 0.927 | 0.319 | -0.250 |
| 1211 | 1.038 | 0.934 | 0.372 | -0.102 |
| 1500 | 1.084 | 0.963 | 0.408 | -0.069 |
| max abs(D/S - 1), 50-1500 | 0.084 | 0.074 | 0.975 | 9.8 |

O0 (pure E by construction) sits at 1 to 0.1% for 60 <= l <= 300 and drifts to 1.08 at 1500 for the small-angle-truncation reason above (the floor). FF is 0.93-0.96, consistent with its own 3% B-mode ((1-r)/(1+r) with r = 0.032 gives 0.938). FK is 0.025-0.41: the deployed difference channel is 40x (l=60) to 2.4x (l=1500) too small for pure-E consistency with the sum channel, far outside the O0 floor. The dense integer-ell version (`fig_dense_DS.png`, ell = 2..3000) shows the FK D/S rising smoothly from ~0.25 at l=100 to 0.65 at l=3000 with no sign of reaching 1; O0 stays at 1 until the floor sets in above ~1000. FK_june's D/S is NEGATIVE (D ~ 0 with a small negative large-angle contribution), which is why its BB/EE exceeds 1.

## 4. Robustness of the FK B-mode to the large-gamma treatment, DC convention and small-angle floor

BB/EE at l = 60 / 300 / 1500 and the 50-1500 median (`results.json: step4_rows`; full table with f_BB in `tables.md` section 4). "tail(>1 deg)" replaces the 14 rows with gamma > 60' (the region the paper itself calls unconverged); the power law is fitted on 15'-60' (xi_+ ~ gamma^-1.67, xi_- ~ gamma^-0.87).

| variant | DC | l=60 | l=300 | l=1500 | median |
|---|---|---:|---:|---:|---:|
| FK deployed (83 deg) | on | +0.951 | +0.618 | +0.420 | +0.618 |
| FK deployed (83 deg) | off | +0.946 | +0.614 | +0.426 | +0.614 |
| FK truncated 40 deg | on / off | +1.167 / +0.855 | +0.634 / +0.620 | +0.429 / +0.417 | +0.634 / +0.620 |
| FK truncated 20 deg | on / off | +0.968 / +0.925 | +0.630 / +0.601 | +0.426 / +0.416 | +0.630 / +0.618 |
| FK truncated 10 deg | on / off | +0.740 / +0.757 | +0.610 / +0.604 | +0.416 / +0.394 | +0.610 / +0.604 |
| FK truncated 5 deg | on / off | +0.945 / +0.812 | +0.644 / +0.616 | +0.431 / +0.386 | +0.644 / +0.616 |
| FK truncated 2 deg | on / off | +0.999 / +0.994 | +0.709 / +0.654 | +0.462 / +0.463 | +0.709 / +0.625 |
| FK truncated 1 deg | on / off | +1.000 / +1.000 | +0.946 / +0.743 | +0.543 / +0.422 | +0.946 / +0.743 |
| FK tail(>1 deg) -> zero | on / off | +1.000 / +1.000 | +0.650 / +0.650 | +0.410 / +0.410 | +0.650 |
| FK tail(>1 deg) -> const | on / off | +1.000 / +1.057 | +0.949 / +0.589 | +0.544 / +0.384 | +0.949 / +0.408 |
| FK tail(>1 deg) -> power law | on / off | +0.536 / +0.534 | +0.655 / +0.645 | +0.422 / +0.402 | +0.598 / +0.588 |
| FK extended below 0.5' (floor removed) | on | +0.951 | +0.620 | +0.455 | +0.620 |
| O0 deployed (control) | on / off | +0.000 / +0.000 | -0.002 / -0.002 | -0.040 / -0.041 | -0.002 |
| O0 truncated 10 deg | on | +0.100 | +0.023 | -0.022 | +0.023 |
| O0 truncated 2 deg | on | +0.994 | +0.317 | +0.222 | +0.322 |
| O0 truncated 1 deg | on | +1.000 | +0.898 | +0.538 | +0.898 |
| O0 tail(>1 deg) -> zero | on | +0.998 | -0.030 | -0.083 | +0.093 |
| O0 tail(>1 deg) -> const | on | +1.000 | +0.906 | +0.527 | +0.906 |
| O0 extended below 0.5' | on | +0.001 | +0.000 | +0.001 | +0.000 |

Which of the paper's values survive:

* **l = 1500, 0.42: survives** every treatment (range 0.38-0.54; 0.455 with the small-angle floor removed; 0.41 with the unconverged tail zeroed). It is set by the deployed FK 2PCFs at gamma <~ 1 deg, i.e. inside the paper's converged region. It is therefore a genuine property of the deployed product and, by the theorem, a defect of the vertex's spin channels rather than of the transform.
* **l = 300 (band interior), 0.62: mostly survives** (0.59-0.71 for the physically motivated treatments; 0.95 only when the tail is frozen to a constant, which the O0 control shows is an artifact: pure-E O0 gives 0.91 under the same surgery).
* **l = 60, 0.95: does not survive.** Range 0.53-1.17 across treatments; and the O0 control is decisive: a pure-E field whose 2PCF is cut at 1 deg (or has its tail zeroed or frozen) returns BB/EE = 1.000 at l=60 with this transform, because d^60_{2,-2}(theta) ~ (60 theta)^4/384 has no support below ~1 deg while d^60_{2,2} ~ 1 there. The FK value at l=60 is thus determined by xi^FK at 1-10 deg, which the paper itself declares unconverged. The "0.95 at l=60" and the "near-equal polarizations at the large-angle end" sentence are statements about the unconverged tail.
* The DC convention is immaterial for l >= 300 (<= 0.01 shifts) and matters only below.

## 5. Vertex-version sensitivity and the small-gamma exponent

| quantity | FK (Aug, cut15360 permfix) | FK_june (cut1000) |
|---|---:|---:|
| BB/EE at l = 60 / 300 / 1500 | 0.951 / 0.618 / 0.420 | 1.042 / 1.806 / 1.147 |
| BB/EE median 50-1500 | 0.618 | 1.147 |
| D/S at l = 60 / 300 / 1500 | 0.025 / 0.236 / 0.408 | -0.021 / -0.287 / -0.069 |
| xi_kk(0.5') | +1.9480e-5 | +4.2877e-6 (ratio 0.220) |
| xi_-/xi_+ at 0.5' / 2' / 8.5' | 8.6e-4 / 3.9e-2 / 0.113 | -2e-11 / -4e-10 / -7e-9 |
| fitted exponent of xi_- on 0.5'-2' | 2.44 (ratio: 2.77) | 2.0000 (ratio: 2.0000) |

For reference, the pure-E terms on the same 0.5'-2' window: O0 xi_- ~ gamma^3.85 (ratio 3.98), FF ~ gamma^3.71 (3.85). So (i) the E/B split moved by O(1) between vertex versions (BB = EE exactly in June, since xi_11 = xi_22 to 1e-11; BB = 0.42-0.95 EE in August) while the theorem requires BB = 0 in both: the split is a property of the spin-channel construction of zeta, not of the physics; (ii) the paper's "suppressed as gamma^4 below an arcminute" is not what the deployed FK does (exponent 2.4-2.8, shallower than O0 because its vertex table extends to ell_max = 15360 vs O0's 5000) and IS what the pure-E O0 does (3.85-3.98, the band-limited d^l_{2,-2} kernel). The gamma^4 clause is a kernel property, as the theorem states.

## 6. The pure-E prediction of xi_-^FK from xi_+^FK

Two routes, both applied identically to O0 (control) and FK:

* **Curved-sky**: S_l = T_{2,2}[xi_+] on every integer l in [2, 3000] (same quadrature, DC-subtracted), extended beyond 3000 by the power law fitted on [1000, 3000] (slopes: O0 -2.53, FK -2.37), cos^2-tapered over the top 25% of [2, l_sum], and forward-summed xi_-^pred(gamma) = sum_l (2l+1)/(4 pi) S_l w_l d^l_{2,-2}(gamma) for l_sum in {3000, 6000, 12000, 24000}. Compared with the DC-subtracted deployed xi_-. Limitation: the sub-arcminute prediction depends on the l > 3000 extension; the O0 control's own band limit is 5000, so l_sum = 6000 is its fair setting; FK's vertex ell_max is 15360, so l_sum = 12000-24000.
* **Flat-sky** (Crittenden+2002 / Schneider+2002, exact for a pure-E field at small angles): xi_-(t) = xi_+(t) + int_0^t (dt' t'/t^2) xi_+(t') [4 - 12 (t'/t)^2]. Needs xi_+ only at t' <= t (the converged FK region for t <= 1 deg), is insensitive to a constant (maps to 0), and needs xi_+ below 0.5', extended either as a constant or as the 0.5'-1.3' power law (both shown; their spread is the systematic). Compared with the raw deployed xi_-.

xi_-^pred / xi_-^deployed (`results.json: step6`, `tables.md` section 6):

| gamma | O0 curved l_sum=6000 | O0 flat const / power | FK curved 12000 / 24000 | FK flat const / power |
|---:|---:|---:|---:|---:|
| 1' | 1.25 | (band-limit region) | 0.69 / 3.35 | 0.02 / 9.5 |
| 2' | 1.14 | 0.52 / 1.11 | 1.52 / 1.54 | 2.03 / 2.96 |
| 5' | 0.80 | 0.977 / 0.985 | 1.82 / 1.83 | 2.35 / 2.48 |
| 10' | 0.92 | 0.988 / 0.990 | 2.52 / 2.52 | 2.70 / 2.73 |
| 30' | 0.993 | 0.998 / 0.998 | 3.83 / 3.83 | 3.82 / 3.83 |
| 60' | 0.999 | 1.000 / 1.000 | 5.08 / 5.08 | 5.00 / 5.00 |

The control works: for O0 both routes return 0.98-1.00 for gamma >= 5' (curved-sky 0.92 at 10', 0.99 at 30'); below ~3' the deployed O0 xi_- is in its gamma^4 band-limit regime and the prediction is dominated by the sub-0.5' / l > 3000 extension (flagged, not used). For FK, both routes agree to 7% for gamma >= 5' and say the deployed xi_-^FK is 1.8-2.5x too small at 5'-10', 3.8x at 30', 5.0x at 60'. Equivalently: the pure-E xi_- implied by the deployed xi_+^FK peaks at ~1.4e-6 near 10'-20', while the deployed xi_-^FK peaks at 5.5e-7 near 3'-5' (`fig_pureE_prediction.png`, middle column). Beyond 1 deg the FK ratio oscillates (the unconverged tail; not quoted). For FK_june the prediction is 1e5-1e7 times the deployed xi_- (which is zero): that vertex had no difference channel at all.

This is the products-level content of the theorem: for a pure-E linear response the FK difference channel <Phi Psi_+ Psi_+> - <Phi Psi_x Psi_x> is fixed by the sum channel, and the deployed difference channel is 2.5-5x (10'-60') below that requirement. Which channel is wrong cannot be decided from the products alone; the 2026-08 audit's findings (the deployed zeta_Bmod = zeta_TTT with the pair phase dropped; the 23% permutation ambiguity on zeta_TPP) are the natural suspects.

## 7. kappa-E cross

C_l^{kappa E} from -xi_01 with d^l_{2,0}: FK/O0 = 6.5e-3 (l=60), 1.03e-2 (300), 1.36e-2 (1500) (`results.json: step7_CkE_FK_over_O0`). The theorem's C^{kappa B}^FK = 0 needs <kappa gamma_x> in the rotated frame, which the sweeps' xi_02 (a parity null, identically 0) does not provide; not testable here.

## Caveats

* ell = 300 is not on the paper's ELL grid; it was added as an extra integer multipole of the same transform (paper-grid neighbours for FK BB/EE: 0.649 at 270, 0.609 at 335).
* The transform's small-angle floor (missing [0, 0.5']) biases S but not D: O0's D/S is 1.08 at l=1500 (BB/EE -0.04). For FK the same correction moves BB/EE at 1500 from 0.420 to 0.455, i.e. away from zero; it cannot explain the FK B-mode.
* The pure-E prediction below ~3' depends on the sub-0.5' / high-ell extension (both routes); numbers are quoted from 5' up, where the O0 control is at 0.98-1.00.
* The curved-sky prediction for O0 uses a power-law extension without the table's 5000 cutoff, so its l_sum = 12000/24000 rows over-predict xi_-^O0 at 1'-2'; l_sum = 6000 is the fair O0 control and is the one quoted.
* Nothing here tests the vertex directly; it tests the deployed 2PCF products for internal E-consistency. The MC validation in the paper shares the zeta table and cannot see this.
* FK_june's large-gamma behaviour (xi_+ flat to 1 deg then falling) is that of the cut1000 table; its E/B numbers are reported only as the vertex-version comparison.

## Files

`n2_products_eb.py` (main, default python), `n2_crosscheck_analysis3.py` (PyCCL interpreter, imports analysis3), `results.npz`, `results.json`, `tables.md` (auto-generated full tables), `crosscheck_analysis3.npz`, `fig_2pcf.png`, `fig_eb_DS.png`, `fig_robustness.png`, `fig_pureE_prediction.png`, `fig_dense_DS.png` — all under `/Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/n2_products/`. Nothing was written into the STF_lensing or SFT-WL-B trees; nothing committed.