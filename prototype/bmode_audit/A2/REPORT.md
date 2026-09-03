
# A2 audit: the B-mode programme repository `/Users/zzhang/projects/SFT-WL-B/`

Read-only audit performed 2026-09-03. Nothing in SFT-WL-B, the parent tree, or `~/Workspace/*` was modified. No files written.

## 0. Git state

`/Users/zzhang/projects/SFT-WL-B`, branch `main`, HEAD `c8f28df`:

```
c8f28df docs(canoes): the D-OP-02 acceptance criterion, set and validated; D-CN-07
caf7817 docs: D-OP-02 -- operator authorises the n_phi rebuild; F11 closes, F7 stays held
782d94b D-RZ-57: the review is closed
92aa4e7 D-RZ-56: the soft tower is unusable at any separation; one claim withdrawn
b395fca D-RZ-55: no rebuild needed -- the tower's bound was the quadrature, not the data
69e3861 docs(canoes): the exact branch's share bounds D-CN-05's open item; D-CN-06
e290c4c D-RZ-54: I1 closed -- route (i) is sign-wrong, and the exact m=4 tower is in the package
d819172 docs: two of the three write, not three, and anchor the warning to code
590163a docs: the REPRODUCE.md hazard is three commands, and the worst is not a check_*
263bce5 docs(canoes): audit closed -- the deployed vertex is ~99.99 percent Limber; D-CN-05
2b61da7 docs: correct the attribution again -- RESULTS section 17 is a third session's
19a42ff docs(wp0): the evidence D-W0-21 cites, and one attribution correction
b5c99b7 docs: tell WP-0 that D-W0-21 was swept into a rezeta commit
e724feb docs(rezeta): renumber the D5 write-back to section 18
3183a42 D-RZ-53: D5 closed -- the trunk's tidal readout was the superseded one
```

`git status --short`: ` M proposal/.quarto/xref/d9c3752e`; untracked `docs/handoff_canoes_bmod_phase.md`, `docs/handoff_rezeta_full_review.md`, `proposal/.gitignore`.

Workspace repos: `~/Workspace/SFT-Sachs`, `~/Workspace/stf-transfer`, `~/Workspace/gevlens` are **not** git repositories. `~/Workspace/gevolution-1.3` is (HEAD `c18fab1`, `makefile` and `settings.ini` modified, untracked `.claude/`, `CLAUDE.md`); its `.git` directory is **109 GB** (the working tree is ~100 MB), so something large was committed there at some point; do not clone it casually.

## 1. What exists, and what each work package concluded

The plan (`docs/plan_bmode_programme.md`, 2026-08-31, amended through 2026-09-02) builds a Letter ("The cosmic-shear B-mode is a cosmological observable", `letter/main.tex`, 5 pages, `\hold{H3}` markers still in) and an ERC proposal (`proposal/proposal.qmd`) on eight "established facts" F1-F8 taken from the paper, of which **F4** (B/E = 0.95 at l=60 -> 0.42 at l=1500, `[P] conclusion.tex:68-71`) and **F7** (FK = 1.3-1.7% of Order-0 in xi_+ over 2'-12' at z_s = 5; 0.9-1.1% at z_s = 1) carry every B-mode amplitude.

| Package | Dir / decisions | What it concluded | Depends on F4? |
|---|---|---|---|
| WP-0 (table re-derivation) | `analysis/wp0/FINDINGS.md`, D-17..19, D-W0-20/21 | Vendored the permutation-aware `cut15360` vertex table (7/7 upstream tests pass). W0-1: the gamma^4 slope is the `d^{l3}_{2,-2}` kinematics (3.90 at 0.53', 4.00 in the coincidence limit). **W0-2: the table is self-inconsistent under leg relabelling of the fully symmetric zeta_TTT by 8-43% over 2'-12'** (later traced to `n_phi = 64`, D-CN-04). W0-3: no single crossover angle. W0-5/D-W0-20: the `+4.29e-6` anchor is the pre-fix cosine-grid artifact at l_max ~ 1000; benchmark restated as 2.31% of Order-0 = 1.95e-5 at 0.5', z_s = 5. D-W0-21: upstream reorganised 2026-09-02; a post-fix FK sweep (`..._FK_cut15360_permfix`) exists upstream but is not vendored. | No (vertex-level; feeds C_kk and C_EE^FK too) |
| WP-A (inventory, misabsorption, bias, detectability) | `analysis/wpa/FINDINGS_A1/A3/A4.md`, D-A-01..10 | A-1: Krause & Hirata 2010 ceiling on the SUM of known cosmological B-modes (3.3e-4 for z<=1, 2.0e-3 for z<=3); FK f_BB = 4.4e-3..8.3e-3 at l=60 is **8-16x** that ceiling. A-2 figure. **A-1 section 7 / D-A-03: F4 could NOT be reproduced from the deployed 2PCF sweep through the paper's own curved-sky transform** (B/E = 1.04 at l=60, 1.15 at l=1500, four sign flips; a spurious additive Order-0 BB of 1e-11); A-2 therefore copies the published F4/F7. A-3: misabsorption bounded by S/N, Delta p/sigma a few tenths (up to 1.7 sigma). A-4: idealised S/N 12.2 (LSST Y10), 12.1 (Euclid), 70% below l=100. | **Yes, entirely** (f_BB = a r/(1+r) C_EE^O0 with r = F4, a = F7) |
| WP-B-1 (degeneracy direction) | `analysis/wpb1/FINDINGS_B11..B16.md`, `SUMMARY_WPB1.md`, D-B1-* | Exponent theorems (2,2)/(4,4)/(3,4); FF exactly parallel to Order-0; **B12-3: "the rotation vanishes iff C^BB is pure FF"; B12-4: C^BB is 99% FK** (from F4 x F7 vs F5 x FF amplitude). Cov(EE,BB) = 0 exactly (parity). l_h = 972, window l<=300, 1-cos^2 theta = 0.20-0.30, 16 deg / 22 deg. | Direction vectors u_BB and every angle assume C_BB = C_BB^FK; algebra survives, application falls |
| WP-B-2 (Stage-IV Fisher) | `analysis/wpb2/FINDINGS_B2.md`, D-B2-01..08 | With C_BB^FK = sqrt(a_i a_j) r/(1+r) C_GG (A-3's model): sigma(sigma8) improves 6-7%, sigma(Omega_m) ~6%, sigma(alpha_s) only 1-2%; the earlier 42-46% running gain was a toy-baseline artefact (withdrawn). | **Yes** (`b2_model.py:260`, `b2_fisher.py:24`) |
| WP-B-3 (cost vs 3PCF) | `analysis/wpb3/FINDINGS_B3.md`, D-B3-01..05 | 120 band powers vs 2250 (i3PCF) vs millions (3PCF); same parameter yield on the same multipoles; B-mode holds 3% of the squeezed S/N^2 once hard legs go to 1500. | Dimension counts no; "same yield", S/N 11, 3% yes |
| WP-C Route A (rezeta, external) | `analysis/rezeta_driving_fields/RESULTS.md`, D-RZ-20..57 | Six spin channels built; exact m=4 tower; **D-RZ-44: the F-vertex contraction is `G_00 = -2(TTT+Bmod)`, `xi_+ = -4 Bmod`, `xi_- = -4 TPP` and nothing else, so E=B at leading order is a derived identity from zeta_TPP = 0**; prefactor-free `|xi_-/xi_+|^FK` tables (see section 4); D-RZ-47: FK kk at 0.5' = 1.75e-5 (table convention) / 1.29e-5 (physical phase); **the deployed zeta_Bmod carries no pair phase** (Bmod/TTT should be 0.48 at 0.5'). D-RZ-56/57: the response-sector tower is unusable; several intervals corrected. | Vertex-level; not F4 |
| canoes stream | D-CN-01..07 | D-CN-01: pair-phase bug confirmed and fixed in canoes (blast radius: Bmod only). D-CN-04: the probe was on the wrong shell; the two engines agree to <1%; the real defect is `n_phi = 64` (slot-1 zeta_TTT low by 4.2/7.7/15.7/38% at 0.5/1.94/5/12.12'). D-CN-05/06: the deployed vertex is 99.99% Limber; the exact low-l branch is unvalidated but contributes <0.44% inside 0.5'-12'. **D-OP-02: operator authorised the n_phi rebuild; F11 closes at ~1.53%; F7 stays held.** D-CN-07: acceptance criterion validated (rebuild passes 270/270, current table fails 864/864); rebuild not yet executed in the parent tree. | Not F4, but F4/F7 are held on these |
| WP-C Route B (this repo) | `analysis/wpc_routeb/FINDINGS_CB2.md`, `analysis/wpc_cb3/FINDINGS_CB3.md`, D-CB-01..07 | C-B.1: Takahashi 2017 chosen; **R4 fired: no public simulation has metric output**. C-B.2: B_kappa gate **PASS** on zs16 (1.113/1.028/1.099 for equilateral/isosceles/open-squeezed, 200<=l<=1400; residual in BiHalofit); F31: three-point Limber error ~20x the two-point one. C-B.3: gevolution transfer validation **complete** (see section 5); Phi00 sign settled (D-13 closed); F32 shell-thickness law. **C-B.4 (collapsed zeta) and the spin-2 three-point estimator (R-CB3-3) never started.** | No |
| WP-X (four-way cross-validation) | not started | -- | -- |
| Letter | `letter/main.tex`, `NUMBERS.md`, `REVIEW_SELF.md`, D-LT-01..03 | H1, H2 closed 2026-09-02; **H3 (F4, F7, 8-16x, S/N ceiling, A-3 shifts) open**, awaiting canoes verification and the F4 second path. | -- |
| Proposal | `proposal/proposal.qmd`, D-PR-01 | Measurement framing for O2; Table `tbl-prep`. | rows 2-6, 8, 9 yes |

Handoffs written but never executed: `docs/handoff_f4_second_path.md` (build a second production path for F4 from rezeta's fold, with an Order-0 gate `|C_BB^O0|/C_EE^O0 < 1e-4` over 50<=l<=1500), `docs/handoff_rezeta_full_review.md`, `docs/handoff_canoes_bmod_phase.md` (Part A executed as D-CN-01..07; Part B partially).

## 2. Inventory: artifact -> what it is -> reusable for an N-body B-mode test? -> depends on F4?

### Data

| Artifact | What | Reusable? | F4? |
|---|---|---|---|
| `data/allskymap_nres12r000.zs{10,16,38}.mag.dat` | Takahashi 2017 full-sky ray-tracing, Nside 4096, RING, four float32[npix] fields kappa, gamma1, gamma2, omega; z_s = 0.574, 1.033, 5.342; 3,221,225,524 B each; SHA-256 in `data/PROVENANCE.md` (2b76e0c0..., 42f92d09..., 95bd51ea...) | **Yes, primary.** All-order ray-traced shear with deflection; usable to l ~ 1400 (5% accuracy) / 3000 (support). gamma1/gamma2 -> `map2alm_spin` -> C_BB directly; omega -> image rotation (post-Born cross-check). Only kappa has ever been read. | No |
| `data/gevolution/n{128,256,512}/` | cb3 runs: box 1024 Mpc/h, Takahashi cosmology, seed 42, z_in 100, GR; light cone 0 = full-sky **phi only** to 2450 Mpc/h (z~1.03) at Nside <= 256; light cones 1-5 = phi, chi, B on 33 Mpc/h windows at r0 = 304/608/1200/1800/2400 Mpc/h, Nside <= 1024; 15 phi/T00 snapshots; P(k) at 8 redshifts; lcmap kappa/potential/ISW/Shapiro FITS (Nside 256). n512 raw `.map`/`.h5` **deleted** (`.npy` snapshot caches and `cb3_window_*.npz` kept); n128/n256 fully recomputable. 64 GB on disk. | Partially. The phi cone can drive a Level-1 trace to z_s ~ 1 at l <~ 500 but lacks chi (Psi = phi - chi) and stops at 2450 Mpc/h; the metric windows are too thin for a full trace. A new full-sky run with phi (+chi) on every shell to the source is needed (n512 costs 19 min on 16 cores). | No |
| `data/gevolution/cb3_kappa_cl.npz` | kappa C_ell of the n256/n512 cones vs halofit and vs Takahashi zs16 (0.96 over l=20-100 at dx=2) | Cross-check only; one 1024 box is 13-22% low at k<=0.05 | No |
| `data/gevolution/bin/gevolution_cb3`, `class_tk_takahashi2017_z100.dat` | gevolution binary compiled with `-DHAVE_HEALPIX -DMAX_INTERSECTS=128`; CAMB-built CLASS-format IC table for the Takahashi cosmology (validated to 8e-4 vs gevolution's CLASS table) | **Yes** | No |
| `vendor/analysis3_cl/` | the paper's June 2026 2PCF sweeps (O0, FF, FK cut1000) + `plot_analysis3_cl_decomposition.py` | O0/FF references for comparison; the FK sweep is the pre-fix artifact (flat, xi_- = 0). Known: the curved-sky transform of a 2PCF truncated at 83 deg leaves an additive BB floor of 1e-11 (D-A-03). | -- |
| `vendor/stf_lensing_mirror/` | permutation-aware kappa3 callable + `table_permclosed.npz` (2142 triples x 16 shells, six channels, l_max 15360) | For vertex-level checks only | -- |
| `vendor/bihalofit_ref/`, `cutoff_study/`, `cutoff_ladder/`, `consumer_pk/` | BiHalofit reference P(k), cutoff ladders, consumer P(k) | For modelling kappa bispectrum / FF, not for the B-mode measurement | -- |

### Code in SFT-WL-B

| Artifact | What | Reusable? | F4? |
|---|---|---|---|
| `analysis/wpc_routeb/cb2_takahashi_io.py` | `read_header`, `read_field(path, "kappa"|"gamma1"|"gamma2"|"omega")` (memmap, 0.8 GB per field, Fortran records, RING), `write_synthetic` | **Yes** | No |
| `analysis/wpc_routeb/cb2_cosmology.py` | `make_cosmo()` (pyccl, `TAKAHASHI2017 = Omega_c 0.233, Omega_b 0.046, h 0.7, sigma8 0.82, n_s 0.97`), `TAKAHASHI_PLANES`, chi(z), growth, P_lin/P_nl, single-plane lensing kernel in h units | **Yes** (PyCCL env only) | No |
| `analysis/wpc_routeb/cb2_estimator.py` | binned full-sky **scalar** bispectrum estimator (Bucher et al. 2016 h^2 weights), Gaussian variance | Partially (scalar only; a spin-2/E-B version does not exist, R-CB3-3) | No |
| `cb2_bdelta.py`, `cb2_limber.py`, `cb2_exact.py`, `cb2_bkappa.py` | BiHalofit port (pinned to C++ ref at 1.3e-4), Limber and exact (Komatsu-Spergel) B_kappa projections, gate driver | For kappa-bispectrum / FF / post-Born modelling; not needed for a BB measurement | No |
| `analysis/wpc_cb3/cb3_gevolution_io.py` | `MetricLightcone` (all `.map` shells by (field, distance), pixbatch unpack via `sft_sachs.gevolution_io`, linear time interpolation between the two cycles that wrote a shell), `read_info`, `read_background` (correct H/H0 handling), `rotation_matrix` (gevolution's HEALPix-frame rotation), Gadget-2 light-cone reader, `project_particles` | **Yes** for building `DrivingField` inputs to sachsray from gevolution shells | No |
| `analysis/wpc_cb3/cb3_two_routes.py`, `cb3_snapshot_io.py` | native Phi00/Psi0 alms per shell through `stf_transfer`'s catalogue (frozen potential, radial finite differences) and the Newtonian route `Phi00 = -A_tr delta`, `Psi0_lm = +A_tr sqrt((L2-2)/L2) delta_lm`; T00 snapshot trilinear + quadratic-in-time interpolation | **Yes** (validated: ratio 0.99-1.00 to l ~ Nside) | No |
| `cb3_settings.py`, `settings_cb3_n*.ini`, `cb3_tk_from_camb.py` | settings generator and IC table builder | **Yes** as templates | No |
| `analysis/wpa/a2_cl_extract.py`, `make_a2_figure.py` | 2PCF -> C_ell via the vendored curved-sky transform; the A-2 figure with `r_of_ell` log-linear between F4 anchors | Transform: with the D-A-03 caveat; figure: F4 | Yes |
| `analysis/wpa/a1_inventory.py` | `FK_A`, `FK_R = {60: 0.95, 1500: 0.42}`, `KH_CEILING = {1.0: 1/3000, 3.0: 1/500}`, source table | The ceiling and source table are the comparator for any measured BB | FK rows yes |
| `analysis/wpa/a3_*.py`, `a4_detectability.py`; `analysis/wpb1/*.py`; `analysis/wpb2/b2_*.py`; `analysis/wpb3/*.py` | misabsorption, bias, S/N, response vectors, Fisher, cost table | Machinery could be re-pointed at a measured C_BB, but every current result takes `C_BB = a r/(1+r) C_EE^O0` | Yes |
| `tests/test_wpa_a1.py`, `tests/test_letter_figures.py` | pin the F4-derived f_BB rows and the A-1 reproduction FAILURE (`c["reproduced"] is False`, sign flips > 0) | -- | Yes |

### External (`~/Workspace`, import-only)

| Artifact | What | Reusable? | Notes |
|---|---|---|---|
| `~/Workspace/SFT-Sachs/src/sft_sachs/raytrace/` | **Another Sachs solver**: numpy symplectic multi-plane integrator of the matrix Jacobi equation `D'' = -T D` per HEALPix pixel (`jacobi.py`, drift-kick with column-integrated `U_k = T_k dchi_k`), `tidal.py` (`T = [[Phi00+Re Psi0, Im Psi0],[Im Psi0, Phi00-Re Psi0]]`), `optical_scalars.py` (Riccati cross-check `S' = -T - S^2`), `observables.py` (kappa = 1 - tr A/2, gamma1 = (A22-A11)/2, gamma2 = -(A12+A21)/2, rotation = (A12-A21)/2, magnification), `spectra.py` (kk, EE, BB, kE, kB, EB via `hp.map2alm_spin`), `radial_stack.py` (observer-to-source shell stack with pixbatch unpack, adaptive-Nside unification), `pipeline.py` (`trace_rays`, `born_amplification`, `compare_born_vs_exact`), `reduce.py` (per-realisation and ensemble C_l), `plotting.py` (Fig-12 style). 14 test files incl. analytic nonlinear checks (`1 - sinc`, `sin/sinh`). **Level 1 only** (sources on the unperturbed ray; no deflection). Same sign conventions for A as sachsray but **Phi00 = +(1/2) R k k** in `driving_fields.py` (negative of the paper's/stf-transfer's, D-CB-06); a factor-2 Psi0 convention bug was fixed 2026-06-16. Legacy `gevolution_io.read_shells` does NOT unpack pixbatch. | **Yes, as an independent cross-check of sachsray** (numpy multi-plane vs JAX/diffrax), and as the source of the existing measurements in section 3 | not git |
| `~/Workspace/SFT-Sachs/analysis_exact_raytrace/` | June 2026 exact-ray-trace campaign: scripts (`z5_fk_cross.py` pure-FK cross estimator, `cellscan_cumulant.py`, `compare_resolution_excess.py`, `run_zs1_ensemble.py`), reduced npz data, two handovers | Data reusable (section 3); raw gevolution ensembles deleted | |
| `~/Workspace/stf-transfer/src/stf_transfer/` | per-multipole transfer catalogue metric -> Phi00, Psi0 alms (mirrors the paper's `driving_fields_harmonics.wl`); `frozen_potential` v1 adequate on light-cone shells (D-CB-07 result 5); B sector SUPERSEDED; `load_background` misreads gevolution 1.3's H/H0 column (R-CB3-2) | Yes via the cb3 wrappers | not git |
| `~/Workspace/gevlens/src/gevlens/` (`io.py`, `maps.py`, `shear.py`) | lcmap kappa FITS -> gamma1, gamma2 by SHT (Born) | Born E-mode reference only | not git |
| `~/Workspace/gevolution-1.3/` | compiled gevolution + lcmap (HEALPix on); settings for the June campaign survive: `settings_z5big_seed{42..49}.ini` (box 1920, Ngrid 256), `settings_z5_n512_seed42.ini`, `settings_zs1_ens_seed42.ini`, `settings_cellscan_n{128,256,512b512}.ini`; run outputs (`output_z5_big_seed*`, `output_zs1_*_ens`) **gone**; `.git` 109 GB | Binary and settings yes | |

## 3. Existing measurements of C_BB or the rotation

**Inside SFT-WL-B: none.** `grep -rn map2alm_spin analysis/` returns nothing; `gamma1|gamma2|"omega"` appear only in `cb2_takahashi_io.py` (definition) and `cb1_selection.md` (description). The `alm2cl(b, ...)` in `analysis/wpc_cb3/cb3_compare.py:98` is the E/B split of the scalar-coefficient Psi0 on a shell (a transfer test), not a shear B-mode. The only B-mode quantities in the repo are the F4-derived FK rows and the FF value BB/EE = 0.03206 re-derived from the vendored June sweeps (`FINDINGS_A1.md` section 6).

**Outside, in `~/Workspace/SFT-Sachs/analysis_exact_raytrace/data/` (June 2026, Level-1 exact Sachs ray-trace of gevolution light cones):**

1. `raytrace_cl_z5_ensemble.npz`: 8 seeds, box 1920 Mpc/h, Ngrid 256 (cell 7.5 Mpc/h), Nside 256, chi_s = 5378 Mpc/h (z_s = 5), l <= 384. Computed this session:

   | l | C_BB/C_EE (exact) | (kk_exact - kk_Born)/kk_Born | C_BB/C_kk^Born |
   |---|---|---|---|
   | 8 | 2.34e-5 +- 3.1e-6 | +5.3e-3 | 9.1e-5 |
   | 16 | 2.00e-5 +- 2.7e-6 | -6.4e-4 | 7.9e-5 |
   | 32 | 1.91e-5 +- 1.5e-6 | -2.6e-4 | 7.6e-5 |
   | 64 | 2.01e-5 +- 5.9e-7 | +4.5e-4 | 8.0e-5 |
   | 128 | 2.47e-5 +- 8.4e-7 | +1.9e-4 | 9.9e-5 |
   | 200 | 2.46e-5 +- 8.0e-7 | +7.3e-5 | 9.8e-5 |

   For comparison the Letter's FK f_BB at z_s = 5 is 6.3e-3..8.3e-3 at l = 60. The ray-traced ratio is ~300x smaller (caveats: Level-1, cell 7.5 Mpc/h, the file predates the 2026-06-16 Psi0 factor-2 fix).

2. `fkcross_z5s{42..49}.npz` (`z5_fk_cross.py`): the pure-FK kk estimator `<kappa_Born x (kappa_exact - kappa_Born)>/C_Born` over 8 seeds: -4.9e-4 +- 1.9e-3 (l=4), +8.6e-4 +- 7.9e-4 (l=8), +1.3e-5 +- 4.1e-4 (l=16), +1.7e-5 +- 2.8e-4 (l=32), +1.2e-4 +- 9.4e-5 (l=64), -8.5e-5 +- 1.2e-4 (l=128), -1.2e-4 +- 2.0e-4 (l=256). Consistent with zero.

3. z_s = 1 ensembles (32 seeds Ngrid 256, 8 seeds Ngrid 512; `figures/zs1_ensemble_bmode.pdf`, `zs1_bmode_resolution.pdf`): B-mode detected at 13-77 sigma, red/low-l shape, identified by that campaign as **FF (post-Born-like lens-lens), not FK** (`HANDOVER_FK_leakage_check.md`). Raw runs deleted.

4. `HANDOVER_FK_leakage_check.md` (T1, synthetic local-fNL potential through the real pipeline, N = 24): kk FK lift reproduced (+1.5e-6 at l=4); **shear sector: EE+BB FK ~ 1% of kk FK, EE-BB FK small, kE FK non-zero at 5-40% (later read as noise)**. The handover's author attributed the absence of a shear-sector FK to a Level-1 limitation (`Sigma @ Sigma = |Psi0|^2 I` along one line of sight, so two shear legs feed only kappa) and deferred to "Level-2 deflection". That attribution is a hypothesis of that agent, not a demonstration. Under the theorem in the brief, "no FK B-mode" is the expected result for BB regardless of level.

## 4. What rests on F4/F7, exactly (from `letter/NUMBERS.md`)

If C_BB^FK = 0 identically, `r(ell) -> 0` and every quantity below vanishes or reverts to the FF/known-ceiling level:

**Abstract**: "0.95 ... 0.42"; "exceeds the sum of every known cosmological B-mode by 8 to 16"; "Stage-IV S/N about 12"; "seventy percent below l=100"; "a few tenths of a standard deviation"; "running response opposite in sign, five to ten times larger"; "amplitude parameters tighten by six to seven percent"; "almost entirely through direction". Survives: "the linear scalar signal is purely E"; "Gaussian covariance with the E-mode is exactly zero" (trivially).

**Setup**: F1 (no Order-0 BB) survives; "FK (1,1)" as *the* mechanism of the B-mode falls.

**An irreducible scalar B-mode** (`main.tex:110-135`): gamma^4 clause (F3: the exponent is the `d^l_{2,-2}` kernel, survives as kinematics, its reading as a B-mode falls); 0.95 -> 0.42 (F4, `\hold{H3}` at line 123); 1.3-1.7% / 0.9-1.1% (F7, hold at 128: this is xi_+ = EE+BB and survives as an E-mode statement); f_BB (4.4-8.3)e-3 -> (2.7-5.0)e-3 and the z_s = 1 row; lower-bound tag (F11, kk); "13 to 16 times at l=60, 8 to 10 at l=1500" (hold at 163); FF thirty times below its own E-mode (F5, reproduced, **survives**); FF EB residual (F6, **survives**); Fig. 1 FK band falls, ceilings and FF curve survive.

**Not a null test**: S/N 12.2 / 12.1 / 10.2 (hold at 182), Order-0 EE S/N 289 (survives), 70% / 26%, S/N 10.9, red-template overlap 0.97 / S = 11.8 / bound 12.2, "2.6x too large, seven times in power", CCD template 0.11 / 1.3, "divides by 3.9; 3.1 sigma remains", UNIONS S/N 0.96 / 0.22, TATT structural gate (survives), bias algebra theorems (survive), Delta S8/sigma = +0.35, Delta Omega_m/sigma = -0.46, Delta sigma8/sigma = +0.55, |Delta S8/sigma| <~ 0.15, 1.7 sigma, direct-channel +0.39/+0.15/+0.61/+0.28, +0.56 inside 60-300: **all fall**.

**A physically different observable**: exponent theorems and (2,2)/(4,4)/(3,4) (algebra survives; applied to C^BB it becomes (4,4) = FF, exactly parallel to Order-0, i.e. **B12-3's falsification condition is met**); u_BB[sigma8] = 4 (becomes FF's 4, no rotation); "C^BB is 99% FK"; Cov(EE,BB) = 0 (survives); l_h = 972, z_eff 0.85-1.0, k_h ~ 0.4, k_s; u_BB/u_EE = 2.00/-3.6/0.33/0.66; 1-cos^2 theta = 0.20-0.30, 0.32 -> 0.13, 0.022-0.056, 16 deg, caption 22 deg at l=240; lensing-only ~6%, Planck-prior 2-8%: **all fall**; condition number 1e3 (EE baseline) survives.

**What the shape measurement buys**: u_BB[alpha_s] in [1.62, 2.03], four to eight times, Stage-IV top-bin +1.9..+2.6, sigma(sigma8) 6-7%, sigma(Omega_m) ~6%, shares 0.71-0.85, sigma(alpha_s) 1-2%, four-prescription stability: **all fall**. Planck 2018 values and k_p survive.

**What the two-point route costs**: 120 / 2250 / millions / 1500x / analytic covariance / zero cross term survive as counting statements; "the same parameter yield", "three percent with hard legs to 1500", "the collapsed hard leg below every published cut" fall.

**Outlook**: 8-16x falls. **Figures**: Fig. 1 FK band; Fig. 2 panels A-D entirely.

**Proposal** (`proposal/proposal.qmd`): O1 paragraphs A1-A5 (lines ~405-452), O2/B1, and Table `tbl-prep` rows "FK against every catalogued B-mode source", "Idealised significance", "Misabsorption", "Induced parameter bias", "Misalignment of the E and B parameter directions", "Running of the spectral index", "Response-formulation shear difference" all rest on F4/F7; row 1 (2.3% normalisation benchmark) is C_kk^FK (F11) and does not; row 7 (cross-covariance zero) survives.

**Already-recorded fragility of F4** (independent of the theorem): D-A-03 (not reproducible from the deployed sweep; single production path, `R-A1-c`); `docs/handoff_f4_second_path.md` unexecuted; H3 holds from the pair-phase bug (D-RZ-47/D-CN-01: `xi_+^FK` high by up to 1.35 at 0.5', `xi_-` untouched, so B/E moves) and the `n_phi` defect (D-CN-04); D-OP-02 keeps F7 held until the rebuild lands.

**The direct quantitative lead already on disk.** D-RZ-44 proves the contraction `xi_+^FK = -4 zeta_Bmod`, `xi_-^FK = -4 zeta_TPP`, and gives prefactor-free `|xi_-/xi_+|^FK` = 5.2e-4, 6.4e-3, 4.0e-2, 6.3e-2, 6.3e-2, 4.7e-2, 3.3e-2, 2.2e-2, 1.6e-2 at 0.5, 1, 2, 3, 5, 8, 12, 20, 30 arcmin (R = 0.58); D-RZ-47 gives 1.4e-3 / 4.5e-2 / 8.9e-2 at 0.5' / 2' / 12' with the physical phase. A pure-E FK spectrum would require `T_{22}^{-1}[xi_+] = T_{2,-2}^{-1}[xi_-]`, i.e. xi_-/xi_+ of order 0.1-0.5 at these separations; the tabulated ratios are one to two orders below that, which is why the transform returns BB ~ EE. Nobody in the programme has run that consistency check on the fold; it is a table-only computation with `prototype/fk_mc/figure12.py`'s `wigner_d`/`build_curved_matrix`.

## 5. Simulation data: status and provenance

**Takahashi et al. 2017** (`data/PROVENANCE.md`): source `http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/sub1/nres12/`; realisation r000; downloaded 2026-09-02 (`curl -C - --retry 8`) after per-file operator approval (zs16 first, then zs38, zs10); size verified against HTTP HEAD and SHA-256 recorded (verified on arrival; not re-hashed by me). Layout: Fortran sequential records, header `int32 nside, int64 npix`, then kappa, gamma1, gamma2, omega as float32[npix] in records of <= 536,870,908 elements, RING. Validation done: kappa mean 5e-8, rms 0.0166, range -0.037..+0.76; C_ell^kappa / Limber-halofit = 0.969/0.986/0.996/0.986/0.972/0.956 in bands 50-100 ... 1200-1600 (zs16), consistent with Takahashi's stated accuracy and the damping model `(1 + (l/1.6 Nside)^2)^-1`; B_kappa gate passed. gamma1, gamma2, omega: never read, never validated. Cosmology: Omega_m 0.279, Omega_b 0.046, h 0.7, sigma8 0.82, n_s 0.97 (WMAP9-like); planes at 150 i Mpc/h; 38 planes exist upstream (only three on disk).

**gevolution cb3** (`data/gevolution/PROVENANCE.md`): source tree `~/Workspace/gevolution-1.3` at `c18fab1` (makefile paths + `-DHAVE_HEALPIX`); binary `data/gevolution/bin/gevolution_cb3` (SHA-256 37c90211..., flags `-O3 -std=c++11 -DFFT3D -DHDF5 -DPHINONLINEAR -DBENCHMARK -DEXACT_OUTPUT_REDSHIFTS -DHAVE_HEALPIX -DMAX_INTERSECTS=128`, double precision, Open MPI, LATfield2 at `~/Workspace/LATfield2`); `mpirun -np 16 -n 4 -m 4`; box 1024 Mpc/h; A_s 2.183791e-9 (sigma8 0.82), seed 42, sc1_crystal, one particle per cell, baryons blended, z_in 100. Runs: n128 (18 s, 3.1 GB), n256 (125 s, 14 GB), n512 (1164 s, 84 GB -> raw deleted, 41 GB kept). Outputs: **no particle light cone** (gevolution 1.3 writes each particle at most once per cycle, so a replicated full-sky cone loses 16-88% of particles beyond half a box, R-CB3-1); density from T00 snapshots. **It is not a ray-tracing light cone**: the only full-range cone is phi-only at Nside <= 256 to 2450 Mpc/h. Compared: native Phi00/Psi0 vs Newtonian from delta (ratio 0.99-1.00, corr 0.999, all 15 (window, dx) cells; departure collapses onto k dx, +8% at k dx ~ 1); sign +0.98..+1.01; equal-shell scalar three-point of Phi00 native/Newtonian 0.995-1.04 to l ~ 500 at dx = 2 (12-35% errors, one realisation); lcmap kappa vs Takahashi zs16: 1.09/0.96/0.96/0.94/0.88 at l 10-20/20-50/50-100/100-200/200-300 (dx = 2).

**June 2026 SFT-Sachs campaign**: settings survive in `~/Workspace/gevolution-1.3/settings_z5big_seed{42..49}.ini` (box 1920 Mpc/h, Ngrid 256, full-sky cone to ~5500 Mpc/h), `settings_z5_n512_seed42.ini`, `settings_zs1_ens_seed42.ini`, cell-scan settings; run directories deleted; reduced spectra in `~/Workspace/SFT-Sachs/analysis_exact_raytrace/data/` (8 x `raytrace_z5s*.npz`, 8 x `fkcross_z5s*.npz`, `raytrace_z5big.npz`, `raytrace_cell{1,2,4,8}.npz`, `cellscan_cumulant.npy`) and 12 + 5 synthetic `mc_raytrace_*` demo dirs in `~/Workspace/SFT-Sachs/output/` (43 MB).

## 6. Consequences for the N-body B-mode test (A2's reading)

1. **Cheapest decisive measurement**: `read_field(zs38, "gamma1"/"gamma2")` -> `hp.map2alm_spin(..., 2, lmax ~ 4000-8000)` -> C_BB, C_EE; compare C_BB/C_EE^O0 with the Letter's f_BB (6.3e-3..8.3e-3 at l=60 for z_s = 5.34; 4.4e-3..5.4e-3 for zs16), with the Krause-Hirata ceiling (2.0e-3 / 3.3e-4), and with the FF prediction (BB/EE_FF = 0.032 x FF/O0 ~ 0.2% -> ~6e-5). The `omega` field gives the image rotation for the post-Born consistency check. Both Takahashi's own paper and the Level-1 SFT-Sachs result (BB/EE ~ 2e-5 at z_s = 5) set the expectation two to three orders below the Letter. Takahashi's ray-tracing includes deflection (all orders), its resolution window closes at l ~ 1400, and the maps carry their own numerical B-mode; the comparison must be to their published B-mode/rotation spectra.
2. **Level-1 sachsray on gevolution**: the cb3 phi cone is usable only as a smoke test (no chi, Nside 256, z_s <= 1.03). A proper run needs phi (+chi) on every shell to z_s; `settings_z5big_seed42.ini` is the template (1920 Mpc/h, Ngrid 256, minutes; Ngrid 512 ~20 min per cb3 timing). `cb3_gevolution_io.MetricLightcone` + `cb3_two_routes.native_route` deliver per-shell Phi00/Psi0 alms in the paper's sign convention for `fields.driving_from_components`. SFT-Sachs's `raytrace` is the ready-made numpy cross-check of sachsray on identical shells.
3. **Transform gate**: any 2PCF-route comparison must first pass the Order-0 gate of `docs/handoff_f4_second_path.md` (`|C_BB^O0|/C_EE^O0 < 1e-4`); the A-1 route failed it, so harmonic-space measurement directly from maps is the safer path.
4. **Table-level check of the theorem's counter-statement**: apply `T_{22}^{-1}` and `T_{2,-2}^{-1}` to the D-RZ-44 / D-RZ-47 `xi_+^FK`, `xi_-^FK` folds (or to the upstream `..._FK_cut15360_permfix` sweep) and test whether they return one spectrum; the tabulated `|xi_-/xi_+|` of 1e-3..9e-2 already indicates they do not.

## 7. Sign and convention hazards for reuse

- Paper / stf-transfer / cb3: `Phi00 = -(1/2) R_{mu nu} k^mu k^nu = -A_tr delta` (positive overdensity focuses). SFT-Sachs `driving_fields.py`: `+(1/2) R k k` (D-CB-06). SFT-Sachs `tidal.py` then uses `Tr T = 2 Phi00` with `D'' = -T D`; check the overall sign chain before feeding SFT-Sachs driving fields to sachsray's `J'' = T J`.
- SFT-Sachs Psi0 = trace-free screen Hessian of (Phi+Psi)/2 after the 2026-06-16 fix; `harmonic_kernels.NEWTONIAN_PSI0_KERNELS` coefficients 1/a^2, -0.5/a^2. Pre-fix data are 2x in shear.
- gevolution `.map` files are pixbatch-packed; use `sft_sachs.raytrace.unpack` / `cb3_gevolution_io` (not `sft_sachs.gevolution_io.read_shells`).
- `stf_transfer.background.load_background` misreads gevolution 1.3's `conformal H/H0` column (R-CB3-2); use `cb3_gevolution_io.read_background`.
- Every FK amplitude must be quoted with its l_max (D-W0-21 rule); the vendored June FK sweep is l_max ~ 1000 and pre-fix.