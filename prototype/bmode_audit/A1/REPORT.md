
# Agent A1 -- audit of `prototype/fk_mc/` and `sachsray/` against the B-mode theorem

Audit scripts and logs: `/Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/A1/` (four scripts, four logs). Nothing outside that directory was modified; nothing committed.

## 0. Verdict

* **prototype/fk_mc FK numbers: WRONG, retract.** They were produced with (i) the June cut1000, frozen-corner vertex, (ii) the single-leg `simulate_fk_vr` estimator (share 0.25 at 0.5', negative beyond 12'), and (iii) the nominal `Q` calibration (x3.9 inflation at n_lambda = 1000, uncharacterised at the n_lambda = 600 used). Every downstream statement ("FK dominates O0 by 10-80x above 200'", "C_l^FK/C_l^O0 up to 7x", the four `max|FK/(O0+FF)|` values 46/78/47/5, the Figure-12 "modulus pair" story) is an artifact.
* **Theorem: HOLDS, and is confirmed on the paper's own sweeps.** For any spin-2 field cross-correlated with a pure-E field, xi_+ and xi_- are the d^l_{2,2} / d^l_{2,-2} transforms of one spectrum. The paper's O0 sweep satisfies this at the percent level; its FK (cut15360 permfix) sweep violates it by a factor 2-5 -- and that violation *is* the paper's B/E = 0.95 -> 0.42. C_BB^FK = 0 identically; the first shear B-mode is fourth order (FF + K4).
* **sachsray: SOUND and reusable.** Observables/tidal matrix are correct; the screen-basis convention is now pinned with a spin-2 test; float32 is adequate at rtol <= 1e-6; the streaming path works. Concrete gaps for N-body shells are listed in section 4.

## 1. Provenance of the June FK numbers (question 1)

| item | what the June run used | evidence |
|---|---|---|
| vertex table | `callables/kappa3_vertex/equal_time_limber` -> `equal_time_limber_kappa3_callable` (ell_max = 1000, 15-node cosine grid frozen at the (1,1,1) corner) | `git show 7c1afd0:sachs_sft/analyses/mc_sachs_2pt/driver_stats.py` lines 109, 115 (commit 2026-06-17); the current file (lines 109-117) was rebound to `equal_time_limber_cut15360_permaware` on 2026-09-02 (commit 921041f) with the comment "was the cut1000 ... callable; only a fresh Monte-Carlo run is affected". The prototype outputs are dated 2026-06-26. |
| estimator | `mc.simulate_fk_vr(MCConfig(sigma_lambda=8, n_lambda=600, use_f_vertex=True, skew_scale=1), gamma)` | `vr_channels.py:39-56`, `fk_highstat.py:28-38` |
| placement | deforms only the observable leg; captured share T1/(T1+T2) = 0.251 (0.5'), 0.170 (2'), 0.089 (5'), 0.031 (8'), -0.024 (12'), -0.163 (30') | `sachs_mc_core.py:444-520` (T = Q : Cov(kappa_g - kappa_g0, zz), FK = T + T^T); `mc_fk_complete/NOTES.md` section 3.1 |
| Q calibration | nominal: `Q[k] = solve_Q(V_k, Z6/(2 sigma)^2)`, V_k = Sigma2/(2 sigma) | `sachs_mc_core.py:250-254`; NOTES 3.2: x3.92 inflation at sigma = 8, n_lambda = 1000; grid-unstable (exact/fold 1.86, 65.5, 31.9, 2.79, 3.98 at n_lambda 1000-4000) -- the prototype's n_lambda = 600 is uncharacterised |

Consequences, measured (`theorem_check_2pcf.log`):

| gamma | prototype FK_kk | cut1000 fold | permfix fold | proto/cut1000 | proto/permfix |
|---|---|---|---|---|---|
| 1.0' | 6.97e-6 | 4.288e-6 | 1.598e-5 (SPEC) | 1.63 | 0.44 |
| 10.9' | 6.02e-6 | 4.287e-6 | 3.48e-6 | 1.40 | 1.73 |
| 165.7' | 7.82e-6 | 4.2e-6 | -8.2e-9 | 1.86 | -950 |

The June cut1000 fold is flat at 4.288e-6 from 0.5' to 70' and has xi_- = O(1e-16) everywhere (frozen corner); the manuscript's permfix fold is 4.54x higher at 0.5' and decays by three orders of magnitude, changing sign near 3 degrees. The "FK dominates O0 by 10-80x above 200'" is the flat corner value read against a decaying Order-0 (the paper's 2026-08 audit, `docs/fk_audit_2026-08/SUMMARY.md` section 4, withdrew its own 2-degree crossover statements for the same reason). With the corrected table |FK/O0| = 0.02 (294'), 0.10 (597'), 0.34 (1212'), 0.7 (2462'), and the paper declares FK unconverged beyond ~1 degree (ell <~ 50). The near-agreement of the June MC with the June fold at ~1' (0.89-1.03 in NOTES 3.3) is the accidental product of share x inflation.

Also outdated: the prototype's FF channel is ~0.5x the paper's fold at 1' (9.99e-7 vs ~1.9e-6) and has an unphysical negative constant floor in xi_kk, xi_+ *and* xi_- at large separation (-4.6e-8, -6.8e-8, -5.9e-8; a spin-2 xi_- cannot tend to a constant), whereas the paper's FF floor is +5.785e-7 in kk with xi_+- -> 0. NOTES 5d already records the MC FF as unresolved.

## 2. The theorem versus the three artifacts (question 2)

**Structure.** FK = <[F R f R f](n1) [R f](n2)>_{K3} + (1 <-> 2) = 2 Re <gamma^(2)(n1) gamma^(1)*(n2)> restricted to the connected three-point part. The n2 leg is the F-free linear response; with Psi0 = eth^2 of a scalar on the unperturbed ray, gamma^(1) = -int G eth^2 phi = eth^2(-int G phi) is pure E realisation by realisation, so <A gamma^(1)> has C_BB = 0 for any spin-2 A, and xi_-^FK = T_{2,-2}[T_{2,2}^{-1} xi_+^FK]. mc_fk_complete's own estimator has exactly this shape (T1 = Cov(kappa_g^(2), kappa_d^(1)), T2 = Cov(kappa_g^(1), kappa_d^(2)): every term pairs an F-free linear leg with an F-induced one).

**Numerical test** (`theorem_check_2pcf.py`): flat-sky pure-E relation xi_-(t) = xi_+(t) + int_0^t dv v xi_+(v)(4/t^2 - 12 v^2/t^4) (derived from J4 - J0 = (48/x^3 - 8/x)J1 - 24 J0/x^2), applied to the paper's sweeps and the prototype's channels; then the paper's own curved-sky transform (ported verbatim in `figure12.py`) for E/B.

| sweep | xi_-/xi_-^pureE (3.3', 5.3', 22', 71') | BB/EE at l = 60 / 1500 | (EE-BB)_pureE / (EE-BB)_sweep at l = 60 / 1500 |
|---|---|---|---|
| O0 (`C_corr_op_O0`) | 0.97, 1.01, 1.01, 1.01 | 0.000 / -0.040 | 1.001 / 0.996 |
| FF | 0.86, 0.91, 0.92, 0.95 | 0.023 / 0.019 | 1.05 / 1.11 |
| FK cut15360 permfix (manuscript) | **0.44, 0.41, 0.30, 0.19** | **0.951 / 0.420** | **42.9 / 2.6** |
| FK cut1000 (June) | 0.000 (xi_- = O(1e-16)) | 1.04 / 1.15 (meaningless) | -- |
| prototype MC fk | xi_-/xi_+ = 0.005-0.013 for 5'-30' (noise) | garbage (VR blow-ups at 460'/647') | -- |

So the O0 control passes (1-3%), FF shows its genuine fourth-order B (5-12%, "thirty times below E"), and the manuscript's FK xi_- is 2-5x *too small* relative to its own xi_+. Feeding that sweep through the transform reproduces the paper's BB/EE = 0.951 (60), 0.769 (93), 0.649 (270), 0.487 (977), 0.420 (1500) exactly. Since K011 = (Bmod + TPP)/2 feeds xi_+ and K022 = (Bmod - TPP)/2 feeds xi_- (`notes/input_ready_b_to_zeta_spec.md:75`), the paper's B-mode is precisely a spin inconsistency between the deployed TPP and Bmod channels (the deployed Bmod is zeta_TTT with no pair phase, D-RZ-47; TPP carries the 23% permutation ambiguity, F27). The "gamma^4 below an arcminute" clause is the d^l_{2,-2} kernel with the ell_max = 15360 cutoff (kinematic, true for any pure-E field); the pure-E relation predicts a xi_-^FK 2-5x larger than tabulated at 3'-70'.

**Corollary.** If K011 is right, the true FK E-mode is T22[xi_+] = (EE+BB)_paper = 1.95x (l = 60) to 1.42x (l = 1500) the paper's quoted EE_FK, and f_BB = BB_FK/EE_O0 implied by the deployed sweep (6.4e-3 at l = 60, 6.5e-3 at l = 1500, z_s = 5) is zero. The Letter's "exceeds the sum of every known cosmological B-mode by 8 to 16" and plan facts F2/F4/F26/F27c (as B-mode statements) fall; F1 ("C_BB has no Order-0 component") stands, but the first non-zero B is fourth order: Gaussian FF (post-Born-like without remapping) + connected K4.

**Which statement each artifact asserts:**

| artifact | asserts | status |
|---|---|---|
| `prototype/fk_mc/README.md` ("feeds the modulus pair ... cancels in the rest") | B = E | wrong |
| `notebooks/demo_fk_cl.ipynb` cell 6 ("Delta C_EE^FK = Delta C_BB^FK") | B = E | wrong |
| memory `fk-nongaussian-mc.md` item 6 ("cancels in EE-BB ... matches the paper's spin/parity story") | B = E (the paper's pre-2026-08-29 version) | wrong |
| `fk_highstat.py`, `figure12.py` docstrings ("xi_- = zeta_TPP, gamma^4-suppressed, paper predicts ~0") | B = E | wrong |
| manuscript (conclusion.tex 49-81, cl_EB_polarization), Letter, plan F4 | B/E = 0.42-0.95 | wrong (table artifact) |
| theorem | B = 0 | holds |

Why the MC saw xi_- ~ 0: the cut1000 callable returns the coincident-leg cumulant everywhere (frozen corner), for which <Phi Psi+ Psi+> = <Phi Psix Psix> identically; and for a flat xi_+ the pure-E xi_- is also zero, so the run could not distinguish B = E from B = 0. The notebook's "kappa E parity-forbidden" is also wrong: kappa-B and EB are parity-forbidden; kappa-E is the standard cross-spectrum (the paper's own Figure 12 has a kappa-E panel with a 1-2% FK correction).

## 3. What remains valid (question 3)

* **Jacobi robustness** (`stability_test.py`): a linear ODE cannot blow up in finite lambda; 100% finite where the Riccati overflowed 29-46% is a true numerical statement. Caveat: that driving was unphysical (NOTES 3.2: the nominal calibration hitting non-PSD nodes of the Sigma2 table spikes |Q| by ~5000x), so the *relevance* is reduced; the caustic/vertex-regularity argument of `REDESIGN.md` remains the real case for the Jacobi form.
* **Gaussian bridge gate** (`route.py`): sachsray kappa = int (theta - theta_sa) dlambda vs the paper's Riccati kappa = int s to ~2% at Gaussian level. Valid (two-point functions are even, so the sign convention of route.py's kappa -- opposite to physical kappa = 1 - D/D_bg ~ -int dtheta -- is harmless).
* **Curved-sky transform port** (`figure12.py`): verbatim; Wigner self-test passes; used here to reproduce the paper's E/B to three digits.
* **Order-0 convention gate**: (EE+BB)/(EE-BB) = 1.000, (EE+BB)/kk = 0.971.
* **Brute-force infeasibility** (`brute_fk.py`): valid *for the nominal-Q deformation*; but the Levy-peaked Q is a calibration artifact (smeared `solve_Q(B[k], zeta[k])` gives <kappa^3>/target = 0.9999 vs 4.97 nominal, NOTES 3.2), so "infeasible at every scale" is overstated -- partially outdated.
* **"VR is necessary and correct"**: the variance-reduction *idea* (Q outside the average) is fine, but `simulate_fk_vr` is NOT correct (single leg, nominal Q); the correct estimator is mc_fk_complete's (T1+T2)+transpose with smeared calibration (placement identity proven symbolically, gates 2-4 at 1%).

## 4. sachsray as an N-body-driven engine (question 4)

**Observables and signs.** `observables_from_jacobi`: A = J/D_bg, kappa = 1 - tr A/2, gamma1 = -(A00-A11)/2, gamma2 = -(A01+A10)/2 -- standard and verified (Born limit test, focusing sign test). omega = +(A01-A10)/2 is a convention; Takahashi et al. 2017 appear to write A = [[1-k-g1, -g2-w],[-g2+w, 1-k+g1]] (omega = -(A01-A10)/2) -- verify before comparing omega maps (spectra unaffected). Tidal matrix T = [[Phi00+W1, W2],[W2, Phi00-W1]]: S' = T - S^2 reproduces theta' = Phi00 - theta^2 - s+^2 - sx^2, s+' = W1 - 2 theta s+ exactly, consistent with the Riccati pin (`test_jacobi_matches_riccati`). Twist: S stays symmetric (omega_S = 0) while J acquires an antisymmetric part at second order -- matches the paper.

**Screen basis (the key gap, now pinned).** sachsray has no sky-basis knowledge: (gamma1, gamma2) are returned in whatever basis (W1, W2) were supplied. The legacy `sachsfield.FullSkySource.generate` (sources.py:82-127) builds Re Psi0 and Im Psi0 as three *independent scalar* `hp.alm2map` maps -- no spin-2 structure -- so the June demo's shear has E = B and the basis was never confronted. New test (`sachsray_eb_basis_test.py`): from one scalar S per shell, Phi00 = alm2map(-l(l+1) S_lm) and (W1, W2) = alm2map_spin([-sqrt((l+2)!/(l-2)!) S_lm, 0], nside, 2) in HEALPix (e_theta, e_phi). Rule: **the healpy E-alm of Psi0 carries the same sign as the spin-0 alm of Phi00, ratio sqrt(1 - 2/L^2)** (the paper's L^2/chi^2 and sqrt(L^2(L^2-2))/chi^2 multipliers). Then sachsray gives kappa_E = +gamma_E (corr 0.997, EE/kk 0.99) and a pure-E output with BB/EE = 2.04e-3 (kappa_rms 0.107), 1.92e-4 (0.034), 2.31e-5 (0.0106) -- scaling as kappa_rms^2, i.e. the engine's genuine unperturbed-path second-order B (consistent with the paper's FF B ~ 1/30 of FF's E ~ 1e-3 O0). Controls: U -> -U gives BB/EE = 0.93 (pure B); opposite E-alm sign gives kappa = -gamma_E (still pure E). Born-integrated reference floor 3-8e-6 (pixelisation).

**Affine parameter / background.** lambda in physical Mpc, lambda = 0 at the observer, dchi/dlambda = 1/a^2 (E0 = 1), D_bg = a chi (closed form; `test_perflrw_background.py` verifies the Jacobi solver reproduces it and that Phi00_bg = -(1+z)^4 (H^2 - H') < 0). Green's function G = a^2 chi (chi_s - chi)/chi_s in dlambda = a^2 dchi, which turns kappa = -int G Phi00 dlambda into the Born integral with -a^4 Phi00 = grad_perp^2 (Phi+Psi)/2 -- consistent with cosmology.tex eq. Phi00 scalar. `trace_rays`/`trace_rays_streaming` always start at ts[0] with the vertex IC and integrate D_bg from ts[0]; only `solve_jacobi_matrix` accepts y0.

**Streaming.** `trace_rays_streaming(ts, lam_s, n_rays, gen_chunk, phi00_bg, chunk)` never materialises the field (zero-pads the last chunk; equals `trace_rays` to 1e-7 in `test_streaming_matches_trace_rays`). ~13 MB per 8192-ray chunk; output 4 x n_rays x 8 B.

**float32.** Per-ray error is tolerance-dominated, not roundoff-dominated (`float32_test.log`, Mpc units, J ~ 2e3): rms err(gamma1)/std(gamma) = 3.1e-3 (f32) vs 3.2e-3 (f64) at rtol 1e-6; 1.4e-2 at 1e-5; 6.4e-4 vs 5.2e-4 at 1e-7. Direct B floor (`numerical_b_floor.log`, kappa_rms 0.0106, physical BB/EE 2.31e-5): excess +1.5e-8 (f64, 1e-6), +6.8e-8 (f32, 1e-6), +1.1e-6 (f64, 1e-5), +7.1e-7 (f32, 1e-5). **float32 is adequate for B-mode power at rtol <= 1e-6 (atol = 1e-2 rtol); rtol = 1e-5 (the `scale_test.py` setting) leaks ~5% excess B and must not be used**; certify the production configuration with the pure-E gate.

**Gaps for Takahashi / gevolution shells.**
1. `trace_rays`/`trace_rays_streaming` need a y0/lambda0 option (or prepend a background-only segment [0, lambda_first]).
2. lambda(chi) = int a^2 dchi and Phi00_bg from the cosmology: reuse the paper's `Background` (read-only import; D = a chi spline in Mpc) or perFLRW (sign fixed 2026-06-25).
3. Shell driving: Phi00 = -(1+z)^4 grad_perp^2 phi_W, Psi0 = -(1+z)^4 eth^2 phi_W/chi^2 (phi_W = (Phi+Psi)/2) with the sign rule above; for density shells, Limber-Poisson -L^2 phi_W/chi^2 = (3/2) Omega_m H0^2 delta/a (the same equal-shell approximation the paper's vertex uses; radial-derivative/ISW terms dropped). Needs a small `sachsray.fields.driving_from_shells(delta or phi shells, cosmology)` module with alm2map_spin.
4. Shell thickness (150 Mpc/h) >> correlation length: the driving is piecewise-constant per shell; the cubic control through shell samples does not preserve int Phi00 dlambda. Add a per-shell exact 2x2 Jacobi propagator (constant symmetric T per shell -> closed-form cosh/sinh step; this is the multi-plane recursion in Jacobi form) or a fine piecewise-constant lambda grid.
5. gen_chunk gathering (size, n_lam, 3) from shell-major memmaps (n_lam reads per chunk); full field would be ~240 GB at nside 4096, n_lam 100 -- streaming only.
6. Throughput: 66k rays/s single process (June, float32, rtol 1e-5, n_lam 128) -> ~50 min for 2.0e8 rays; x2-4 at rtol 1e-6..1e-7; parallelise over pixel blocks across 28 cores.
7. The Takahashi files on disk (`allskymap_nres12r000.zs*.mag.dat`) are OUTPUT maps, not driving shells: the nres12 density shells must be downloaded, or gevolution cb3 phi snapshots (512^3, 14 snapshots) turned into lightcone shells by replication (separate build).
8. Comparison logic: sachsray on N-body shells (unperturbed path) yields B = FF-Gaussian + K4 only; Takahashi contains all orders including deflection remapping (Krause-Hirata level); the difference isolates post-Born remapping.
9. Minor: `perFLRW.sachs_cls` has cl_psi0/cl_phi00 = L^2/(L^2-2), inverted relative to (L^2-2)/L^2 (<= 0.1% for l >= 45); not used by sachsray tests.

## 5. Reusable / outdated / wrong table

| artifact | status | note |
|---|---|---|
| `sachsray/physics.py`, `solvers.py`, `raytrace.py`, `fields.py` | REUSABLE | conventions verified; add y0/lambda0 to trace_rays; add a shells adapter + per-shell exact propagator |
| `tests/test_sachsray.py`, `test_fields.py`, `test_xval_endtoend.py`, `test_perflrw_background.py` | REUSABLE | add the pure-E spin-2 gate (`sachsray_eb_basis_test.py`) as a test |
| `prototype/REDESIGN.md`, memory `sachs-raytracer-redesign.md` | REUSABLE with addendum | add: legacy generator has no spin-2 structure; basis rule; float32/rtol verdict; omega sign; perFLRW ratio |
| `prototype/fk_mc/figure12.py` (transform, `combine`, `smooth_33`) | REUSABLE (transform) / OUTDATED (smoothing, docstrings) | docstrings assert B = E |
| `prototype/fk_mc/route.py`, `stability_test.py` | REUSABLE (gates) | relevance of the Riccati blow-up reduced (unphysical Levy-Q) |
| `prototype/fk_mc/brute_fk.py` | OUTDATED | infeasibility is specific to the nominal-Q calibration |
| `prototype/fk_mc/vr_channels.py`, `fk_highstat.py`, `cl_contribution.py` | WRONG (FK), OUTDATED (FF) | single-leg VR, nominal Q, cut1000 vertex, n_lambda = 600 |
| `prototype/fk_mc/outputs/fk_channels.npz` (fk, fk_se), `figure12_mc.png` | WRONG | retract; keep only as a record |
| `prototype/fk_mc/README.md` result/finding 3/Figure-12 paragraph | WRONG | banner below |
| `notebooks/demo_fk_cl.ipynb` cells 0 and 6 | WRONG | replacement text below |
| memory `fk-nongaussian-mc.md` items 3-6, FINALIZED block | WRONG | replacement below |
| `sachsfield/sources.py` FullSkySource/FlatSkySource | OUTDATED for B-mode work | independent scalar maps, no spin-2 structure |
| `perFLRW/cosmology.py` sachs_cls psi0 ratio | minor bug | inverted L^2/(L^2-2) |
| paper's `simulate_fk_vr` (read-only) | WRONG estimator | superseded by mc_fk_complete |

## 6. Proposed cleanup texts (NOT applied)

### 6.1 Banner for the top of `prototype/fk_mc/README.md`

```
> **SUPERSEDED / CORRECTION (2026-09-03).** The FK results in this folder
> (`outputs/fk_channels.npz` fk/fk_se, `figure12_mc.png`, the "Result in one
> line", "FK dominates O0 by 10-80x above 200'", "up to 7x C_l", the per-panel
> max|FK/(O0+FF)| = 46/78/47/5, and the Figure-12 "modulus pair" story) are
> RETRACTED. Three independent defects, each established in the paper's 2026-08
> audit (`SFT-lensing-paper-analyses/sachs_sft/analyses/mc_fk_complete/NOTES.md`
> sections 3.1-3.3; `docs/fk_audit_2026-08/SUMMARY.md`):
> 1. Wrong vertex. The run bound the June `equal_time_limber` kappa3 callable
>    (ell_max = 1000, cosine grid frozen at the (1,1,1) corner). Its fold is 4.5x
>    low at 0.5' and FLAT to ~600' where the manuscript's ell_max = 15360,
>    permutation-aware vertex decays by three orders of magnitude and changes
>    sign near 3 degrees. "FK dominates O0 above 200'" is that frozen corner read
>    against a decaying Order-0; with the corrected table |FK/O0| is 0.02 at
>    5 deg and 0.1 at 10 deg, and FK is unconverged there anyway.
> 2. Incomplete estimator. `simulate_fk_vr` deforms only the observable leg and
>    captures a share T1/(T1+T2) = 0.25 (0.5'), 0.17 (2'), 0.03 (8'), negative
>    beyond ~12' (-0.16 at 30') of the FK diagram.
> 3. Mis-calibrated deformation. Q was solved against the nominal node
>    covariance V = Sigma2/(2 sigma). The "Levy-peaked Q" of finding 1 is this
>    mis-calibration hitting non-PSD nodes of the Sigma2 table (x3.9 inflation
>    at sigma_lambda = 8, n_lambda = 1000; grid-unstable; this run used
>    n_lambda = 600). The correct calibration is solve_Q(B[k], zeta[k]).
> The near-agreement with the June fold at ~1' was the product of (2) and (3).
>
> Spin structure. The claim "FK feeds EE+BB and cancels in EE-BB, i.e.
> Delta C_EE^FK = Delta C_BB^FK" is wrong. The FK diagram is
> 2 Re <gamma^(2)(n1) gamma^(1)*(n2)>; the linear response gamma^(1) is pure E
> realisation by realisation, so C_BB^FK = 0 identically and xi_-^FK is fixed
> by xi_+^FK through the d^l_{2,-2} kernel. The MC's xi_- ~ 0 was the
> frozen-corner coincident-leg identity (the cut1000 fold has xi_- = 0 to
> 1e-16), and a flat xi_+ has a vanishing pure-E xi_- anyway, so the run could
> not distinguish B = E from B = 0. The first non-zero shear B-mode is fourth
> order (Gaussian FF + connected K4). "kappa E parity-forbidden" is also wrong
> (kappa-B and EB are; kappa-E is the standard cross-spectrum).
>
> What survives: the sachsray Jacobi robustness and Gaussian bridge gates
> (`stability_test.py`, `route.py`), the verbatim curved-sky transform in
> `figure12.py`, and the Order-0 convention gate. Audit scripts and logs:
> `prototype/bmode_audit/A1/`.
```

### 6.2 Replacement for `notebooks/demo_fk_cl.ipynb` cell 6 ("Reading the figure")

```
## Reading the figure -- SUPERSEDED (2026-09-03)

This figure is retained as a record only. Its FK curves are not physical:
the run used the June cut1000 vertex (frozen cosine-grid corner: xi_+ flat,
xi_- identically zero), the single-leg `simulate_fk_vr` estimator (captures
0.25 of the FK diagram at 0.5', negative beyond ~12') and the nominal Q
calibration (x3.9 inflation at n_lambda = 1000). The "low-ell excess" in the
top row is the transform of a flat, step-like 2PCF, and the "cancellation in
EE-BB" is the frozen-corner coincident-leg identity, not physics.

The correct spin statement is the opposite of the one written here: the FK
diagram is 2 Re <gamma^(2) gamma^(1)*> with gamma^(1) pure E, so
Delta C_BB^FK = 0 exactly (not Delta C_EE^FK = Delta C_BB^FK), and xi_-^FK is
the d^l_{2,-2} partner of xi_+^FK. The paper's own O0 sweep obeys that
relation at the percent level; its deployed FK sweep violates it by a factor
2-5, which is exactly its published B/E = 0.95 -> 0.42 (see
`prototype/bmode_audit/A1/theorem_check_2pcf.py`). The FF curve here is also
~0.5x the paper's fold and carries an unphysical constant xi_- floor.
kappa-E is not parity-forbidden (kappa-B and EB are).
```

Also change the cell-0 sentence "**Spin structure (the figure's punchline):** FK feeds the modulus pair ... and cancels ..." to "**Spin structure: see the SUPERSEDED note at the end; the punchline written here is wrong (C_BB^FK = 0).**"

### 6.3 Replacement block for memory `fk-nongaussian-mc.md` (replace "KEY FINDINGS" items 3-6 and the "FINALIZED" paragraph)

```
**SUPERSEDED (2026-09-03 audit, prototype/bmode_audit/A1/).** Items 3-6 below and
the FINALIZED block are WRONG and retracted:
- The June run bound the cut1000 kappa3 callable (frozen (1,1,1) cosine corner:
  fold flat at 4.288e-6 to 70', xi_- = 0 to 1e-16; 4.5x low vs the manuscript's
  ell_max = 15360 permutation-aware vertex), used simulate_fk_vr (single-leg
  placement: share 0.25 at 0.5', negative beyond ~12') with the nominal Q
  calibration (x3.9 inflation at n_lambda = 1000; run used 600). Every FK
  number in outputs/fk_channels.npz is uncontrolled (1.6x the June fold at 1',
  wrong sign at 165').
- "FK dominates O0 by 10-80x above 200'" and "C_l^FK/C_l^O0 up to 7x" are the
  frozen-corner artifact (corrected table: |FK/O0| = 0.02 at 5 deg; FK is
  unconverged beyond ~1 deg anyway).
- "FK feeds the modulus pair and cancels in EE-BB (Delta C_EE^FK = Delta
  C_BB^FK)" is wrong: FK = 2 Re <gamma^(2) gamma^(1)*> with gamma^(1) pure E,
  so C_BB^FK = 0 identically; xi_-^FK is the d^l_{2,-2} partner of xi_+^FK. The
  paper's deployed FK sweep violates this pure-E relation by 2-5x (its B/E =
  0.95 -> 0.42 is a vertex-table spin inconsistency between K011 and K022).
  The MC's xi_- ~ 0 was the coincident-leg identity of the frozen corner.
- "VR is necessary and correct": the idea is fine, simulate_fk_vr is not; the
  correct estimator is mc_fk_complete's (T1+T2)+transpose with smeared
  calibration solve_Q(B[k], zeta[k]).
- "Brute force infeasible (Levy-Q)": specific to the nominal-Q mis-calibration
  hitting non-PSD Sigma2 nodes (NOTES.md 3.2), not intrinsic.
Still valid: sachsray Jacobi finite where the (unphysically driven) Riccati
blew up; Gaussian bridge kappa = int dtheta to ~2%; the verbatim curved-sky
transform; the Order-0 convention gate. The prototype FF channel is ~0.5x the
paper's fold at 1' with an unphysical constant xi_- floor (also outdated).
```

### 6.4 Addendum block for memory `sachs-raytracer-redesign.md`

```
**2026-09-03 audit additions (prototype/bmode_audit/A1/).**
- The legacy sachsfield.FullSkySource generates Re Psi0 and Im Psi0 as three
  INDEPENDENT scalar alm2map maps: no spin-2 structure, so demo_sachsray's shear
  has E = B. sachsray itself has no sky-basis knowledge: (gamma1, gamma2) come
  out in the basis (W1, W2) were supplied in.
- Basis rule (pinned by sachsray_eb_basis_test.py): from one scalar S per shell,
  Phi00 = alm2map(-l(l+1) S_lm), (W1, W2) = alm2map_spin([-sqrt((l+2)!/(l-2)!)
  S_lm, 0], nside, 2) in HEALPix (e_theta, e_phi), i.e. the healpy E-alm of Psi0
  has the SAME sign as the spin-0 alm of Phi00 (ratio sqrt(1-2/L^2)). Then
  kappa_E = +gamma_E (corr 0.997) and the output is pure E with BB/EE ~
  kappa_rms^2 (2.3e-5 at kappa_rms 0.011; U -> -U gives BB/EE = 0.93).
- float32 is adequate for B-mode power at rtol <= 1e-6 (excess BB/EE < 1e-7 vs
  physical 2.3e-5); rtol = 1e-5 (scale_test setting) leaks ~5% excess B.
  Per-ray error is tolerance-dominated (3e-3 of std gamma at rtol 1e-6).
- omega = +(A01-A10)/2 is a convention (Takahashi 2017 likely opposite; verify).
- trace_rays/trace_rays_streaming lack a y0/lambda0 option (only
  solve_jacobi_matrix has y0). Shell input needs a driving_from_shells adapter
  and a per-shell exact 2x2 propagator (piecewise-constant T).
- perFLRW.sachs_cls: cl_psi0/cl_phi00 = L^2/(L^2-2) is inverted (should be
  (L^2-2)/L^2); <= 0.1% for l >= 45.
```

## 7. Files written

`/Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/A1/{theorem_check_2pcf, sachsray_eb_basis_test, float32_test, numerical_b_floor}.{py,log}`. `theorem_check_2pcf.py` is proposed as a standing spin-consistency gate for any vertex table (O0 must pass at the percent level for gamma >= 3'; FK must pass exactly).

## 8. Caveats

See the structured caveats list: flat-sky relation with constant extrapolation below 0.5' (ratios meaningful only for gamma >= 3'; O0 is the control); the theorem's premise checked against the paper's diagram description and mc_fk_complete's T1/T2 structure, not the sft-wick source; toy spectra and dimensionless/vacuum backgrounds in the sachsray tests; Takahashi omega convention from memory; June throughput not re-measured; FF discrepancy reported, not diagnosed; no edits applied anywhere outside `prototype/bmode_audit/A1/`.