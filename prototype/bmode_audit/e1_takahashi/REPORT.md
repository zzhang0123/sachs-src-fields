# E1 — Shear E/B, convergence and rotation spectra of the Takahashi et al. 2017 full-sky ray-traced maps, versus the paper's FK B-mode prediction

Agent E1, 2026-09-03. Everything here is reproducible from this directory:

```bash
P=/opt/homebrew/Caskroom/miniconda/base/bin/python     # healpy 1.19.0, numpy 2.3.5, camb
export OMP_NUM_THREADS=24
$P e1_takahashi_bmode.py measure --plane zs16 --mode fast   # Nside 1024, lmax 2048 (22 s)
$P e1_takahashi_bmode.py measure --plane zs16 --mode full   # Nside 4096, lmax 4096 (45 s)
# same for zs38, zs10
$P e1_takahashi_bmode.py floor --plane zs16                 # pipeline B-mode floor (pure-E synthesis)
$P post_born_theory.py                                      # post-Born rotation spectrum, CAMB halofit
$P e1_bmap_diagnostic.py                                    # B map vs rotation-implied B map
$P e1_report.py                                             # tables.md, results.json, figures
```

Outputs: `outputs/spectra_<plane>_<mode>_nside*_lmax*.npz` (all auto/cross spectra of kappa, omega, E, B, raw and
pixel-window corrected, for both signs of gamma2 in the fast pass), `outputs/results.json` (every number below, with
provenance), `outputs/tables.md` (the full tables, appended at the end of this file), `outputs/post_born_theory.json`,
`outputs/bmap_diag.json`, figures `outputs/fig_spectra.png`, `outputs/fig_ratio_overlay.png`,
`outputs/fig_consistency.png`, `outputs/fig_bmaps.png`. Total compute used: about 6 minutes.

## 0. Verdict

**The paper's FK B-mode is absent from the ray-traced maps by three orders of magnitude at ell = 60 and two at
ell = 300.** On the z_s = 1.033 plane the measured shear B-mode fraction is C_BB/C_EE = 4.9e-6 at ell = 60 (the paper
predicts 4.4e-3 to 5.4e-3, i.e. 900 to 1100 times more), 7.9e-6 at ell = 100 (predicted 4.2e-3 to 5.1e-3, 530 to 650
times more), 2.9e-5 at ell = 300 (predicted 3.7e-3 to 4.5e-3, 130 to 160 times more), 9.4e-5 at ell = 1000 (31 to 38
times) and 1.35e-4 at ell = 1500 (20 to 24 times). Cosmic variance on these ratios is 5 percent at ell = 60 and below
1 percent above ell = 300, so the exclusion is by factors, not sigmas. On the z_s = 5.342 plane (the paper's z_s = 5
case) the exclusion factors are 420 to 540 (ell = 60), 220 to 290 (ell = 100), 65 to 85 (ell = 300), 15 to 19 (ell =
1000) and 9 to 12 (ell = 1500).

Those are conservative: the measured C_BB is itself dominated by a numerical B-mode of the ray-tracer. The part of the
shear B-mode that is coherent with the rotation omega has exactly the amplitude the Helmholtz identity requires
(C_BB^coh = F_ell C_omega-omega, F_ell = (ell+2)(ell-1)/(ell(ell+1)), verified to better than 1 percent through the
cross-spectrum), and the rotation spectrum itself agrees to 3 to 7 percent with the post-Born lens-lens ('22') prediction
computed here for the Takahashi cosmology with CAMB halofit. The *physical* B-mode fraction of the z_s = 1.033 plane is
therefore F_ell C_ww/C_EE = 1.6e-6 (ell = 60), 2.4e-6 (100), 6.7e-6 (300), 1.6e-5 (1000), 2.0e-5 (1500): 2800 to 3400
times below the paper's prediction at ell = 60, 540 to 660 at ell = 300, 130 to 160 at ell = 1500, and 16 to 210 times
below even the Krause and Hirata (2010) ceiling of 3.3e-4 for the sum of all known post-Born-type B-modes at z <= 1.
The maps, which contain every order of the lensing expansion (lens-lens coupling, deflections, all non-Gaussian orders),
show only the Krause and Hirata post-Born B-mode. This is the outcome the THEOREM of the task predicts (C_BB^FK = 0
identically; first B-mode at fourth order = Gaussian post-Born + connected four-point).

## 1. Data, reader, sanity statistics

Maps: `/Users/zzhang/projects/SFT-WL-B/data/allskymap_nres12r000.{zs10,zs16,zs38}.mag.dat` (Takahashi et al. 2017,
arXiv:1706.01472, Nside 4096, RING, float32 kappa/gamma1/gamma2/omega; checksums in `SFT-WL-B/data/PROVENANCE.md`),
read with `SFT-WL-B/analysis/wpc_routeb/cb2_takahashi_io.py::read_field`. Cosmology (Takahashi 2017 Sec. 2):
Omega_m = 0.279, Omega_b = 0.046, Omega_Lambda = 0.721, h = 0.7, sigma_8 = 0.82, n_s = 0.97. Takahashi's Jacobian
convention (their Eq. 1): A = [[1-kappa-gamma1, -gamma2-omega], [-gamma2+omega, 1-kappa+gamma1]]; the shear is the pure
Jacobian shear (not reduced shear); their text: "the multiple-lens scattering ... leads to a non-zero curl mode" and
"the curl-mode power spectrum agrees with the leading-order post-Born correction at ell < 1000"; accuracy "within 5
percent at ell < 3000 (1400) for Nside = 8192 (4096)"; resolution damping C_ell -> C_ell/[1+(ell/ell_res)^2],
ell_res = 1.6 Nside (their Eq. 5); the lens-shell thickness lowers C_ell by < 5 percent at ell = 10 to 100.

Field statistics (float64 accumulation over all 201,326,592 pixels; `results.json: planes.*.full.field_stats`):

| plane | z_s | kappa mean / rms / min / max | gamma1 rms | gamma2 rms | omega rms | omega min / max |
|---|---|---|---|---|---|---|
| zs10 | 0.574 | 1.2e-8 / 9.283e-3 / -0.0155 / 0.670 | 6.564e-3 | 6.559e-3 | 2.634e-5 | -6.4e-3 / 4.1e-3 |
| zs16 | 1.033 | 4.9e-8 / 1.6576e-2 / -0.0371 / 0.762 | 1.1725e-2 | 1.1710e-2 | 8.334e-5 | -6.6e-3 / 4.8e-3 |
| zs38 | 5.342 | -1.6e-7 / 4.439e-2 / -0.191 / 0.954 | 3.169e-2 | 3.105e-2 | 6.048e-4 | -2.0e-2 / 1.6e-2 |

zs16 reproduces `SFT-WL-B/analysis/wpc_routeb/FINDINGS_CB2.md` (mean 5e-8, rms 0.0166, max 0.76). No non-finite pixels.

## 2. Method

* Fast pass: `healpy.ud_grade` (averaging) to Nside 1024, lmax 2048. Full pass: native Nside 4096, lmax 4096.
  Scalars with `map2alm(iter=3)`; spin-2 with `map2alm_spin` plus the same three Jacobi refinement iterations
  (implemented by hand, `map2alm_spin_iter`, since healpy's spin transform has no `iter`). Both signs of gamma2 were
  transformed in the fast pass (convention gate, Section 3). All ten auto/cross spectra of (kappa, omega, E, B) saved.
* Pixel window: `healpy.pixwin(nside, pol=True)`; kappa and omega divided by the temperature window squared, E and B by
  the polarization window squared; for the degraded map the product of the Nside-1024 and Nside-4096 windows was used.
  Takahashi's resolution damping (1+(ell/6554)^2)^-1 (0.956 at ell = 1400, 0.95 at ell = 1500) is stored in the npz
  (`damping_takahashi`) but not applied; it cancels in every ratio quoted here. The sphere factor
  F_ell = (ell+2)(ell-1)/(ell(ell+1)) is the exact relation C_EE = F C_kk and C_BB = F C_ww for a shear derived from a
  scalar / pseudo-scalar potential on the sphere (0.995 at ell = 20, 0.99995 at ell = 200).
* Bands {20-50, 50-100, 100-200, 200-400, 400-800, 800-1500, 1500-2500, 2500-4000}: plain mean of C_ell over the band;
  quoted multipoles: mean over [0.9 ell, 1.1 ell]. Gaussian cosmic variance of a full-sky band mean:
  sigma(C)/C = sqrt(sum_l 2 C_l^2/(2l+1))/sum_l C_l; for a ratio of two independent spectra the two are added in
  quadrature (column "CV rel (ratio)"): 4.7 percent for 20-50, 2.5 percent for 50-100, 1.2 percent for 100-200,
  0.6 percent for 200-400, 0.3 percent and below above ell = 400. The maps are one noiseless realisation.
* Pipeline B-mode floor: a Gaussian pure-E shear map with the measured (as-pixelised) kappa spectrum of zs16, extended
  as a power law to lmax = 8192, synthesised at Nside 4096 with `alm2map_spin`, then analysed exactly as the full pass
  and exactly as the fast pass. Any B-mode that comes out is pixelisation, aliasing or quadrature leakage.
* Post-Born theory (`post_born_theory.py`): flat-sky Limber lens-lens ('22') rotation spectrum
  C_ell^ww = int_0^chi_s dchi_2 int_0^chi_2 dchi_1 int d^2l_1/(2pi)^2 sin^2(2 phi_12) P_kappa(l_1; chi_2 -> chi_s)
  P_kappa(l_2; chi_1 -> chi_2), derived from omega = antisym(U_{2s} U_{12}), U(l) = 2 l_a l_b kappa(l)/l^2, so that
  omega(l) = int sin(2 phi_12) kappa_2(l_2) kappa_1(l_1) (Cooray and Hu 2002; Krause and Hirata 2010, whose
  Delta C_gammaB^(22) = Delta C_omega^(22) is the flat-sky form of C_BB = F C_ww). CAMB halofit (Takahashi version)
  P(k) at the map cosmology (sigma_8 renormalised to 0.8200 exactly), 48 x 24 chi grid, 96 x 48 (l_1, phi) grid.
  The first-order Limber C_kk (nu = ell + 1/2) from the same code checks the normalisation.

## 3. Convention gate

Synthetic calibration (Nside 256, pure-E map from `alm2map_spin`): E/kappa = 0.9993, B/E = 2e-8 with (Q, U) as
built; with U -> -U: E/kappa = 0.503, B/E = 1.02. A wrong gamma2 sign therefore shows up as C_EE = C_BB = C_kk/2, not
as C_BB ~ C_kk.

Takahashi maps, gamma2 as stored ("plus"), band 50-1000: C_EE/(F C_kk) = 0.9996 (zs16 fast), 0.9999 (full), 0.9999
(zs10), 0.9999 (zs38); C_BB/C_EE = 1.1e-4 (fast) / 2.5e-5 (full); kappa-E correlation coefficient +1.0000 in every
band. With gamma2 -> -gamma2: C_EE/C_kk = 0.5007, C_BB/C_EE = 0.991 (zs16), 0.502/0.977 (zs10), 0.501/0.996 (zs38).
**The stored gamma2 with healpy's `map2alm_spin` is the physical convention; E reproduces kappa to 1e-4 and correlates
with it at +1.0000.** (The overall sign of the shear relative to kappa is also the healpy one: C_kE > 0.)

## 4. Results, zs16 (z_s = 1.033), native Nside 4096, lmax 4096 (`results.json: planes.zs16.full`)

Band averages (pixel-window corrected):

| band | C_kk | C_EE | C_BB | C_ww | BB/EE | ww/kk | BB/(F ww) | EE/(F kk) | EB/sqrt(EE BB) | CV rel (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|
| 20-50 | 2.333e-08 | 2.327e-08 | 5.911e-14 | 1.797e-14 | 2.54e-06 | 7.70e-07 | 3.30 | 0.9999 | +1.2e-02 | 0.047 |
| 50-100 | 1.080e-08 | 1.079e-08 | 6.130e-14 | 1.946e-14 | 5.68e-06 | 1.80e-06 | 3.15 | 0.9998 | +4.9e-03 | 0.025 |
| 100-200 | 4.632e-09 | 4.631e-09 | 5.599e-14 | 1.590e-14 | 1.21e-05 | 3.43e-06 | 3.52 | 0.9999 | +6.1e-03 | 0.012 |
| 200-400 | 1.840e-09 | 1.840e-09 | 4.987e-14 | 1.187e-14 | 2.71e-05 | 6.45e-06 | 4.20 | 0.9999 | -3.2e-03 | 0.006 |
| 400-800 | 7.663e-10 | 7.662e-10 | 4.284e-14 | 8.532e-15 | 5.59e-05 | 1.11e-05 | 5.02 | 0.9999 | +2.6e-03 | 0.003 |
| 800-1500 | 3.456e-10 | 3.456e-10 | 3.549e-14 | 5.866e-15 | 1.03e-04 | 1.70e-05 | 6.05 | 0.9999 | +1.8e-04 | 0.002 |
| 1500-2500 | 1.617e-10 | 1.616e-10 | 2.780e-14 | 3.765e-15 | 1.72e-04 | 2.33e-05 | 7.38 | 0.9998 | +1.3e-03 | 0.001 |
| 2500-4000 | 7.146e-11 | 7.144e-11 | 1.972e-14 | 2.149e-15 | 2.76e-04 | 3.01e-05 | 9.18 | 0.9997 | +1.4e-03 | 0.001 |

C_EB/sqrt(C_EE C_BB), C_kappa-B, C_omega-E and C_kappa-omega correlation coefficients are all at the 1e-2 to 1e-4 level
consistent with zero (parity), see `tables.md`.

**Quoted multipoles** (mean over [0.9 ell, 1.1 ell]; "physical" = F_ell C_ww/C_EE, the rotation-implied B-mode; theory =
F_ell C_ww^post-Born/C_EE^measured; paper = f_BB = a r/(1+r) with a = 0.009 to 0.011 and r(ell) linear in log ell from
0.95 at 60 to 0.42 at 1500, exactly `SFT-WL-B/analysis/wpa/make_a2_figure.py::r_of_ell` and
`results_a1_inventory.json`; KH = Krause and Hirata 2010 ceiling 3.3e-4 for z <= 1):

| ell | measured BB/EE (total) | CV | physical F ww/EE | post-Born theory | paper f_BB (z_s = 1) | paper / measured | paper / physical | measured / KH | physical / KH |
|---|---|---|---|---|---|---|---|---|---|
| 60 | 4.86e-6 | 5.1% | 1.58e-6 | 1.54e-6 | 4.38e-3 to 5.36e-3 | 900 to 1100 | 2770 to 3390 | 0.015 | 0.0047 |
| 100 | 7.90e-6 | 3.1% | 2.44e-6 | 2.62e-6 | 4.18e-3 to 5.10e-3 | 530 to 650 | 1710 to 2090 | 0.024 | 0.0073 |
| 300 | 2.86e-5 | 1.1% | 6.74e-6 | 6.69e-6 | 3.66e-3 to 4.47e-3 | 128 to 156 | 540 to 660 | 0.086 | 0.020 |
| 1000 | 9.40e-5 | 0.3% | 1.61e-5 | 1.74e-5 | 2.95e-3 to 3.60e-3 | 31 to 38 | 180 to 220 | 0.28 | 0.048 |
| 1500 | 1.35e-4 | 0.2% | 2.03e-5 | 2.28e-5 | 2.66e-3 to 3.25e-3 | 20 to 24 | 130 to 160 | 0.41 | 0.061 |

The z <= 3 ceiling (2.0e-3) is 410 (ell = 60) to 15 (ell = 1500) times above the measured total.

## 5. Internal consistency: what the map's B-mode is made of

**(a) Pipeline floor.** Pure-E synthesis analysed as the full pass: B/E = 4e-14 (20-50) to 1.7e-11 (2500-4000), EB/EE
about 1e-9. Analysed as the fast pass (ud_grade to 1024): B/E = 3.2e-8 (20-50), 3.0e-7 (50-100), 2.7e-6 (100-200),
2.8e-5 (200-400), 2.8e-4 (400-800), 2.7e-3 (800-1500). The fast-pass excess over the full pass is exactly this floor
(measured fast/full C_BB = 1.01, 1.04, 1.17, 1.78, 4.9, 23.5 in the same bands; e.g. 400-800: fast 2.75e-4, full
5.6e-5, floor 2.8e-4). **Averaging a spin-2 map onto coarser HEALPix pixels leaks E into B at the (ell/Nside_out)^3
level; the native-resolution analysis has no measurable leakage, so every B-mode in Section 4 is in the map itself.**
The fast pass is not usable for the B-mode above ell ~ 200 (it is fine for E, kappa and omega to 1 percent up to 1500).

**(b) Helmholtz identity and the omega-B coherence.** For the Jacobian of any map of the sphere, the deflection is
alpha = grad psi + star grad Omega, the rotation is omega = -(1/2) Laplacian Omega and the shear B-mode is
eth eth Omega, so C_BB = F_ell C_ww exactly, at every order. The maps give C_BB/(F C_ww) = 3.3 (ell 20-50) rising to
9.2 (2500-4000) at z_s = 1.033; 4.6 to 14.6 at z_s = 0.574; 2.1 to 3.9 at z_s = 5.342. The cross-spectrum settles what
the excess is: the correlation coefficient r_wB = C_wB/sqrt(C_ww C_BB) equals -sqrt(F C_ww/C_BB) in every band of
every plane (zs16: -0.537/-0.551, -0.561/-0.563, -0.532/-0.533, -0.489/-0.488, -0.446/-0.446, -0.404/-0.407,
-0.365/-0.368, -0.327/-0.330; zs10 and zs38 likewise, `tables.md`). That equality is the signature of
B_lm = -sqrt(F_ell) omega_lm + N_lm with N uncorrelated with omega: **the coherent part of the shear B-mode has exactly
the amplitude the rotation demands (unit coefficient, to better than 1 percent), and the remainder N is an independent
component with no counterpart in the rotation.** In map space (`e1_bmap_diagnostic.py`, 20 <= ell <= 1024):
corr(B, B_omega) = 0.446 = sqrt(var_Bomega/var_B) = sqrt(7.0e-10/3.5e-9), corr(B - B_omega, B_omega) = -4e-4, and the
variance of B, of B_omega and of the residual are uniform in latitude (3.2e-9 to 3.9e-9, 6.6e-10 to 7.3e-10,
2.6e-9 to 3.2e-9 in twelve 15-degree bands): the excess is isotropic, near-white (C_NN = 4.1e-14 at ell 20-50 falling
slowly to 1.8e-14 at 2500-4000 on zs16; 9e-15 to 3e-15 on zs10; 4e-14 flat then rising on zs38), and not a coordinate
or polar artifact. Its natural origin is the ray-tracer itself: the tidal matrix interpolated at deflected ray positions
stays symmetric (so omega is untouched) but is no longer the Hessian of a potential (so the shear acquires a B-mode). It
carries no physics. **The rotation is therefore the clean measurement of the physical post-Born B-mode, and the total
C_BB is a strict upper bound.**

**(c) The rotation is the Krause and Hirata post-Born B-mode.** Measured C_ww against the lens-lens theory computed
here (zs16, window means; CV on C_ww 10 percent at ell = 20, 2 percent at 100, 0.2 percent at 1000):
ratio = 1.03 (ell 20), 0.97 (30), 0.93 (50), 1.03 (60), 0.95 (80), 0.93 (100), 0.95 (150), 0.98 (200), 1.01 (300),
0.98 (400), 1.01 (500), 0.99 (700), 0.93 (1000), 0.89 (1500). The first-order C_kk from the same code gives 0.97 to
1.01 over ell = 30 to 1000 (0.98 at 1000, 0.97 at 1500), reproducing FINDINGS_CB2's 0.97 to 1.00 and Takahashi's own
accuracy statement; the fall at ell >= 1000 in both is the resolution damping (0.95 at ell = 1500) plus shell
thickness. zs38: C_ww ratio 0.95 (60), 0.98 (100), 0.97 (300), 0.88 (1000), 0.85 (1500); zs10: 0.95, 0.95, 1.06, 0.98,
0.94. The maps' curl mode is thus, independently of Takahashi's own check, the leading-order post-Born rotation of
Cooray and Hu 2002 / Krause and Hirata 2010 to better than 10 percent, on all three planes.

**(d) Where numerics could contaminate.** Fast pass: B above ell ~ 200 (Section 5a). Full pass: none measurable in
the pipeline; the map's own excess N (Section 5b) is present at all ell and dominates the total C_BB by a factor 2
(z_s = 5.3) to 15 (z_s = 0.57, ell = 2500-4000); Takahashi's accuracy window closes at ell ~ 1400 (resolution damping,
5 percent) and the shell discretisation acts at ell < 200 (< 5 percent). None of these affect the conclusion: at
ell = 60 to 300 the paper's B-mode is 2 to 3 orders of magnitude above the total measured B-mode including all numerics.

## 6. z_s = 5.342 (zs38) and z_s = 0.574 (zs10), native resolution

zs38 (window means): total C_BB/C_EE = 1.52e-5 (ell 60), 2.73e-5 (100), 8.14e-5 (300), 2.88e-4 (1000), 4.22e-4
(1500); physical F ww/EE = 8.56e-6, 1.64e-5, 4.57e-5, 1.19e-4, 1.52e-4; post-Born theory 9.05e-6, 1.67e-5, 4.70e-5,
1.35e-4, 1.80e-4. The paper predicts f_BB = 6.33e-3 to 8.28e-3 (ell = 60), 6.03e-3 to 7.89e-3 (100), 5.28e-3 to
6.91e-3 (300), 4.26e-3 to 5.57e-3 (1000), 3.85e-3 to 5.03e-3 (1500): paper/measured-total = 416 to 543 (60), 221 to
289 (100), 65 to 85 (300), 15 to 19 (1000), 9 to 12 (1500); paper/physical = 740 to 970 (60), 370 to 480 (100), 116
to 151 (300), 36 to 47 (1000), 25 to 33 (1500). Against the z <= 3 KH ceiling (2.0e-3; z_s = 5.3 is beyond KH's
table): measured total = 0.008, 0.014, 0.041, 0.14, 0.21 of it at the five multipoles. C_EE/(F C_kk) = 0.9997 to
1.0000; r_wB = -sqrt(F ww/BB) to 0.5 percent in all bands.

zs10 (window means): total C_BB/C_EE = 2.53e-6 (60), 4.61e-6 (100), 1.59e-5 (300), 4.88e-5 (1000), 7.19e-5 (1500);
physical F ww/EE = 5.40e-7, 9.00e-7, 2.41e-6, 5.44e-6, 6.95e-6 (theory 5.72e-7, 9.51e-7, 2.27e-6, 5.57e-6, 7.38e-6).
All are 5 to 130 (total) and 47 to 610 (physical) times below the KH z <= 1 ceiling of 3.3e-4. No paper prediction
exists for z_s = 0.57, but the paper states the FK amplitude grows with z_s from 0.9-1.1 percent at z_s = 1, so an
f_BB of order a few 1e-3 at ell = 60 would be expected there too, of order 1000 times the measurement.

## 7. Takahashi 2017's own statements (arXiv:1706.01472, via the ar5iv HTML)

Their deflection-angle section: the curl mode of the deflection is generated by multiple-lens scattering and "the
curl-mode power spectrum agrees with the leading-order post-Born correction at ell < 1000" (their Fig. 10 shows the
gradient and curl deflection spectra, the curl multiplied by 100 for display, i.e. two orders of magnitude below the
gradient at the CMB source plane); the post-Born deflection "enhances the [CMB polarization] B-mode power spectrum by
0.5-1 percent at ell >~ 2000" (their Fig. 12). They make no statement about the galaxy-shear B-mode of the released
maps; the shear B/E quoted here (4.9e-6 at ell = 60 for z_s = 1.033, of which 1.6e-6 is the physical post-Born part)
is therefore new information about the product, and its numerical B floor (Section 5b) is a caveat for anyone using
these maps for B-mode work below the 1e-5 level.

## 8. Reading for the audit

1. A full-sky N-body ray-trace containing every order of the lensing expansion has a shear B-mode of 5e-6 of the E-mode
   at ell = 60 (z_s = 1), 1.5e-5 at z_s = 5.3, most of it a numerical floor; the physical part equals the rotation
   (Helmholtz identity, verified to < 1 percent) and the rotation equals the Krause and Hirata lens-lens post-Born
   prediction (verified here to < 10 percent). This is precisely the fourth-order Gaussian post-Born term (Sigma_2^2)
   of the THEOREM; no O(zeta) B-mode at the 4e-3 to 8e-3 level exists.
2. The paper's FK B-mode (f_BB = 4.4e-3 to 8.3e-3 at ell = 60, "B-mode carrying about two thirds of the E-mode power
   over 50 <~ ell <~ 1500") is excluded by factors of 900 to 1100 (z_s = 1) and 420 to 540 (z_s = 5) at ell = 60, and
   by at least 20 (z_s = 1) and 9 (z_s = 5) at ell = 1500, using the most conservative (numerics-inclusive) measured
   B-mode; by 2800 to 3400 (z_s = 1) and 740 to 970 (z_s = 5) at ell = 60 using the physical B-mode. The Letter's
   premise ("exceeds the sum of every known cosmological B-mode by 8 to 16") is inverted: the maps' physical B-mode is
   16 to 210 times below that known sum at z_s = 1.
3. Because the FK diagram is linear in the three-point cumulant and the maps carry the full non-Gaussian density field
   (validated on the same file to 3 to 11 percent on the kappa bispectrum, FINDINGS_CB2), the absence cannot be blamed
   on the maps missing the non-Gaussianity. It is consistent with C_BB^FK = 0 identically.

## Appendix: full tables (outputs/tables.md)


### zs10 (z_s = 0.574), fast pass: Nside 1024, lmax 2048, gamma2 sign convention = plus; gate {"plus": {"EE_over_kk": 0.9996906664325352, "BB_over_EE": 8.255725206461537e-05}, "minus": {"EE_over_kk": 0.5020075850058583, "BB_over_EE": 0.9769783579516202}}

Field statistics (native Nside 4096 map, float64 accumulation):

| field | mean | rms | std | min | max | non-finite |
|---|---|---|---|---|---|---|
| kappa | 1.190e-08 | 9.2831e-03 | 9.2831e-03 | -1.548e-02 | 6.695e-01 | 0 |
| gamma1 | 3.023e-05 | 6.5640e-03 | 6.5640e-03 | -1.948e-01 | 1.911e-01 | 0 |
| gamma2 | -5.023e-08 | 6.5594e-03 | 6.5594e-03 | -1.843e-01 | 1.734e-01 | 0 |
| omega | -2.230e-10 | 2.6342e-05 | 2.6342e-05 | -6.382e-03 | 4.110e-03 | 0 |

Band-averaged spectra (pixel-window corrected; C_ell mean over the band):

| band | C_kk | C_EE | C_BB | C_ww | BB/EE | ww/kk | BB/(F ww) | EE/(F kk) | EB/sqrt(EE BB) | kE/sqrt(kk EE) | CV rel (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20-50 | 8.783e-09 | 8.762e-09 | 1.170e-14 | 2.518e-15 | 1.336e-06 | 2.867e-07 | 4.658 | 0.9999 | -7.795e-04 | +1.0000 | 0.048 |
| 50-100 | 3.591e-09 | 3.588e-09 | 1.219e-14 | 2.319e-15 | 3.396e-06 | 6.457e-07 | 5.258 | 0.9999 | +8.201e-03 | +1.0000 | 0.025 |
| 100-200 | 1.467e-09 | 1.466e-09 | 1.285e-14 | 1.845e-15 | 8.763e-06 | 1.258e-06 | 6.964 | 0.9999 | +8.645e-03 | +1.0000 | 0.012 |
| 200-400 | 6.279e-10 | 6.279e-10 | 1.948e-14 | 1.435e-15 | 3.103e-05 | 2.286e-06 | 13.576 | 0.9999 | -1.403e-03 | +1.0000 | 0.006 |
| 400-800 | 2.806e-10 | 2.805e-10 | 5.198e-14 | 1.057e-15 | 1.853e-04 | 3.766e-06 | 49.190 | 0.9997 | -2.115e-03 | +0.9999 | 0.003 |
| 800-1500 | 1.251e-10 | 1.248e-10 | 2.162e-13 | 7.225e-16 | 1.732e-03 | 5.777e-06 | 299.319 | 0.9983 | -9.291e-04 | +0.9987 | 0.002 |

Quoted multipoles (mean over ell in [0.9 l, 1.1 l]):

| ell | measured BB/EE | single-ell | CV rel | paper f_BB lo-hi | paper r_B/E | exclusion lo-hi | meas/KH(z<=1) | meas/KH(z<=3) | BB/ww |
|---|---|---|---|---|---|---|---|---|---|
| 60 | 2.625e-06 | 3.004e-06 | 0.051 | n/a | n/a | n/a | 0.008 | 0.0013 | 4.855 |
| 100 | 5.045e-06 | 3.930e-06 | 0.031 | n/a | n/a | n/a | 0.015 | 0.0025 | 5.603 |
| 300 | 3.228e-05 | 3.104e-05 | 0.011 | n/a | n/a | n/a | 0.097 | 0.0161 | 13.426 |
| 1000 | 9.987e-04 | 9.765e-04 | 0.003 | n/a | n/a | n/a | 2.996 | 0.4993 | 182.856 |
| 1500 | 5.067e-03 | 5.154e-03 | 0.002 | n/a | n/a | n/a | 15.202 | 2.5337 | 718.475 |

omega-B coherence (r_wB = C_wB/sqrt(C_ww C_BB); equality with -sqrt(F C_ww/C_BB) means B = -sqrt(F) omega + N, N uncorrelated with omega):

| band | r_wB | -sqrt(F ww/BB) | excess C_BB - F C_ww | excess / C_EE |
|---|---|---|---|---|
| 20-50 | -0.4635 | -0.4634 | 9.191e-15 | 1.049e-06 |
| 50-100 | -0.4439 | -0.4361 | 9.869e-15 | 2.750e-06 |
| 100-200 | -0.3927 | -0.3789 | 1.100e-14 | 7.504e-06 |
| 200-400 | -0.2722 | -0.2714 | 1.805e-14 | 2.875e-05 |
| 400-800 | -0.1429 | -0.1426 | 5.093e-14 | 1.815e-04 |
| 800-1500 | -0.0591 | -0.0578 | 2.155e-13 | 1.726e-03 |

### zs10 (z_s = 0.574), full pass: Nside 4096, lmax 4096, gamma2 sign convention = plus; gate {"plus": {"EE_over_kk": 0.9999150804957974, "BB_over_EE": 1.382485510008991e-05}}

Field statistics (native Nside 4096 map, float64 accumulation):

| field | mean | rms | std | min | max | non-finite |
|---|---|---|---|---|---|---|
| kappa | 1.190e-08 | 9.2831e-03 | 9.2831e-03 | -1.548e-02 | 6.695e-01 | 0 |
| gamma1 | 3.023e-05 | 6.5640e-03 | 6.5640e-03 | -1.948e-01 | 1.911e-01 | 0 |
| gamma2 | -5.023e-08 | 6.5594e-03 | 6.5594e-03 | -1.843e-01 | 1.734e-01 | 0 |
| omega | -2.230e-10 | 2.6342e-05 | 2.6342e-05 | -6.382e-03 | 4.110e-03 | 0 |

Band-averaged spectra (pixel-window corrected; C_ell mean over the band):

| band | C_kk | C_EE | C_BB | C_ww | BB/EE | ww/kk | BB/(F ww) | EE/(F kk) | EB/sqrt(EE BB) | kE/sqrt(kk EE) | CV rel (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20-50 | 8.783e-09 | 8.762e-09 | 1.154e-14 | 2.518e-15 | 1.317e-06 | 2.867e-07 | 4.591 | 0.9999 | -3.548e-03 | +1.0000 | 0.048 |
| 50-100 | 3.590e-09 | 3.588e-09 | 1.145e-14 | 2.318e-15 | 3.190e-06 | 6.457e-07 | 4.939 | 0.9999 | +9.126e-03 | +1.0000 | 0.025 |
| 100-200 | 1.467e-09 | 1.466e-09 | 1.038e-14 | 1.845e-15 | 7.078e-06 | 1.258e-06 | 5.626 | 0.9999 | +5.161e-03 | +1.0000 | 0.012 |
| 200-400 | 6.276e-10 | 6.276e-10 | 9.462e-15 | 1.435e-15 | 1.508e-05 | 2.286e-06 | 6.596 | 0.9999 | -1.737e-03 | +1.0000 | 0.006 |
| 400-800 | 2.800e-10 | 2.800e-10 | 8.225e-15 | 1.054e-15 | 2.938e-05 | 3.764e-06 | 7.803 | 0.9999 | +1.115e-04 | +1.0000 | 0.003 |
| 800-1500 | 1.238e-10 | 1.238e-10 | 6.631e-15 | 7.121e-16 | 5.356e-05 | 5.750e-06 | 9.313 | 0.9999 | +3.984e-04 | +1.0000 | 0.002 |
| 1500-2500 | 5.254e-11 | 5.253e-11 | 4.864e-15 | 4.216e-16 | 9.259e-05 | 8.026e-06 | 11.535 | 0.9998 | +8.049e-04 | +0.9999 | 0.001 |
| 2500-4000 | 2.083e-11 | 2.083e-11 | 3.170e-15 | 2.177e-16 | 1.522e-04 | 1.045e-05 | 14.563 | 0.9998 | +1.700e-03 | +0.9999 | 0.001 |

Quoted multipoles (mean over ell in [0.9 l, 1.1 l]):

| ell | measured BB/EE | single-ell | CV rel | paper f_BB lo-hi | paper r_B/E | exclusion lo-hi | meas/KH(z<=1) | meas/KH(z<=3) | BB/ww |
|---|---|---|---|---|---|---|---|---|---|
| 60 | 2.529e-06 | 2.915e-06 | 0.051 | n/a | n/a | n/a | 0.008 | 0.0013 | 4.677 |
| 100 | 4.607e-06 | 3.758e-06 | 0.031 | n/a | n/a | n/a | 0.014 | 0.0023 | 5.115 |
| 300 | 1.592e-05 | 1.456e-05 | 0.011 | n/a | n/a | n/a | 0.048 | 0.0080 | 6.621 |
| 1000 | 4.881e-05 | 4.980e-05 | 0.003 | n/a | n/a | n/a | 0.146 | 0.0244 | 8.967 |
| 1500 | 7.191e-05 | 7.311e-05 | 0.002 | n/a | n/a | n/a | 0.216 | 0.0360 | 10.343 |

omega-B coherence (r_wB = C_wB/sqrt(C_ww C_BB); equality with -sqrt(F C_ww/C_BB) means B = -sqrt(F) omega + N, N uncorrelated with omega):

| band | r_wB | -sqrt(F ww/BB) | excess C_BB - F C_ww | excess / C_EE |
|---|---|---|---|---|
| 20-50 | -0.4667 | -0.4667 | 9.023e-15 | 1.030e-06 |
| 50-100 | -0.4566 | -0.4500 | 9.129e-15 | 2.544e-06 |
| 100-200 | -0.4315 | -0.4216 | 8.534e-15 | 5.820e-06 |
| 200-400 | -0.3934 | -0.3894 | 8.028e-15 | 1.279e-05 |
| 400-800 | -0.3606 | -0.3580 | 7.171e-15 | 2.561e-05 |
| 800-1500 | -0.3261 | -0.3277 | 5.919e-15 | 4.781e-05 |
| 1500-2500 | -0.2925 | -0.2944 | 4.442e-15 | 8.456e-05 |
| 2500-4000 | -0.2604 | -0.2620 | 2.952e-15 | 1.417e-04 |

Measured vs post-Born theory (post_born_theory.py: CAMB halofit, flat-sky Limber lens-lens '22' term; window mean over [0.9 l, 1.1 l]):

| ell | C_kk meas | C_kk Limber | ratio | C_ww meas | C_ww post-Born | ratio | CV rel ww | physical BB/EE = F ww/EE | theory F ww_th/EE | total BB/EE |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 1.351e-08 | 1.425e-08 | 0.948 | 2.658e-15 | 2.430e-15 | 1.094 | 0.102 | 1.969e-07 | 1.800e-07 | 9.518e-07 |
| 30 | 9.220e-09 | 9.851e-09 | 0.936 | 2.436e-15 | 2.614e-15 | 0.932 | 0.072 | 2.643e-07 | 2.835e-07 | 1.244e-06 |
| 50 | 5.847e-09 | 5.703e-09 | 1.025 | 2.490e-15 | 2.577e-15 | 0.966 | 0.043 | 4.259e-07 | 4.407e-07 | 2.043e-06 |
| 60 | 4.474e-09 | 4.596e-09 | 0.973 | 2.417e-15 | 2.557e-15 | 0.945 | 0.036 | 5.404e-07 | 5.717e-07 | 2.529e-06 |
| 80 | 3.072e-09 | 3.173e-09 | 0.968 | 2.216e-15 | 2.367e-15 | 0.936 | 0.027 | 7.216e-07 | 7.705e-07 | 3.653e-06 |
| 100 | 2.328e-09 | 2.350e-09 | 0.991 | 2.096e-15 | 2.213e-15 | 0.947 | 0.022 | 9.004e-07 | 9.508e-07 | 4.607e-06 |
| 150 | 1.389e-09 | 1.380e-09 | 1.007 | 1.820e-15 | 1.890e-15 | 0.963 | 0.015 | 1.310e-06 | 1.360e-06 | 7.500e-06 |
| 200 | 9.753e-10 | 9.557e-10 | 1.021 | 1.669e-15 | 1.613e-15 | 1.035 | 0.011 | 1.711e-06 | 1.654e-06 | 1.021e-05 |
| 300 | 5.931e-10 | 5.883e-10 | 1.008 | 1.426e-15 | 1.345e-15 | 1.061 | 0.007 | 2.405e-06 | 2.267e-06 | 1.592e-05 |
| 400 | 4.266e-10 | 4.241e-10 | 1.006 | 1.248e-15 | 1.226e-15 | 1.018 | 0.006 | 2.925e-06 | 2.874e-06 | 2.096e-05 |
| 500 | 3.311e-10 | 3.299e-10 | 1.004 | 1.135e-15 | 1.083e-15 | 1.048 | 0.004 | 3.427e-06 | 3.270e-06 | 2.588e-05 |
| 700 | 2.243e-10 | 2.239e-10 | 1.001 | 9.716e-16 | 9.382e-16 | 1.036 | 0.003 | 4.333e-06 | 4.184e-06 | 3.518e-05 |
| 1000 | 1.433e-10 | 1.441e-10 | 0.995 | 7.801e-16 | 7.986e-16 | 0.977 | 0.002 | 5.443e-06 | 5.573e-06 | 4.881e-05 |
| 1500 | 8.082e-11 | 8.230e-11 | 0.982 | 5.619e-16 | 5.961e-16 | 0.943 | 0.001 | 6.953e-06 | 7.377e-06 | 7.191e-05 |
| 2000 | 5.065e-11 | 5.282e-11 | 0.959 | 4.156e-16 | 4.401e-16 | 0.944 | 0.001 | 8.206e-06 | 8.691e-06 | 9.552e-05 |
| 3000 | 2.359e-11 | 2.642e-11 | 0.893 | 2.413e-16 | 2.701e-16 | 0.893 | 0.001 | 1.023e-05 | 1.145e-05 | 1.445e-04 |

### zs16 (z_s = 1.033), fast pass: Nside 1024, lmax 2048, gamma2 sign convention = plus; gate {"plus": {"EE_over_kk": 0.9995600391553785, "BB_over_EE": 0.0001137731658475537}, "minus": {"EE_over_kk": 0.5006863914977291, "BB_over_EE": 0.9913388004852356}}

Field statistics (native Nside 4096 map, float64 accumulation):

| field | mean | rms | std | min | max | non-finite |
|---|---|---|---|---|---|---|
| kappa | 4.856e-08 | 1.6576e-02 | 1.6576e-02 | -3.708e-02 | 7.624e-01 | 0 |
| gamma1 | 4.930e-05 | 1.1725e-02 | 1.1725e-02 | -2.206e-01 | 2.093e-01 | 0 |
| gamma2 | -1.019e-07 | 1.1710e-02 | 1.1710e-02 | -2.463e-01 | 2.032e-01 | 0 |
| omega | 1.117e-10 | 8.3345e-05 | 8.3345e-05 | -6.595e-03 | 4.761e-03 | 0 |

Band-averaged spectra (pixel-window corrected; C_ell mean over the band):

| band | C_kk | C_EE | C_BB | C_ww | BB/EE | ww/kk | BB/(F ww) | EE/(F kk) | EB/sqrt(EE BB) | kE/sqrt(kk EE) | CV rel (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20-50 | 2.333e-08 | 2.327e-08 | 5.991e-14 | 1.797e-14 | 2.574e-06 | 7.704e-07 | 3.340 | 0.9999 | +1.286e-02 | +1.0000 | 0.047 |
| 50-100 | 1.080e-08 | 1.079e-08 | 6.351e-14 | 1.946e-14 | 5.885e-06 | 1.802e-06 | 3.265 | 0.9998 | +2.324e-03 | +1.0000 | 0.025 |
| 100-200 | 4.633e-09 | 4.632e-09 | 6.571e-14 | 1.590e-14 | 1.419e-05 | 3.432e-06 | 4.132 | 0.9999 | +9.121e-03 | +1.0000 | 0.012 |
| 200-400 | 1.841e-09 | 1.841e-09 | 8.863e-14 | 1.188e-14 | 4.814e-05 | 6.449e-06 | 7.463 | 0.9999 | -2.051e-03 | +1.0000 | 0.006 |
| 400-800 | 7.680e-10 | 7.677e-10 | 2.113e-13 | 8.560e-15 | 2.753e-04 | 1.115e-05 | 24.688 | 0.9996 | -1.865e-03 | +0.9998 | 0.003 |
| 800-1500 | 3.495e-10 | 3.487e-10 | 8.344e-13 | 5.973e-15 | 2.393e-03 | 1.709e-05 | 139.692 | 0.9976 | -6.385e-04 | +0.9983 | 0.002 |

Quoted multipoles (mean over ell in [0.9 l, 1.1 l]):

| ell | measured BB/EE | single-ell | CV rel | paper f_BB lo-hi | paper r_B/E | exclusion lo-hi | meas/KH(z<=1) | meas/KH(z<=3) | BB/ww |
|---|---|---|---|---|---|---|---|---|---|
| 60 | 4.967e-06 | 6.315e-06 | 0.051 | 4.38e-03-5.36e-03 | 0.95 | 883-1079 | 0.015 | 0.0025 | 3.137 |
| 100 | 8.415e-06 | 7.875e-06 | 0.031 | 4.18e-03-5.10e-03 | 0.87 | 496-607 | 0.025 | 0.0042 | 3.445 |
| 300 | 5.033e-05 | 4.664e-05 | 0.011 | 3.66e-03-4.47e-03 | 0.68 | 73-89 | 0.151 | 0.0252 | 7.470 |
| 1000 | 1.414e-03 | 1.401e-03 | 0.003 | 2.95e-03-3.60e-03 | 0.49 | 2-3 | 4.243 | 0.7071 | 87.402 |
| 1500 | 6.580e-03 | 6.393e-03 | 0.002 | 2.66e-03-3.25e-03 | 0.42 | 0-0 | 19.741 | 3.2902 | 318.796 |

omega-B coherence (r_wB = C_wB/sqrt(C_ww C_BB); equality with -sqrt(F C_ww/C_BB) means B = -sqrt(F) omega + N, N uncorrelated with omega):

| band | r_wB | -sqrt(F ww/BB) | excess C_BB - F C_ww | excess / C_EE |
|---|---|---|---|---|
| 20-50 | -0.5316 | -0.5472 | 4.197e-14 | 1.804e-06 |
| 50-100 | -0.5486 | -0.5534 | 4.406e-14 | 4.083e-06 |
| 100-200 | -0.4933 | -0.4919 | 4.981e-14 | 1.075e-05 |
| 200-400 | -0.3652 | -0.3661 | 7.675e-14 | 4.169e-05 |
| 400-800 | -0.1991 | -0.2013 | 2.028e-13 | 2.641e-04 |
| 800-1500 | -0.0853 | -0.0846 | 8.284e-13 | 2.376e-03 |

### zs16 (z_s = 1.033), full pass: Nside 4096, lmax 4096, gamma2 sign convention = plus; gate {"plus": {"EE_over_kk": 0.9998767450300844, "BB_over_EE": 2.4572227108336936e-05}}

Field statistics (native Nside 4096 map, float64 accumulation):

| field | mean | rms | std | min | max | non-finite |
|---|---|---|---|---|---|---|
| kappa | 4.856e-08 | 1.6576e-02 | 1.6576e-02 | -3.708e-02 | 7.624e-01 | 0 |
| gamma1 | 4.930e-05 | 1.1725e-02 | 1.1725e-02 | -2.206e-01 | 2.093e-01 | 0 |
| gamma2 | -1.019e-07 | 1.1710e-02 | 1.1710e-02 | -2.463e-01 | 2.032e-01 | 0 |
| omega | 1.117e-10 | 8.3345e-05 | 8.3345e-05 | -6.595e-03 | 4.761e-03 | 0 |

Band-averaged spectra (pixel-window corrected; C_ell mean over the band):

| band | C_kk | C_EE | C_BB | C_ww | BB/EE | ww/kk | BB/(F ww) | EE/(F kk) | EB/sqrt(EE BB) | kE/sqrt(kk EE) | CV rel (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20-50 | 2.333e-08 | 2.327e-08 | 5.911e-14 | 1.797e-14 | 2.540e-06 | 7.704e-07 | 3.295 | 0.9999 | +1.153e-02 | +1.0000 | 0.047 |
| 50-100 | 1.080e-08 | 1.079e-08 | 6.130e-14 | 1.946e-14 | 5.681e-06 | 1.802e-06 | 3.152 | 0.9998 | +4.919e-03 | +1.0000 | 0.025 |
| 100-200 | 4.632e-09 | 4.631e-09 | 5.599e-14 | 1.590e-14 | 1.209e-05 | 3.432e-06 | 3.522 | 0.9999 | +6.058e-03 | +1.0000 | 0.012 |
| 200-400 | 1.840e-09 | 1.840e-09 | 4.987e-14 | 1.187e-14 | 2.710e-05 | 6.449e-06 | 4.202 | 0.9999 | -3.209e-03 | +1.0000 | 0.006 |
| 400-800 | 7.663e-10 | 7.662e-10 | 4.284e-14 | 8.532e-15 | 5.591e-05 | 1.113e-05 | 5.021 | 0.9999 | +2.626e-03 | +1.0000 | 0.003 |
| 800-1500 | 3.456e-10 | 3.456e-10 | 3.549e-14 | 5.866e-15 | 1.027e-04 | 1.697e-05 | 6.050 | 0.9999 | +1.796e-04 | +0.9999 | 0.002 |
| 1500-2500 | 1.617e-10 | 1.616e-10 | 2.780e-14 | 3.765e-15 | 1.720e-04 | 2.329e-05 | 7.383 | 0.9998 | +1.295e-03 | +0.9999 | 0.001 |
| 2500-4000 | 7.146e-11 | 7.144e-11 | 1.972e-14 | 2.149e-15 | 2.760e-04 | 3.007e-05 | 9.177 | 0.9997 | +1.384e-03 | +0.9999 | 0.001 |

Quoted multipoles (mean over ell in [0.9 l, 1.1 l]):

| ell | measured BB/EE | single-ell | CV rel | paper f_BB lo-hi | paper r_B/E | exclusion lo-hi | meas/KH(z<=1) | meas/KH(z<=3) | BB/ww |
|---|---|---|---|---|---|---|---|---|---|
| 60 | 4.861e-06 | 6.238e-06 | 0.051 | 4.38e-03-5.36e-03 | 0.95 | 902-1103 | 0.015 | 0.0024 | 3.070 |
| 100 | 7.900e-06 | 7.464e-06 | 0.031 | 4.18e-03-5.10e-03 | 0.87 | 529-646 | 0.024 | 0.0039 | 3.233 |
| 300 | 2.862e-05 | 2.784e-05 | 0.011 | 3.66e-03-4.47e-03 | 0.68 | 128-156 | 0.086 | 0.0143 | 4.248 |
| 1000 | 9.404e-05 | 9.181e-05 | 0.003 | 2.95e-03-3.60e-03 | 0.49 | 31-38 | 0.282 | 0.0470 | 5.845 |
| 1500 | 1.352e-04 | 1.323e-04 | 0.002 | 2.66e-03-3.25e-03 | 0.42 | 20-24 | 0.406 | 0.0676 | 6.676 |

omega-B coherence (r_wB = C_wB/sqrt(C_ww C_BB); equality with -sqrt(F C_ww/C_BB) means B = -sqrt(F) omega + N, N uncorrelated with omega):

| band | r_wB | -sqrt(F ww/BB) | excess C_BB - F C_ww | excess / C_EE |
|---|---|---|---|---|
| 20-50 | -0.5372 | -0.5509 | 4.117e-14 | 1.769e-06 |
| 50-100 | -0.5613 | -0.5632 | 4.186e-14 | 3.879e-06 |
| 100-200 | -0.5315 | -0.5329 | 4.009e-14 | 8.656e-06 |
| 200-400 | -0.4893 | -0.4878 | 3.800e-14 | 2.065e-05 |
| 400-800 | -0.4460 | -0.4463 | 3.431e-14 | 4.478e-05 |
| 800-1500 | -0.4043 | -0.4066 | 2.962e-14 | 8.572e-05 |
| 1500-2500 | -0.3651 | -0.3680 | 2.403e-14 | 1.487e-04 |
| 2500-4000 | -0.3270 | -0.3301 | 1.757e-14 | 2.459e-04 |

Measured vs post-Born theory (post_born_theory.py: CAMB halofit, flat-sky Limber lens-lens '22' term; window mean over [0.9 l, 1.1 l]):

| ell | C_kk meas | C_kk Limber | ratio | C_ww meas | C_ww post-Born | ratio | CV rel ww | physical BB/EE = F ww/EE | theory F ww_th/EE | total BB/EE |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 3.662e-08 | 3.362e-08 | 1.089 | 1.550e-14 | 1.508e-14 | 1.028 | 0.101 | 4.235e-07 | 4.120e-07 | 1.480e-06 |
| 30 | 2.523e-08 | 2.598e-08 | 0.971 | 1.766e-14 | 1.812e-14 | 0.974 | 0.069 | 6.998e-07 | 7.184e-07 | 2.316e-06 |
| 50 | 1.639e-08 | 1.639e-08 | 1.000 | 1.869e-14 | 2.000e-14 | 0.934 | 0.043 | 1.140e-06 | 1.220e-06 | 3.733e-06 |
| 60 | 1.324e-08 | 1.356e-08 | 0.977 | 2.095e-14 | 2.039e-14 | 1.027 | 0.036 | 1.583e-06 | 1.540e-06 | 4.861e-06 |
| 80 | 9.442e-09 | 9.869e-09 | 0.957 | 1.885e-14 | 1.990e-14 | 0.947 | 0.027 | 1.997e-06 | 2.108e-06 | 6.300e-06 |
| 100 | 7.293e-09 | 7.550e-09 | 0.966 | 1.781e-14 | 1.914e-14 | 0.931 | 0.022 | 2.443e-06 | 2.624e-06 | 7.900e-06 |
| 150 | 4.382e-09 | 4.394e-09 | 0.997 | 1.583e-14 | 1.674e-14 | 0.946 | 0.015 | 3.613e-06 | 3.820e-06 | 1.282e-05 |
| 200 | 2.969e-09 | 2.976e-09 | 0.998 | 1.402e-14 | 1.431e-14 | 0.980 | 0.011 | 4.722e-06 | 4.819e-06 | 1.815e-05 |
| 300 | 1.738e-09 | 1.730e-09 | 1.005 | 1.171e-14 | 1.164e-14 | 1.006 | 0.007 | 6.737e-06 | 6.694e-06 | 2.862e-05 |
| 400 | 1.202e-09 | 1.199e-09 | 1.002 | 1.022e-14 | 1.041e-14 | 0.981 | 0.006 | 8.502e-06 | 8.662e-06 | 3.873e-05 |
| 500 | 9.107e-10 | 9.129e-10 | 0.998 | 9.220e-15 | 9.112e-15 | 1.012 | 0.004 | 1.013e-05 | 1.001e-05 | 4.918e-05 |
| 700 | 6.072e-10 | 6.133e-10 | 0.990 | 7.806e-15 | 7.901e-15 | 0.988 | 0.003 | 1.286e-05 | 1.302e-05 | 6.769e-05 |
| 1000 | 3.955e-10 | 4.033e-10 | 0.981 | 6.363e-15 | 6.869e-15 | 0.926 | 0.002 | 1.609e-05 | 1.737e-05 | 9.404e-05 |
| 1500 | 2.363e-10 | 2.443e-10 | 0.967 | 4.785e-15 | 5.394e-15 | 0.887 | 0.001 | 2.025e-05 | 2.283e-05 | 1.352e-04 |
| 2000 | 1.572e-10 | 1.662e-10 | 0.946 | 3.726e-15 | 4.222e-15 | 0.883 | 0.001 | 2.371e-05 | 2.687e-05 | 1.765e-04 |
| 3000 | 8.018e-11 | 9.122e-11 | 0.879 | 2.356e-15 | 2.865e-15 | 0.822 | 0.001 | 2.940e-05 | 3.575e-05 | 2.616e-04 |

### zs38 (z_s = 5.342), fast pass: Nside 1024, lmax 2048, gamma2 sign convention = plus; gate {"plus": {"EE_over_kk": 0.9994393397471338, "BB_over_EE": 0.00017858167563921144}, "minus": {"EE_over_kk": 0.5008638422443729, "BB_over_EE": 0.995561282507273}}

Field statistics (native Nside 4096 map, float64 accumulation):

| field | mean | rms | std | min | max | non-finite |
|---|---|---|---|---|---|---|
| kappa | -1.624e-07 | 4.4391e-02 | 4.4391e-02 | -1.912e-01 | 9.536e-01 | 0 |
| gamma1 | 4.802e-05 | 3.1687e-02 | 3.1687e-02 | -3.274e-01 | 3.089e-01 | 0 |
| gamma2 | -5.596e-07 | 3.1049e-02 | 3.1049e-02 | -3.063e-01 | 3.321e-01 | 0 |
| omega | 3.016e-09 | 6.0478e-04 | 6.0478e-04 | -1.990e-02 | 1.577e-02 | 0 |

Band-averaged spectra (pixel-window corrected; C_ell mean over the band):

| band | C_kk | C_EE | C_BB | C_ww | BB/EE | ww/kk | BB/(F ww) | EE/(F kk) | EB/sqrt(EE BB) | kE/sqrt(kk EE) | CV rel (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20-50 | 1.092e-07 | 1.089e-07 | 1.035e-12 | 4.911e-13 | 9.506e-06 | 4.499e-06 | 2.112 | 0.9999 | +1.202e-02 | +1.0000 | 0.046 |
| 50-100 | 6.865e-08 | 6.860e-08 | 1.312e-12 | 7.619e-13 | 1.912e-05 | 1.110e-05 | 1.722 | 0.9997 | +3.789e-04 | +1.0000 | 0.024 |
| 100-200 | 3.609e-08 | 3.608e-08 | 1.440e-12 | 8.016e-13 | 3.989e-05 | 2.221e-05 | 1.796 | 0.9999 | +7.733e-03 | +1.0000 | 0.012 |
| 200-400 | 1.571e-08 | 1.571e-08 | 1.544e-12 | 6.733e-13 | 9.825e-05 | 4.285e-05 | 2.293 | 0.9999 | -1.528e-03 | +0.9999 | 0.006 |
| 400-800 | 5.911e-09 | 5.908e-09 | 2.453e-12 | 4.591e-13 | 4.152e-04 | 7.767e-05 | 5.343 | 0.9995 | +6.427e-04 | +0.9997 | 0.003 |
| 800-1500 | 2.253e-09 | 2.245e-09 | 7.208e-12 | 2.853e-13 | 3.210e-03 | 1.267e-04 | 25.264 | 0.9967 | -1.784e-03 | +0.9976 | 0.002 |

Quoted multipoles (mean over ell in [0.9 l, 1.1 l]):

| ell | measured BB/EE | single-ell | CV rel | paper f_BB lo-hi | paper r_B/E | exclusion lo-hi | meas/KH(z<=1) | meas/KH(z<=3) | BB/ww |
|---|---|---|---|---|---|---|---|---|---|
| 60 | 1.534e-05 | 1.533e-05 | 0.051 | 6.33e-03-8.28e-03 | 0.95 | 413-540 | 0.046 | 0.0077 | 1.790 |
| 100 | 2.788e-05 | 2.709e-05 | 0.031 | 6.03e-03-7.89e-03 | 0.87 | 216-283 | 0.084 | 0.0139 | 1.700 |
| 300 | 1.041e-04 | 9.847e-05 | 0.011 | 5.28e-03-6.91e-03 | 0.68 | 51-66 | 0.312 | 0.0520 | 2.280 |
| 1000 | 1.982e-03 | 2.114e-03 | 0.003 | 4.26e-03-5.57e-03 | 0.49 | 2-3 | 5.947 | 0.9912 | 16.544 |
| 1500 | 8.728e-03 | 8.211e-03 | 0.002 | 3.85e-03-5.03e-03 | 0.42 | 0-1 | 26.183 | 4.3638 | 56.242 |

omega-B coherence (r_wB = C_wB/sqrt(C_ww C_BB); equality with -sqrt(F C_ww/C_BB) means B = -sqrt(F) omega + N, N uncorrelated with omega):

| band | r_wB | -sqrt(F ww/BB) | excess C_BB - F C_ww | excess / C_EE |
|---|---|---|---|---|
| 20-50 | -0.6782 | -0.6881 | 5.452e-13 | 5.005e-06 |
| 50-100 | -0.7561 | -0.7620 | 5.500e-13 | 8.017e-06 |
| 100-200 | -0.7464 | -0.7462 | 6.380e-13 | 1.768e-05 |
| 200-400 | -0.6569 | -0.6604 | 8.702e-13 | 5.540e-05 |
| 400-800 | -0.4333 | -0.4326 | 1.994e-12 | 3.375e-04 |
| 800-1500 | -0.1971 | -0.1990 | 6.923e-12 | 3.083e-03 |

### zs38 (z_s = 5.342), full pass: Nside 4096, lmax 4096, gamma2 sign convention = plus; gate {"plus": {"EE_over_kk": 0.9998843176139828, "BB_over_EE": 7.409095832128661e-05}}

Field statistics (native Nside 4096 map, float64 accumulation):

| field | mean | rms | std | min | max | non-finite |
|---|---|---|---|---|---|---|
| kappa | -1.624e-07 | 4.4391e-02 | 4.4391e-02 | -1.912e-01 | 9.536e-01 | 0 |
| gamma1 | 4.802e-05 | 3.1687e-02 | 3.1687e-02 | -3.274e-01 | 3.089e-01 | 0 |
| gamma2 | -5.596e-07 | 3.1049e-02 | 3.1049e-02 | -3.063e-01 | 3.321e-01 | 0 |
| omega | 3.016e-09 | 6.0478e-04 | 6.0478e-04 | -1.990e-02 | 1.577e-02 | 0 |

Band-averaged spectra (pixel-window corrected; C_ell mean over the band):

| band | C_kk | C_EE | C_BB | C_ww | BB/EE | ww/kk | BB/(F ww) | EE/(F kk) | EB/sqrt(EE BB) | kE/sqrt(kk EE) | CV rel (ratio) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 20-50 | 1.092e-07 | 1.089e-07 | 1.026e-12 | 4.911e-13 | 9.418e-06 | 4.499e-06 | 2.092 | 0.9999 | +1.270e-02 | +1.0000 | 0.046 |
| 50-100 | 6.864e-08 | 6.860e-08 | 1.295e-12 | 7.618e-13 | 1.888e-05 | 1.110e-05 | 1.701 | 0.9997 | +1.626e-04 | +1.0000 | 0.024 |
| 100-200 | 3.609e-08 | 3.608e-08 | 1.349e-12 | 8.015e-13 | 3.740e-05 | 2.221e-05 | 1.684 | 1.0000 | +9.071e-03 | +1.0000 | 0.012 |
| 200-400 | 1.570e-08 | 1.570e-08 | 1.197e-12 | 6.728e-13 | 7.624e-05 | 4.284e-05 | 1.779 | 0.9999 | +5.680e-05 | +1.0000 | 0.006 |
| 400-800 | 5.897e-09 | 5.896e-09 | 9.445e-13 | 4.574e-13 | 1.602e-04 | 7.756e-05 | 2.065 | 0.9999 | +1.038e-03 | +0.9999 | 0.003 |
| 800-1500 | 2.224e-09 | 2.224e-09 | 7.009e-13 | 2.798e-13 | 3.152e-04 | 1.258e-04 | 2.505 | 0.9998 | -5.168e-04 | +0.9998 | 0.002 |
| 1500-2500 | 9.674e-10 | 9.670e-10 | 5.166e-13 | 1.672e-13 | 5.342e-04 | 1.728e-04 | 3.090 | 0.9996 | +3.218e-04 | +0.9997 | 0.001 |
| 2500-4000 | 4.481e-10 | 4.478e-10 | 3.728e-13 | 9.568e-14 | 8.325e-04 | 2.135e-04 | 3.896 | 0.9994 | +4.708e-04 | +0.9996 | 0.001 |

Quoted multipoles (mean over ell in [0.9 l, 1.1 l]):

| ell | measured BB/EE | single-ell | CV rel | paper f_BB lo-hi | paper r_B/E | exclusion lo-hi | meas/KH(z<=1) | meas/KH(z<=3) | BB/ww |
|---|---|---|---|---|---|---|---|---|---|
| 60 | 1.524e-05 | 1.503e-05 | 0.051 | 6.33e-03-8.28e-03 | 0.95 | 416-543 | 0.046 | 0.0076 | 1.779 |
| 100 | 2.731e-05 | 2.680e-05 | 0.031 | 6.03e-03-7.89e-03 | 0.87 | 221-289 | 0.082 | 0.0137 | 1.665 |
| 300 | 8.140e-05 | 7.764e-05 | 0.011 | 5.28e-03-6.91e-03 | 0.68 | 65-85 | 0.244 | 0.0407 | 1.783 |
| 1000 | 2.884e-04 | 3.026e-04 | 0.003 | 4.26e-03-5.57e-03 | 0.49 | 15-19 | 0.865 | 0.1442 | 2.422 |
| 1500 | 4.224e-04 | 4.207e-04 | 0.002 | 3.85e-03-5.03e-03 | 0.42 | 9-12 | 1.267 | 0.2112 | 2.779 |

omega-B coherence (r_wB = C_wB/sqrt(C_ww C_BB); equality with -sqrt(F C_ww/C_BB) means B = -sqrt(F) omega + N, N uncorrelated with omega):

| band | r_wB | -sqrt(F ww/BB) | excess C_BB - F C_ww | excess / C_EE |
|---|---|---|---|---|
| 20-50 | -0.6800 | -0.6913 | 5.356e-13 | 4.917e-06 |
| 50-100 | -0.7630 | -0.7667 | 5.339e-13 | 7.784e-06 |
| 100-200 | -0.7697 | -0.7706 | 5.480e-13 | 1.519e-05 |
| 200-400 | -0.7475 | -0.7497 | 5.244e-13 | 3.339e-05 |
| 400-800 | -0.6946 | -0.6959 | 4.871e-13 | 8.262e-05 |
| 800-1500 | -0.6288 | -0.6318 | 4.211e-13 | 1.894e-04 |
| 1500-2500 | -0.5668 | -0.5689 | 3.494e-13 | 3.613e-04 |
| 2500-4000 | -0.5013 | -0.5066 | 2.771e-13 | 6.188e-04 |

Measured vs post-Born theory (post_born_theory.py: CAMB halofit, flat-sky Limber lens-lens '22' term; window mean over [0.9 l, 1.1 l]):

| ell | C_kk meas | C_kk Limber | ratio | C_ww meas | C_ww post-Born | ratio | CV rel ww | physical BB/EE = F ww/EE | theory F ww_th/EE | total BB/EE |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | 1.472e-07 | 1.156e-07 | 1.274 | 3.044e-13 | 3.085e-13 | 0.986 | 0.099 | 2.069e-06 | 2.097e-06 | 5.742e-06 |
| 30 | 1.271e-07 | 1.106e-07 | 1.150 | 4.879e-13 | 4.641e-13 | 1.051 | 0.070 | 3.838e-06 | 3.651e-06 | 8.052e-06 |
| 50 | 8.958e-08 | 9.168e-08 | 0.977 | 6.173e-13 | 6.715e-13 | 0.919 | 0.043 | 6.896e-06 | 7.502e-06 | 1.319e-05 |
| 60 | 8.177e-08 | 8.231e-08 | 0.993 | 6.997e-13 | 7.395e-13 | 0.946 | 0.036 | 8.561e-06 | 9.048e-06 | 1.524e-05 |
| 80 | 6.321e-08 | 6.647e-08 | 0.951 | 7.800e-13 | 8.181e-13 | 0.953 | 0.027 | 1.234e-05 | 1.294e-05 | 2.077e-05 |
| 100 | 5.120e-08 | 5.447e-08 | 0.940 | 8.399e-13 | 8.571e-13 | 0.980 | 0.022 | 1.640e-05 | 1.673e-05 | 2.731e-05 |
| 150 | 3.494e-08 | 3.596e-08 | 0.972 | 7.941e-13 | 8.670e-13 | 0.916 | 0.015 | 2.273e-05 | 2.482e-05 | 3.810e-05 |
| 200 | 2.492e-08 | 2.593e-08 | 0.961 | 7.596e-13 | 8.123e-13 | 0.935 | 0.011 | 3.049e-05 | 3.260e-05 | 5.236e-05 |
| 300 | 1.478e-08 | 1.527e-08 | 0.968 | 6.746e-13 | 6.948e-13 | 0.971 | 0.007 | 4.565e-05 | 4.702e-05 | 8.140e-05 |
| 400 | 1.002e-08 | 1.009e-08 | 0.993 | 5.856e-13 | 6.124e-13 | 0.956 | 0.006 | 5.845e-05 | 6.112e-05 | 1.090e-04 |
| 500 | 7.250e-09 | 7.312e-09 | 0.991 | 5.047e-13 | 5.244e-13 | 0.962 | 0.004 | 6.963e-05 | 7.235e-05 | 1.394e-04 |
| 700 | 4.370e-09 | 4.448e-09 | 0.982 | 4.044e-13 | 4.280e-13 | 0.945 | 0.003 | 9.255e-05 | 9.794e-05 | 2.005e-04 |
| 1000 | 2.586e-09 | 2.652e-09 | 0.975 | 3.079e-13 | 3.485e-13 | 0.884 | 0.002 | 1.191e-04 | 1.348e-04 | 2.884e-04 |
| 1500 | 1.433e-09 | 1.501e-09 | 0.954 | 2.178e-13 | 2.578e-13 | 0.845 | 0.001 | 1.520e-04 | 1.800e-04 | 4.224e-04 |
| 2000 | 9.367e-10 | 1.013e-09 | 0.924 | 1.645e-13 | 1.979e-13 | 0.831 | 0.001 | 1.757e-04 | 2.113e-04 | 5.487e-04 |
| 3000 | 4.969e-10 | 5.831e-10 | 0.852 | 1.042e-13 | 1.370e-13 | 0.760 | 0.001 | 2.097e-04 | 2.759e-04 | 7.896e-04 |

### zs16: fast (Nside 1024) / full (Nside 4096) band ratios, window-corrected

| band | BB fast/full | EE fast/full | ww fast/full |
|---|---|---|---|
| 20-50 | 1.014 | 1.000 | 1.000 |
| 50-100 | 1.036 | 1.000 | 1.000 |
| 100-200 | 1.174 | 1.000 | 1.000 |
| 200-400 | 1.777 | 1.000 | 1.001 |
| 400-800 | 4.933 | 1.002 | 1.003 |
| 800-1500 | 23.512 | 1.009 | 1.018 |

### numerical B-mode floor of the full-pass pipeline (zs16 spectrum, pure-E Gaussian synthesis)

| band | floor BB/EE | floor EB/EE |
|---|---|---|
| 20-50 | 4.361e-14 | +7.575e-09 |
| 50-100 | 9.701e-14 | -5.962e-09 |
| 100-200 | 2.156e-13 | +1.558e-09 |
| 200-400 | 5.495e-13 | -9.400e-10 |
| 400-800 | 1.335e-12 | +4.176e-09 |
| 800-1500 | 3.013e-12 | -2.328e-10 |
| 1500-2500 | 6.745e-12 | -7.040e-10 |
| 2500-4000 | 1.728e-11 | +1.631e-09 |

### numerical B-mode floor of the fast-pass pipeline (zs16 spectrum, pure-E Gaussian synthesis)

| band | floor BB/EE | floor EB/EE |
|---|---|---|
| 20-50 | 3.241e-08 | +1.451e-06 |
| 50-100 | 3.017e-07 | -1.621e-06 |
| 100-200 | 2.690e-06 | -7.086e-06 |
| 200-400 | 2.791e-05 | -8.668e-06 |
| 400-800 | 2.813e-04 | +1.563e-05 |
| 800-1500 | 2.677e-03 | +8.581e-05 |
