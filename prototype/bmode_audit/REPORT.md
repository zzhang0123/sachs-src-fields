# The FK B-mode does not exist: audit of the three-point-sourced cosmic-shear B-mode

Master report of the 2026-09-03 audit. Ten independent agents (three adversarial
theory refuters, two numerical tests, one empirical measurement on N-body maps,
two code/programme audits, two literature surveys) plus the parent session's own
verification of the mechanism. Per-agent reports, scripts, logs and outputs sit in
the sub-directories listed in `README.md`. All trees outside this repository were
read only.

## 0. Verdict

**The B-mode attributed to the FK diagram (conclusion.tex ll. 49-81, figure
`cl_EB_polarization.pdf`, the Letter's premise) is zero identically. The published
B/E = 0.95 (l=60) -> 0.42 (l=1500) is a spin-phase defect in the deployed vertex
table, already fixed in canoes (commit `a95c0d6`, 2026-09-02 14:55) but not yet
propagated to the table, the sweep or the figure.**

Five independent lines agree:

| line | result |
|---|---|
| Theory (T1, T2, T3) | FK = 2 Re <gamma^(2)(n1) gamma^(1)*(n2)>; the linear response gamma^(1) = -int R eth^2 phi is pure E for every realisation (R is proportional to the identity in component space), so C_BB^FK = 0 identically. Verdict "holds" at confidence 0.95-0.97 by all three refuters; every loophole examined closes (section 2). |
| Products (N2, T1, A1) | Feeding the paper's own deployed sweeps through the paper's own transform reproduces 0.951 / 0.420 exactly, and shows the deployed xi_-^FK is 2-5x too small for pure-E consistency with the deployed xi_+^FK (the O0 control passes at 1-3%). The June vertex gave "B = E"; the August vertex gives 0.42-0.95; the theorem gives 0 for both. |
| Mechanism (T1, T3, verified by the session) | On the shear-placement rows of `table_permclosed.npz`, zeta_Bmod = zeta_TTT to 1e-4 (the pair phase e^{-2i phi} was dropped for l > 60). Re-running rezeta's Limber engine WITH the phase gives zeta_TPP / pure-E prediction = 1.000 +- 0.002 at every angle 1'-30'; WITHOUT it, 0.61 -> 0.43, matching the deployed table. |
| Exact solver (N1) | sachsray (nonlinear Jacobi, full sky, skewed scalar-sourced field): the FK-type cross term has C_BB^(2,1)/C_EE^(2,1) = -3e-6 +- 7e-6 (pixel-leakage floor) while its E-mode is detected at 10 sigma; the FF-type term has B/E = 0.2-0.9. |
| N-body maps (E1) | Takahashi et al. 2017 ray-traced maps (all orders, full non-Gaussian density): C_BB/C_EE = 4.9e-6 at l=60 for z_s = 1.03 and 1.5e-5 for z_s = 5.34. The paper's prediction exceeds the measured TOTAL B-mode by 900-1100x (z_s=1) and 420-540x (z_s=5) at l=60, by 20-24x and 9-12x at l=1500; the physical B-mode (= F C_omega) matches the Krause-Hirata post-Born term to 3-7%. |

The literature is unanimous (Krause & Hirata 2010, Sect. 3.1, verbatim: "There is
no '13' B-mode shear or rotation power spectrum because gamma_B^(1) and omega^(1)
vanish"; Cooray & Hu 2002; Hirata & Seljak 2003; Bernardeau, Bonvin & Vernizzi
2010; Pratten & Lewis 2016; Magi, Lepori & Adamek 2026): the shear B-mode starts
at fourth order in the potential, and no published work, analytic or numerical,
contains a bispectrum-linear B-mode.

## 1. The claim under test

* conclusion.tex ll. 56-71: xi_+^FK is sourced by <Phi00 Psi_+ Psi_+> + <Phi00 Psi_x Psi_x>
  ("the Ricci-focusing scalar correlated with the squared modulus of the Weyl shear ...
  blind to the orientation"); xi_-^FK by the difference, "which vanishes when the three
  legs coincide ... suppressed as gamma^4 below an arcminute rather than absent. In
  harmonic space ... the B-mode falls from 0.95 of the E-mode power at l=60 to 0.42 at
  l=1500."
* Caption of `fig: cl EB`: "FK feeds both polarizations, the B-mode carrying about two
  thirds of the E-mode power over 50 <~ l <~ 1500"; FF's B-mode "some thirty times" below
  its E-mode.
* Abstract: the three-point cumulants "populating the E- and B-modes in comparable measure".
* Letter (SFT-WL-B `letter/main.tex`): "purely scalar perturbations generate an irreducible
  B-mode at second order ... exceeds the sum of every known cosmological B-mode by a factor
  of 8 to 16 at survey depth"; Stage-IV S/N ~ 12; parameter biases; Fisher gains. Every
  amplitude in the Letter and the proposal is a linear propagation of
  C_BB^FK = a(z_s) r(l)/(1+r) C_EE^O0 with r from the 0.95 -> 0.42 anchors (A2).
* The numbers: f_BB = C_BB^FK/C_EE^O0 = 6.3e-3..8.3e-3 (l=60) and 3.8e-3..5.0e-3
  (l=1500) at z_s = 5; 4.4e-3..5.4e-3 and 2.7e-3..3.3e-3 at z_s = 1.

## 2. The theorem

**Statement.** Expand the shear in the driving field, gamma = gamma^(1) + gamma^(2) + ...;
E/B decomposition is linear. Then

    C_BB = <|gamma_B^(1)|^2> + 2 Re <gamma_B^(1) gamma_B^(2)*> + [<|gamma_B^(2)|^2> + 2 Re <gamma_B^(1) gamma_B^(3)*>] + ...

The FK diagram (one F vertex, one K3 vertex; path_int.tex ll. 444-478) is exactly the
order-3 term 2 Re <gamma^(2)(n1) gamma^(1)*(n2)> symmetrised (T3 re-derived it by Wick
contraction: two surviving placements, four R lines, zero C lines, the self-contraction
R(z;z) = 0). If gamma^(1) has no B-mode, this term has no B-mode. The first B-mode is
<gamma_B^(2) gamma_B^(2)>: the Gaussian FF piece (unperturbed-path post-Born analogue)
plus the connected four-point cumulant F^2 K4 (Order 3 of the expansion). A zeta-linear
B-mode first appears at F^3 K3 (Order 4), of relative size ~ (FK/O0)(FF/O0) ~ 3e-5 of O0.

**Why gamma^(1) is pure E, as an operator identity (T1).** gamma^(1)(n) = -int dl' w(l',l)
Psi_0(n,l') with a spin-independent weight (R_ij = delta_ij x scalar x directional delta,
path_int.tex ll. 234-248), and per shell Psi_0 is the trace-free screen Hessian, i.e.
eth^2 of the lightcone potential over chi^2 (the paper's own multiplier
sqrt(L^2(L^2-2))/chi^2, appendix.tex ll. 429-433, is the eth^2 eigenvalue). Hence
gamma^(1)_lm = e_l phi_lm with real e_l: B^(1)_lm = 0 for every realisation and any
statistics of the potential. No probability measure or almost-sure argument is needed;
the diagram's n2 leg is this zero operator applied to K3.

**2PCF form.** For any spin-2 field A cross-correlated with a pure-E field G,

    Re<A G*> = sum_l (2l+1)/(4pi) C_l^{A_E E} d^l_{2,2},    Re<A G> = sum_l (2l+1)/(4pi) C_l^{A_E E} d^l_{2,-2},

so xi_+^FK and xi_-^FK are the d_{2,2} and d_{2,-2} images of ONE spectrum, and
T_{22}^{-1}[xi_+] = T_{2,-2}^{-1}[xi_-]. The gamma^4 small-angle law of xi_-^FK is the
d^l_{2,-2} ~ sin^4(theta/2) kernel that every pure-E field shows (the O0 sweep has the
same law); it is the pure-E prediction, not evidence against it. Flat sky (Schneider,
van Waerbeke & Mellier 2002 eq. 27): xi_-(t) = xi_+(t) + int_0^t dv (v/t^2) xi_+(v)
[4 - 12 v^2/t^2], which uses only the converged (< 1 deg) part of FK.

**Loopholes examined and closed** (T1, T2, T3; details in their reports):

| loophole | status |
|---|---|
| formal cumulant expansion / finite-order truncation instead of a measure | not needed: B[gamma^(1)] is the zero operator |
| the paper's finite-range C_BB^O0 artefact taken as a real Order-0 B-mode | Cauchy-Schwarz bound 1.8e-4 (l=60) .. 2.2e-3 (l=1500) of C_EE^O0; the claim exceeds it by 35x .. 3x |
| E/B structure of a cross-correlation, symmetrisation, parity-odd <A_B G_E> | the parity-odd piece is EB (zero by parity and by construction); symmetrisation doubles the same expression |
| screen basis per ray, connecting-line frame, curved-sky transform | Psi_0 -> e^{-2i eps omega} Psi_0 is a proper spin-2 law; xi_+- are frame scalars; the transform is the standard one and passes on O0 |
| a piece of <gamma^(2) gamma^(2)> linear in zeta; tadpoles; <gamma^(3) gamma^(0)>; resummed diagrams | four fields give K4 + K2K2 only; <gamma^(1)> = R<Dsrc> = 0, <gamma_+^(2)> = 0 by isotropy; sigma^(sa) = 0; any external attached by a bare R to K has zero B |
| vector (frame-dragging) sector B_i restored on 2026-08-29 | gives a first-order B-mode in the full theory, but the paper's zeta is scalar-only ("B_i = 0 in every result"); magnitude B/E <~ 1e-5-1e-4 (Magi+2026; Lu, Ananda, Clarkson & Maartens 2009), 2-3 orders below the claim |
| tensor sector | r-suppressed, <~ 1e-4 of E at l ~ 100 for r = 0.1 |
| second-order shear sigma^(2) = -2 int theta^(1) sigma^(1) | generically HAS a B-mode (spin-0 modulation of an E field), but <gamma_B^(2) gamma_B^(1)> = 0 regardless |
| path deflection / reduced shear / image rotation | excluded from the paper's FK by construction (unperturbed path, Jacobian shear); rotation is also second order with omega^(1) = 0 |
| "vanishes at coincidence, gamma^4 at finite separation" | true of any pure-E xi_-; carries no E/B information |
| the paper's MC validation ("FK to about a percent") | shares the zeta table with the fold and tests the kappa-kappa entry only (mc_fk_complete SPEC.md ll. 54-66); it cannot see a spin defect |

## 3. The mechanism in the deployed vertex

* canoes `src/canoes/sachs/kappa3.py::_kappa3_limber_alpha_kernel` returned 2 pi J_0(|Q|)
  as soon as S = s1+s2+s3 == 0, BEFORE applying the residual per-leg phase
  spin_extra = exp(i(s1 l1_angle + s3 phi)). S == 0 is reached by (0,0,0), where the phase
  is 1, and by cancellation at the modulus channel (0,2,-2), where it is exp(-2i phi) and
  carries the relative orientation of the two spin-2 legs. Consequence: the high-l
  zeta_Bmod was bit-identical to zeta_TTT. Fixed in canoes commit `a95c0d6`
  ("fix(sachs): apply the residual spin phase when S = 0 by cancellation", Zheng Zhang,
  2026-09-02 14:55:09 +0100; +162-line regression test). The shortcut entered in `42e4b38`
  (2026-05-28) and became wrong in `63773dd` a week later.
* The paper's `table_permclosed.npz` was built 2026-08-26 09:46:28 and the FK sweep
  `xi_C_corr_op_K_limber_FK_cut15360_permfix.npz` 2026-08-26 09:48:50, both BEFORE the fix;
  `figures/cl_EB_polarization.pdf` (2026-09-02 17:52:04) was redeployed from the unchanged
  sweep. No post-fix table or sweep exists in the reproduction package or in SFT-WL-B.
* Measured on the deployed table (T1 `vertex_table_test.py`, T3 `check_table_econsistency.py`):
  zeta_Bmod/zeta_TTT = 1.0000-1.0004 on the shear-placement rows at every shell (the
  residual is the LOW branch, l <= 60, which carries the phase);
  zeta_TPP(actual)/zeta_TPP(pure-E from zeta_Bmod) = 0.46-0.58 at 2', 0.42-0.52 at 5',
  0.38-0.50 at 12', 0.28-0.50 at 30' (z = 0.1 .. 5.7).
* With the fixed kernel the identity T22^{-1}[Bmod] = T_{2,-2}^{-1}[TPP] holds to 1e-16
  (T3 `kernel_identity_test.py`: on the collapsed geometry the HIGH kernels are
  TTT 2 pi J0, Bmod 2 pi J0 cos 2phi, TPP 2 pi J4 cos 2phi, so Bmod and TPP are the J0
  and J4 transforms of one moment C(v) for ANY bispectrum and ANY l-window).
* Closing the loop (T1 `rezeta_phase_test.py`, rezeta's flat-sky Limber engine, hard
  window (60, 15360], shear placement): zeta_TPP/pred = 1.002, 0.999, 1.000, 0.999,
  0.999, 1.000, 1.000, 1.000 at 1', 2', 3', 5', 8', 12', 20', 30' (z = 0.45) and the same
  to 0.2% at z = 3.10 WITH the phase; WITHOUT it, 0.61 -> 0.43 (z = 0.45) and
  0.59 -> 0.30 (z = 3.10), matching the deployed table.
* Which channel is wrong: zeta_Bmod (feeds xi_+ and, through G_00 = -2(TTT+Bmod), the
  kappa-kappa entry), inflated by 1/(0.20..0.44) = 2.3-5x shell by shell. zeta_TPP (feeds
  xi_-) is the sound channel up to its 23% permutation ambiguity (F27). The paper's
  "EE-BB" curve is therefore, up to transform artefacts, the true FK E-mode.

## 4. Numerical evidence

### 4.1 Products-level test (N2, `n2_products/`)

The paper's transform on the paper's sweeps (z_s = 5), BB/EE and the E-consistency
ratio D/S = T_{2,-2}[xi_-] / T_{2,2}[xi_+] (pure E: 1):

| l | O0 BB/EE | O0 D/S | FF BB/EE | FK BB/EE | FK D/S | FK_june BB/EE | f_BB = BB_FK/EE_O0 | EE_FK/EE_O0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 60 | +0.0003 | 0.9994 | 0.023 | **0.951** | 0.025 | 1.04 | 6.4e-3 | 0.67% |
| 300 | -0.0016 | 1.0033 | 0.034 | 0.618 | 0.236 | 1.81 | 6.3e-3 | 1.02% |
| 1500 | -0.0404 | 1.0843 | 0.019 | **0.420** | 0.408 | 1.15 | 6.5e-3 | 1.54% |

The analysis3 module imported directly and the verbatim port agree to 3e-12. The June
(cut1000) sweep has xi_11 = xi_22 to 1e-11 (xi_- = 0), i.e. the earlier "C_EE^FK =
C_BB^FK" was the same artifact at a lower cutoff. Robustness: across 23 large-angle / DC /
small-angle treatments the l = 1500 value stays in 0.38-0.54 (intrinsic to the vertex),
while the l = 60 value moves over 0.53-1.17 and the O0 control returns exactly 1.000 when a
pure-E 2PCF is truncated at 1 deg (d^60_{2,-2} has no support below ~1 deg): the paper's
0.95 at l = 60 is a statement about the unconverged tail. Pure-E prediction of xi_-^FK from
xi_+^FK (two routes agreeing to 7%): xi_-^pred / xi_-^deployed = 1.8-2.5 (5'), 2.5-2.7
(10'), 3.8 (30'), 5.0 (60'); the O0 control gives 0.98-1.00. Small-angle exponents of xi_-
over 0.5'-2': O0 3.85, FF 3.71, FK 2.44, FK_june 2.00 (band-limit properties, not physics).

### 4.2 Exact-solver test (N1, `n1_sachsray/`)

sachsray (linear-Jacobi form, JAX/diffrax, float64, rtol 1e-9), full sky at nside 64
(4 seeds) and 128 (2 seeds), vacuum background, AR(1)-correlated shells of a scalar
potential; Phi00 = eth-bar eth psi, Psi0 = eth eth psi (pure E); antithetic tracing at
eps = +-1 separates odd (gamma^(1) + gamma^(3)) from even (gamma^(2) + gamma^(4)) orders;
non-Gaussian variant psi -> psi + q(psi^2 - <psi^2>) with skewness 0.8-1.1.

| quantity | result |
|---|---|
| gate C_BB^(1,1)/C_EE^(1,1) (identity convention) | 2e-11 .. 8e-9 |
| gate C_EE^(1,1) / [(l+2)(l-1)/(l(l+1)) C_kk^(1,1)] | 1.000000 +- 1e-6 |
| wrong conventions (flip W2 / swap / flip W1) | B/E = 0.8-1.3 (the test discriminates) |
| rms shear by order (nside 64) | g1 1.35e-2, g2 3.3e-4, g3 4.5e-6, g4 3.8e-8 |
| FK-type E-mode detection (NG) | rho_EE^(2,1) = -0.11 +- 0.03 (l 30-89), -0.070 at l 100-179 with 22400 modes (~10 sigma); C_EE^(2,1)/C_EE^(1,1) = -0.2% |
| **FK-type B-mode (NG)** r1 = C_BB^(2,1)/C_EE^(2,1) | **-2.0e-6 +- 3.0e-6 (l 2-9), +2.2e-6 +- 2.3e-6 (10-29), -3.2e-6 +- 7.0e-6 (30-89); +4.7e-5 at l 100-179 (nside 128, leakage level)** vs the claimed 0.42-0.95 |
| residual B in the raw even x odd cross | 1.5e-4 .. 2.0e-4 of EE, identically the 5th-order <g2_B g3_B> term (BB23/EE12 to 3 digits) |
| FF-type B/E r3 = C_BB^(2,2)/C_EE^(2,2) | 0.90 (l 2-9), 0.45 (10-29), 0.33 (30-99), 0.19 (100-179) |
| 2PCF of the order-(2,1) term | xi_- small-angle slope 4.00, xi_+ slope 0.00; T22^{-1}[xi_+] = T_{2,-2}^{-1}[xi_-] to the operator's floor |

### 4.3 The Takahashi et al. 2017 maps (E1, `e1_takahashi/`)

Native Nside 4096, lmax 4096, pixel windows removed, gamma2 as stored (convention gate:
C_EE/(F C_kk) = 0.9999, kappa-E correlation +1.0000; flipping gamma2 gives
C_EE = C_BB = C_kk/2). Cosmic variance of a full-sky band ratio: 5% at l = 60, < 1%
above l = 300. The maps contain every order of the lensing expansion and the full
non-Gaussian density field (kappa bispectrum validated on the same zs16 file to 3-11%,
SFT-WL-B FINDINGS_CB2).

zs16 (z_s = 1.033):

| l | measured C_BB/C_EE (total) | physical F C_ww/C_EE | post-Born theory | paper f_BB (z_s = 1) | paper / measured | paper / physical |
|---:|---:|---:|---:|---:|---:|---:|
| 60 | 4.86e-6 | 1.58e-6 | 1.54e-6 | 4.38e-3 .. 5.36e-3 | 900-1100 | 2770-3390 |
| 100 | 7.90e-6 | 2.44e-6 | 2.62e-6 | 4.18e-3 .. 5.10e-3 | 530-650 | 1710-2090 |
| 300 | 2.86e-5 | 6.74e-6 | 6.69e-6 | 3.66e-3 .. 4.47e-3 | 128-156 | 540-660 |
| 1000 | 9.40e-5 | 1.61e-5 | 1.74e-5 | 2.95e-3 .. 3.60e-3 | 31-38 | 180-220 |
| 1500 | 1.35e-4 | 2.03e-5 | 2.28e-5 | 2.66e-3 .. 3.25e-3 | 20-24 | 130-160 |

zs38 (z_s = 5.342, the paper's z_s = 5 case): total C_BB/C_EE = 1.52e-5 (60), 2.73e-5
(100), 8.14e-5 (300), 2.88e-4 (1000), 4.22e-4 (1500); paper/measured = 416-543, 221-289,
65-85, 15-19, 9-12; paper/physical = 740-970 .. 25-33. zs10 (z_s = 0.574): 2.53e-6 (60)
.. 7.19e-5 (1500).

What the map's B-mode is made of: the rotation omega is the clean post-Born observable
(C_ww vs the lens-lens theory computed here with CAMB halofit: 0.93-1.03 over l = 20-700
on zs16, 0.85-1.06 on the other planes); the omega-B correlation coefficient equals
-sqrt(F C_ww/C_BB) in every band of every plane, so B_lm = -sqrt(F) omega_lm + N_lm with
N an isotropic near-white component uncorrelated with omega (2-15x the physical part;
latitude-uniform; inferred to be the ray-tracer's tidal-matrix interpolation). The total
C_BB is therefore a strict upper bound on the physical B-mode; either way the exclusion is
by factors of 10-1000, not sigmas. Degrading the maps to Nside 1024 leaks E into B at the
(l/Nside)^3 level and must not be used for B above l ~ 200 (all quoted numbers are native).

### 4.4 An existing exact-Sachs N-body run (A2)

`~/Workspace/SFT-Sachs/analysis_exact_raytrace/` (June 2026; Level-1 exact Sachs
ray-trace of a gevolution box, 1920 Mpc/h, Ngrid 256, Nside 256, 8 seeds, z_s = 5):
C_BB/C_EE = 2.0e-5 .. 2.5e-5 for l = 8-200, i.e. ~300x below the Letter's f_BB at
z_s = 5, and the pure-FK cross estimator <kappa_Born (kappa_exact - kappa_Born)>/C_Born
is consistent with zero. Caveat: the ensemble file predates a factor-2 fix of Psi0 in that
repository, so treat it as indicative.

## 5. What the corrected FK looks like (estimate, not a refold)

With zeta_Bmod restored (physical/deployed = 0.20-0.44 on the shear placement, shell by
shell), xi_+^FK becomes the J_0 partner of the sound xi_-^FK: xi_+^FK/xi_+^O0 at 2'/5'/12'
falls from 1.79/1.68/1.46% to ~0.86/0.70/0.51%; in harmonic space FK EE/O0 EE at
l = 335/789/1500 goes from 1.06/1.28/1.54% to ~0.42/0.62/0.89% (the paper's present
"EE-BB" curve), and BB = 0. The kappa-kappa FK also changes (G_00 = -2(TTT+Bmod); up to
~25% at small gamma per T3), so the 1.3-1.7% headline needs the refold too. These are
sweep-level ratios; the true correction is a fold with shell-dependent factors and should
be produced by rebuilding the table with the fixed kernel (the n_phi rebuild the programme
already authorised, D-OP-02) and re-running the FK sweep.

## 6. Consequences for the draft, the Letter and the programme

Not applied anywhere (those trees are read only for this repository); listed for the author.

**Draft (STF_lensing).**
* conclusion.tex ll. 49-81 and the `fig: cl EB` caption: replace the B-mode paragraph by
  the correct statement (below); the "squared modulus ... blind to the orientation" reading
  is wrong for the shear-shear entries (the two Psi legs sit at different points; the
  modulus language is right only for the kappa-kappa entry).
* abstract: "populating the E- and B-modes in comparable measure" -> E-mode only.
* appendix.tex l. 538: gamma^4 "as gamma -> 0" is the d_{2,-2} kernel, not a channel
  property; insights.tex ll. 199-218 likewise.
* Fig. 12 / Section V C: after the refold the FK E-mode at 2'-12' is ~0.5-0.9% of Order-0
  (z_s = 5), not 1.3-1.7%.
* Validation paragraph (insights.tex ll. 278-289): state that the MC shares the table and
  tests kappa-kappa only; add the pure-E consistency relation as a test the FK must pass
  exactly (O0 passes at the percent level).

Suggested wording (T3):

> Delta C_l^{BB,FK} = 0. xi_+^FK and xi_-^FK are the d^l_{22} and d^l_{2,-2} transforms of
> the single spectrum Delta C_l^{EE,FK}; the gamma^4 law of xi_-^FK is the spin-4 kernel,
> shared with Order-0. The leading shear B-mode is fourth order in the driving field: the
> Gaussian FF term (two F vertices; the unperturbed-path analogue of post-Born lens-lens
> coupling, about 1/30 of its E-mode) plus the connected four-point cumulant through two F
> insertions (F F K4, allowed by the selection rule with k_F = 2, n = 4, p = 2).

**Letter and proposal (SFT-WL-B).** Every B-mode amplitude (0.95/0.42, f_BB, "8-16x",
S/N 12.2/12.1/10.2, 70%/26%, template overlaps, all Delta p/sigma, u_BB responses,
l_h = 972, the Fisher gains, both figures, proposal table rows 2-6, 8, 9) scales to zero
with r = B/E. What survives: F1 (no Order-0 B), F5/F6 (FF B/E = 0.032, EB = 0), the
Krause-Hirata ceiling, Cov(EE, BB) = 0, the WP-C Route B conclusions (Takahashi selection,
the B_kappa gate, the density-to-driver transfer validation, the Limber shell-thickness
law). A constructive reframing consistent with the theorem: the B-mode is sourced by the
FOUR-point cumulant (F^2 K4) on top of the Gaussian FF term, so it measures the matter
trispectrum in a collapsed configuration; its amplitude is at or below the Krause-Hirata
level and must be computed before any forecast is made.

**Acceptance gate for any rebuilt vertex.** BB^FK = 0 to quadrature accuracy; concretely
the 2PCF pure-E relation (`A1/theorem_check_2pcf.py`, flat sky) must give
xi_-/xi_-^pureE = 1 within the quadrature error for gamma >= 3' (O0 passes at 1-3%), and
the curved-sky D/S must be 1 to the O0 floor.

## 7. Audit of the work in this repository (A1)

**Retracted** (banners applied to `prototype/fk_mc/README.md`, the four FK scripts and
`notebooks/demo_fk_cl.ipynb`; memory updated): every FK number in
`prototype/fk_mc/outputs/fk_channels.npz`, "FK dominates O0 by 10-80x above 200'",
"C_l^FK/C_l^O0 up to 7x", the Figure-12 "modulus pair / cancels in EE-BB /
Delta C_EE^FK = Delta C_BB^FK" story. Causes: the June cut1000 vertex (frozen (1,1,1)
cosine corner: fold flat at 4.288e-6 to 70', xi_- = 0 to 1e-16, 4.5x low at 0.5'), the
single-leg `simulate_fk_vr` (captured share 0.25 at 0.5', negative beyond ~12') and the
nominal Q calibration (x3.9 inflation), all established in the paper's own 2026-08 audit
(`mc_fk_complete/NOTES.md` 3.1-3.3). The MC's xi_- ~ 0 was the coincident-leg identity of
the frozen corner. "kappa-E parity-forbidden" is also wrong (kappa-B and EB are). The FF
channel is ~0.5x the paper's fold with an unphysical constant xi_- floor.

**Kept:** the sachsray Jacobi-form robustness (with reduced relevance: the "Levy-peaked Q"
was the nominal-calibration artifact), the Gaussian bridge gate (kappa = int d-theta vs
the paper to ~2%), the verbatim curved-sky transform (`figure12.py`, used throughout this
audit), the Order-0 convention gate.

**sachsray as an engine: sound.** Observables and the tidal matrix verified; omega sign is a
convention (Takahashi 2017 use the opposite); float32 is adequate for B-mode power at
rtol <= 1e-6 (excess BB/EE < 1e-7 against a physical 2.3e-5), while rtol = 1e-5 leaks ~5%;
the streaming path works. The legacy `sachsfield.FullSkySource` draws Re Psi0 and Im Psi0
as independent scalar maps (no spin-2 structure), so the June demo's shear has E = B.

**Fixes made here:**
* `sachsray.fields.driving_from_potential_alms`: pure-E driving field from per-shell
  potential alms with the pinned sign rule (Phi00 <- -l(l+1) S_lm; Psi0 E-alm <-
  -sqrt((l+2)!/(l-2)!) S_lm, B = 0; same sign on both). Gate test
  `tests/test_fields_pure_e.py`: traced linear shear pure E (B/E < 2e-5 at kappa_rms
  2e-3), C_EE = F C_kk to 5%, kappa-E correlation > 0.99, and flipping W2 gives B/E ~ 1.
* `perFLRW.cosmology.spin2_power_ratio`: C_l^{Psi0}/C_l^{Phi00} = (L^2-2)/L^2; the module
  used the inverse at two call sites (0.1% at l = 45). Pinned by `tests/test_spin2_ratio.py`.

## 8. Simulations that solve the Sachs / Jacobi equations (L1, `L1_literature/codes_table.csv`)

Classes: EXACT = null geodesic integrated in the perturbed metric with the Jacobi/Sachs
system along the true path; MP = multi-plane with full lens-lens coupling (post-Born B and
rotation captured); BORN = shell sum on the unperturbed ray (B = omega = 0 by construction).

| code / suite | class | B or omega reported | public code | public data | shells or potential to drive our solver |
|---|---|---|---|---|---|
| Magrathea-Pathfinder (Breton & Reverdy 2022, 2111.08744; Breton & Fleury 2021, 2012.07802) | EXACT (RAMSES AMR, weak-field metric; finite 4-ray beam gives the full A incl. rotation) | rotation available, no spectra published | yes (github.com/vreverdy/magrathea-pathfinder) | via RayGal | RayGal gravity light cone (potential + force on cells), 1-100 TB, by arrangement |
| RayGalGroupSims (Rasera et al. 2022, 2111.08745) | EXACT (Magrathea finite beam) | catalogues store a11, a12, a21, a22 (omega recoverable) | - | yes: full sky z < 0.48; 2500 deg2 to z < 2; 400 deg2 to z < 10; HEALPix kappa/gamma/1/mu at Nside 2048-8192 | as above |
| gevolution ray tracer (Adamek+2019, 1812.04336; Lepori+2020, 2002.04024) | EXACT (scalar sector non-perturbative + frame dragging; Sachs optical equations on the stored metric light cone) | C_omega = CAMB post-Born on large scales; ellipticity B >= 4 orders below E; frame dragging gives a first-order B but no rotation | gevolution public; the Sachs tracer not | UNITY suite by arrangement (P. Bull) | yes: phi, psi, B_i shells = native Phi00, Psi0 (already done in SFT-WL-B wpc_cb3) |
| snapshot-raytracer (Magi, Lepori & Adamek 2026, 2603.24179) | EXACT (geodesic + geodesic deviation on one gevolution snapshot, full sky Nside 16384) | C_BB = C_omega ~ 1e-6..1e-7 of E at z_s = 0.5; ~5% relativistic corrections at l ~ 5 | yes (MIT; CUDA) | no | needs a gevolution snapshot (cb3 has phi, chi, T00) |
| CosmoGRaPH; Einstein Toolkit + mescaline; BiGONLight | EXACT in numerical relativity | convergence only | partly | no | no |
| Killedar+2012 (1110.4894); HRT in pmwd (2405.12913) | EXACT weak-field bundles / on-the-fly geodesics | omega "orders of magnitude smaller" | pmwd public, ray tracing not yet | no | no |
| Takahashi+2017 (1706.01472) | MP curved sky, Jacobian recursion incl. omega | shear B >= 3 orders below E (with numerical E->B leakage, worse at Nside 4096); omega agrees with Krause-Hirata | no | yes: 108 realisations, kappa/gamma1/gamma2/omega at 38 source planes to z = 5.34 plus CMB, Nside 4096/8192 (three planes on disk) | NO density shells released |
| Hilbert+2009 (Millennium); Becker 2013 CALCLENS; Fabbian+2018; DORIAN/MTNG (Ferlito+2023/24); LensTools | MP | B/E <~ 1e-3, < 1e-5 at large scales (Hilbert); C_BB = C_omega to <= 1% (Becker); omega vs post-Born < 5% (Fabbian) | CALCLENS, DORIAN, LensTools public | MTNG shells "on request" | DORIAN/CALCLENS could be run on the same shells we feed sachsray |
| AbacusSummit, CosmoGridV1, Gower Street, FLAMINGO, MICE, Euclid Flagship | BORN | none | - | shells public for CosmoGrid (69 shells, nside 2048, 2500 cosmologies), Gower Street (~100 shells, 791 runs), FLAMINGO (Nside 16384 mass shells to z = 3/5), AbacusSummit (particle light cones) | density shells -> Limber-Poisson potential per shell |

Recommendations: (a) to measure the total B-mode and rotation, the Takahashi planes on
disk are the strongest end-to-end target (done here); RayGal's catalogues give an
independent exact-geodesic estimate at z_s ~ 1-2 and ~5 on the cut sky. (b) To drive our
own exact Sachs solve, gevolution metric light cones (our cb3 run; UNITY by arrangement)
give Phi00 and Psi0 natively; RayGal's gravity light cone gives the potential at high
resolution; CosmoGrid / Gower Street / FLAMINGO give density shells for the Limber-Poisson
transfer already validated in wpc_cb3. (c) To compare an exact-Sachs code with ours,
snapshot-raytracer is the closest analogue (needs an NVIDIA GPU or a port); Magrathea builds
on macOS; CALCLENS/DORIAN on identical shells isolate the thin-lens error.

## 9. The N-body reproduction: what is done and what is next

Done in this audit: the direct empirical test (section 4.3) and the exact-solver structural
test (section 4.2). The N-body maps do not reproduce the paper's B-mode, and the theorem
says they cannot.

Next steps, in order of value:
1. Rebuild the vertex table with the fixed canoes kernel, refold O0/FF/FK, regenerate
   Figs. 12 and cl_EB, and apply the pure-E acceptance gate (section 6). This is the
   correction of the paper.
2. sachsray on N-body shells (the "unperturbed-path Sachs" solve the formalism describes):
   add `driving_from_shells` (density or potential shells -> Phi00, Psi0 per shell via
   `driving_from_potential_alms`, Limber-Poisson for density), a per-shell exact 2x2
   propagator for thick shells, and a lambda0/y0 option in `trace_rays`; run on the
   gevolution cb3 light cone (or CosmoGrid/Gower Street shells), Gaussianise the shells
   (phase randomisation at fixed power) and take C_BB[N-body] - C_BB[Gaussian] to isolate
   the genuine non-Gaussian (K4) B-mode, which is the SFT's next-order prediction.
3. Compare sachsray with snapshot-raytracer or Magrathea on the same snapshot to certify the
   engine's post-Born omega and B against an exact-geodesic code.

## 10. Caveats and open items

* The corrected FK amplitudes in section 5 are sweep-level estimates; the refold has not
  been run (no post-fix table exists).
* N1 tests the structure (pure-E linear response => no order-3 B-mode), not the physical
  amplitude of FK (its bispectrum is a toy).
* E1's decomposition of the map's total B-mode into "physical = F C_ww" plus a white
  ray-tracer component is inferred from its statistical signature, not from the code; the
  exclusion does not depend on it.
* The SFT-Sachs numbers (4.4) predate a factor-2 fix in that repository.
* The 23% permutation ambiguity of zeta_TPP (F27) and the n_phi quadrature defect (D-CN-04)
  remain as second-order issues in the vertex; neither can manufacture a factor 2-5.
* The completeness critic of the workflow did not run (session limit); the parent session
  performed that check by hand: the ten reports contain no contradictions (one verdict
  label, T3's "refuted", refers to the paper's claim, not to the theorem), and the two
  weakest links (the SFT-Sachs provenance and the refold estimate) are flagged above.

## 11. Provenance

Workflow run `wf_7036f78f-238` (2026-09-03), 10 agents, 555 tool uses, 31 minutes.
Scripts and outputs: `T1/{eb_consistency,vertex_table_test,rezeta_phase_test}.py`,
`T2/{theorem_test,theorem_cross,econsistency}.py`, `T3/{check_fk_eb,check_table_econsistency,kernel_identity_test}.py`,
`n1_sachsray/{n1_theorem_sachsray,n1_twopcf_check}.py` (+ outputs/*.json, tables, figures;
raw per-seed npz not tracked), `n2_products/{n2_products_eb,n2_crosscheck_analysis3}.py`
(+ results.json, tables.md, figures), `e1_takahashi/{e1_takahashi_bmode,post_born_theory,e1_bmap_diagnostic,e1_report}.py`
(+ outputs/results.json, tables.md, spectra npz, figures),
`A1/{theorem_check_2pcf,sachsray_eb_basis_test,float32_test,numerical_b_floor}.py` (+ logs),
`L1_literature/codes_table.csv`, `L2_literature/be_ratios_literature.csv`.
Read-only inputs: the draft and its reproduction package (`~/Documents/MyDrafts/STF_lensing`),
SFT-WL-B (`~/projects/SFT-WL-B`, Takahashi maps checksummed in `data/PROVENANCE.md`),
canoes (`~/projects/angular_statistics/canoes`), rezeta (`~/projects/rezeta`),
SFT-Sachs (`~/Workspace/SFT-Sachs`).
