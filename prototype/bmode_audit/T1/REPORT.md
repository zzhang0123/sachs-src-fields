
# T1 report: can a term linear in the three-point cumulant carry shear B-mode auto-power?

**Verdict: the theorem HOLDS (confidence 0.97).** The FK contribution to C_l^{BB} is identically zero. The paper's B/E = 0.95 -> 0.42 is entirely the dropped pair phase in the modulus channel of the deployed vertex table, as shown by three independent numerical checks below. All inputs were read-only; scripts and outputs are under `/Users/zzhang/projects/SFT/src-field/prototype/bmode_audit/T1/` (`eb_consistency.py`, `vertex_table_test.py`, `rezeta_phase_test.py`, `outputs/eb_consistency.npz`).

## 1. What the FK diagram is (paper's own definitions)

* Sachs system `theta' = -theta^2 - sigma_+^2 - sigma_x^2 + Phi00`, `sigma_+' = -2 theta sigma_+ + W1`, `sigma_x' = -2 theta sigma_x + W2` (sachs_dynamics.tex:118-120); coupling table `F_111 = F_122 = F_133 = -1`, `F_212 = F_313 = -2` (sachs_dynamics.tex:158-172); mean-field subtraction makes `<Dsrc> = 0` exactly (sachs_dynamics.tex:204) and `A_ij = (F_ijk + F_ikj) s^(sa)_k = -2 theta^(sa) delta_ij` (sachs_dynamics.tex:219); `sigma^(sa) = 0`, `<Psi_0> = 0` (sachs_dynamics.tex:226-231).
* Response propagator `R_ij(n,l; n',l') = delta_ij delta(n'-n) Theta(l'-l) exp(-2 int theta^(sa))` (path_int.tex:234-248): **proportional to the identity in component space and direction-diagonal**.
* Observables: `kappa = -int Dsachs_1 dl`, `gamma_+ +- i gamma_x = -int (Dsachs_2 +- i Dsachs_3) dl` (cosmology.tex:862, 878); 2PCFs in the connecting-great-circle frame (cosmology.tex:893-908).
* Order-2 two-point function `1/2 <Dsachs Dsachs (V_F^2 + 2 V_K3 V_F)>`, surviving topologies FF and FK, KK vanishing (path_int.tex:413-478); FK's propagators are all `R`, whose delta-function pins the vertex legs to the two observed directions (insights.tex footnote in "Leakage of driving-field non-Gaussianity"; appendix.tex:362-372).

Iterating `Dsachs = R(Dsrc + F Dsachs Dsachs)` gives `Dsachs^(1) = R Dsrc`, `Dsachs^(2) = R F (R Dsrc)(R Dsrc)`, so the FK diagram is exactly

    FK_ab(n1,n2) = <gamma_a^(2)(n1) gamma_b^(1)(n2)> + (1 <-> 2),   linear in K3 = <Dsrc Dsrc Dsrc>.

For the shear entries the response leg of F must be `sigma`, so only `F_212, F_313` contribute: `sigma_+^(2)(n1) = R[-2 theta^(1) sigma_+^(1)](n1)`. Hence the K3 legs are **(Phi00, Psi_+) at n1 and Psi_+ at n2**; the collapsed geometry is "pair (T,P) at n1, soft P at n2" (cos01 = 1 rows of the table; rezeta's slot 0), and

    xi_+^FK ~ <Phi00 Psi_+ Psi_+'> + <Phi00 Psi_x Psi_x'> = Re<[Phi00 Psi_0](n1) Psi_0*(n2)>,
    xi_-^FK ~ <Phi00 Psi_+ Psi_+'> - <Phi00 Psi_x Psi_x'> = Re<[Phi00 Psi_0](n1) Psi_0(n2)>,

(the paper's eqs appendix.tex:500-523 with one Psi leg at n2), matching D-RZ-44's `xi_+ = -4 zeta_Bmod`, `xi_- = -4 zeta_TPP`.

## 2. (a) The linear leg has no B-mode: an operator identity, not an a.s. statement

`gamma^(1)(n) = -int dl' w(l',l) Psi_0(n,l')` with a spin-independent weight (R = identity x scalar). Per shell `Psi_0` is the trace-free screen Hessian = `eth^2` of the lightcone potential over `chi^2`; the paper's own per-multipole multiplier `sqrt(L^2(L^2-2))/chi^2 = sqrt((l+2)!/(l-2)!)/chi^2` (appendix.tex:429-433) is the `eth^2` eigenvalue. Therefore `gamma^(1)_{lm} = e_l phi~_{lm}` with real `e_l` and a real scalar `phi~`: **`B^(1)_{lm} = 0 for every realisation, for any statistics of the potential** (Gaussian or not; the potential's non-Gaussianity is what K3 encodes, and it does not change the spin structure). Consequently in the formal cumulant setting the FK diagram is a linear functional of K3 whose n2 leg is `B[R Dsrc] = 0` as an operator; no probability measure is needed and finite-order truncation is irrelevant. The a.s. argument (E[X^2]=0 => X=0 a.s. => E[XY]=0) is a valid fallback if only the moment were known.

**Finite-range artefact.** The paper's C_BB^O0 "up to a mild finite-range effect" is a property of the truncated transform, not of the field. Even taken at face value, Cauchy-Schwarz `|C_BB^FK| <= 2 sqrt(C_BB^O0 C_BB^(2,2))` with `C_BB^(2,2) ~ C_BB^FF` gives (paper's transform on the production sweeps, z_s = 5):

| l | f_BB^FK (deployed) | C_BB^O0/C_EE^O0 | C_BB^FF/C_EE^O0 | bound | claim/bound |
|---|---|---|---|---|---|
| 60 | 6.4e-3 | 3.0e-4 | 2.8e-5 | 1.8e-4 | 35 |
| 270 | 6.2e-3 | 1.4e-3 | 4.5e-5 | 5.1e-4 | 12 |
| 789 | 6.6e-3 | 9.7e-3 | 5.6e-5 | 1.5e-3 | 4.5 |
| 1500 | 6.5e-3 | 4.0e-2 | 2.9e-5 | 2.2e-3 | 3.0 |

Note the O0 artefact itself: 3e-4 at l = 60 (the low-l finite-range effect the paper mentions) but **4% at l = 1500** from the 0.5' inner cutoff, which the paper does not mention.

## 3. (b) Cross-correlation E/B structure on the sphere

For spin-2 fields A, G rotated into the connecting-line frame (Ng & Liu 1999; Chon et al. 2004 conventions),

    <A(n1) G*(n2)> = sum_l (2l+1)/(4pi) [ C^{A_E G_E} + C^{A_B G_B} + i (C^{A_B G_E} - C^{A_E G_B}) ] d^l_{2,2}(gamma)
    <A(n1) G(n2)>  = sum_l (2l+1)/(4pi) [ C^{A_E G_E} - C^{A_B G_B} + i (C^{A_E G_B} + C^{A_B G_E}) ] d^l_{2,-2}(gamma)

with `xi_+ = Re<AG*>`, `xi_- = Re<AG>`, and the imaginary parts the t-x correlations. With G = gamma^(1) pure E (`G_B = 0`): both real parts are images of the **same** `C^{A_E E}` under `d_{22}` and `d_{2,-2}`; the only other structure is `C^{A_B G_E} = C^{EB}`, confined to the parity-odd t-x parts, zero for a parity-even ensemble and identically zero in the paper's vertex (appendix.tex:496-498; the EB panel of `plot_cl_EB_polarization.py`). The symmetrisation `(1 <-> 2)` reproduces the same expression (real symmetric spectra, `d^l(gamma)` depends only on gamma) and merely doubles it. With the paper's inversion `C^{EE} +- C^{BB} = 2pi int xi_+- d^l_{2,+-2} sin(theta) dtheta` (`plot_analysis3_cl_decomposition.py:242-288`), `C_BB^FK = 0` identically. Flat-sky form: `xi_+ = int l dl/2pi C J_0(l theta)`, `xi_- = int l dl/2pi C J_4(l theta)`, hence (Schneider, van Waerbeke & Mellier 2002 eq. 27; verified via `J_4 = J_0 - 8J_1/x + 24J_2/x^2`, self-test 6e-6)

    xi_-(theta) = xi_+(theta) + int_0^theta dv (v/theta^2) xi_+(v) [4 - 12 v^2/theta^2]     (pure E)

which needs `xi_+` only on `[0, theta]`, i.e. only the converged (gamma < 1 deg) part of FK. **The gamma^4 rise of xi_-^FK (conclusion.tex:63-68; insights.tex:199-205) is the `d^l_{2,-2} ~ sin^4(theta/2)` / `J_4 ~ (l theta)^4` kernel: it is the pure-E prediction, not evidence against it.**

## 4. (c) Basis conventions

`Psi_0 -> e^{-2 i eps omega} Psi_0` under a dyad rotation (cosmology.tex:483-494) is the spin-2 law; the per-ray dyad is a spin frame and `Psi_0 = eth^2 phi/chi^2` holds in any of them; `xi_+-` in the connecting-great-circle frame (cosmology.tex:893-908) are frame scalars; the transform (Wigner-d seeds `cos^4, sin^4(theta/2)`, `2pi sin theta` measure, orthonormality self-test) is the standard one. The theorem does not depend on the basis. A second-order dyad rotation adds a term `~ (angle) x gamma^(1)` to `gamma^(2)`, which the theorem is indifferent to.

## 5. (d) Other routes to an O(zeta) B-mode, all closed

* `<gamma^(2) gamma^(2)>`: four driving fields -> K4 + K2 K2; a K3 leaves a lone field with `<Dsrc> = 0` exactly.
* Tadpoles: `<gamma^(1)> = R<Dsrc> = 0`; `<gamma_+^(2)> ~ int <theta^(1) sigma_+^(1)>(coincident) ~ <Phi00 Psi_+>(0) = 0` by isotropy (only the kappa mean survives: the FF plateau footnote in insights.tex).
* `<gamma^(3) gamma^(0)>`: `gamma^(0) = sigma^(sa) = 0`.
* Any resummed diagram with one external attached by a bare R to a K leg: zero B at that point. BB needs both externals on F vertices: first Gaussian BB = FF (`<gamma_B^(2) gamma_B^(2)>`; paper's own `C_BB^FF ~ 3-6e-5 C_EE^O0`); first non-Gaussian BB = `F^2 K4` (Order 3); first zeta-linear BB = `F^3 K3 ~ O(zeta P)` (Order 4), relative size `~ (FK/O0)(FF/O0) ~ 3e-5` of O0, 150x below the claim. So the paper's "first non-vanishing B-mode is 4th order in the field" is exactly right, and it is the FF/K4 term, not FK.

## 6. Numerical record (production npz, paper's transform)

**6.1 Reproduction.** FK BB/EE = +0.951 (l=60), +0.420 (l=1500) [paper: 0.95, 0.42]; June FK (cut1000): 1.00-1.15 (the old "EE = BB"); FF: 0.02; O0: 3e-4 -> -4e-2.

**6.2 Flat-sky pure-E test on the sweeps** (ratio of actual xi_- to the value forced by xi_+):

| theta | 2' | 3' | 5' | 8' | 12' | 20' | 30' | 60' |
|---|---|---|---|---|---|---|---|---|
| O0 (pure E, control) | 1.17 | 1.03 | 1.02 | 1.02 | 1.01 | 1.00 | 1.00 | 1.00 |
| FF (has its own B) | 1.46 | 0.89 | 0.92 | 0.90 | 0.91 | 0.91 | 0.92 | 0.94 |
| **FK (deployed cut15360 permfix)** | **0.48** | **0.45** | **0.42** | **0.39** | **0.35** | **0.30** | **0.26** | **0.20** |
| FK June (cut1000) | xi_- ~ 1e-15, i.e. 0 | | | | | | | |

(2' carries a <20% sensitivity to the [0, 0.5'] extrapolation; >= 5' is < 3%.) The FK deficit is `(1-r)/(1+r)` with `r` the quoted BB/EE: the "B-mode" is the inconsistency between the two channels.

**6.3 Vertex table, shear placement rows (cos01 = 1; 171 separations x 16 shells).** `zeta_Bmod/zeta_TTT = 1.0000-1.0004` at every shell (pair phase dropped). `zeta_TPP(actual)/zeta_TPP(pure-E from zeta_Bmod)` = 0.46-0.58 at 2', 0.42-0.52 at 5', 0.38-0.50 at 12', 0.28-0.50 at 30' (z = 0.1 ... 5.7). Small-angle slope of the deployed `zeta_TPP` is 3.4 over 0.4'-1' (the 0.344' floor, F27b), against the exact 4.

**6.4 Closing the loop with rezeta's flat-sky Limber engine (`~/projects/rezeta/rezeta/limber_vertex.py`, hard window (60, 15360]), slot 0 = shear placement.** Its residual phase for Bmod and TPP is the same `e^{2i(phi_b - phi_c)}`, with Bessel orders 0 and 4, exactly the structure of section 3:

| | 1' | 2' | 3' | 5' | 8' | 12' | 20' | 30' |
|---|---|---|---|---|---|---|---|---|
| z=0.45, **phase on**: TPP/pred | 1.002 | 0.999 | 1.000 | 0.999 | 0.999 | 1.000 | 1.000 | 1.000 |
| z=0.45, phase off (deployed): TPP/pred | 0.61 | 0.60 | 0.59 | 0.57 | 0.54 | 0.51 | 0.47 | 0.43 |
| z=3.10, **phase on**: TPP/pred | 1.002 | 1.001 | 1.001 | 1.000 | 0.999 | 1.000 | 0.999 | 1.001 |
| z=3.10, phase off (deployed): TPP/pred | 0.59 | 0.57 | 0.54 | 0.50 | 0.44 | 0.40 | 0.34 | 0.30 |
| z=3.10, Bmod(phased)/Bmod(deployed) | 0.40 | 0.35 | 0.31 | 0.28 | 0.24 | 0.22 | 0.20 | 0.20 |

The phased pair satisfies the pure-E identity to 0.1-0.2%; the deployed convention reproduces the table's deficit. (At rezeta's slot 1, pair = two Psi legs and soft Phi00, the theorem does not apply because the soft leg is spin 0, and the phased pair indeed fails the test there; that slot is the kappa-kappa placement, not the shear one.)

**6.5 What the corrected FK looks like (estimate, not a refold).** `xi_-^FK` is sound; `xi_+^FK` must be its J_0 partner: `xi_+^FK/xi_+^O0` at 2'/5'/12' falls from 1.79/1.68/1.46% to ~0.86/0.70/0.51%; in harmonic space the corrected FK E-mode is the paper's "EE-BB" curve: FK EE/O0 EE at l = 335/789/1500 goes from 1.06/1.28/1.54% to 0.42/0.62/0.89%, and BB = 0 (the paper's 0.65% of O0 in BB at every l is the defect). Caveat: the true correction is a fold with shell-dependent factors 0.2-0.44; the numbers above use the sweep-level ratio.

## 7. What the paper got wrong (file:line)

1. conclusion.tex:49-71 and the Fig. caption :83-100; insights.tex:199-218: the difference of `<Phi00 Psi_+ Psi_+'>` and `<Phi00 Psi_x Psi_x'>` is read as a B-mode; for a diagram with a bare linear leg the two 2PCFs are the d_{22}/d_{2,-2} images of one spectrum, so "gamma^4 rather than absent" is the pure-E kernel.
2. The deployed vertex (`callables/kappa3_vertex/equal_time_limber_cut15360_permaware/table_permclosed.npz`) has `zeta_Bmod = zeta_TTT` on the shear-placement rows: canoes `kappa3.py:1789-1835` fast path (now guarded, after the table was built). This inflates `xi_+^FK` by 2.3-5x per shell and is the entire B-mode. `docs/message_from_rezeta_modulus_phase.md` found the defect but quoted slot-1 magnitudes (0.48-0.99); at the shear placement the physical Bmod is 0.2-0.44 of the deployed one.
3. The 2026-08-29 rewording replaced `C_EE^FK = C_BB^FK` (June table, `zeta_TPP = 0`, also inconsistent: it would force `xi_+^FK = 0`) by `B/E = 0.95 -> 0.42`; both violate the identity.
4. insights.tex:280-289: the MC validation shares the table and cannot see the spin defect (the paper says so).
5. The Letter (SFT-WL-B `letter/main.tex:46-56, 103-106, 149`) rests on the FK B-mode; the "irreducible scalar B-mode at second order, 8-16x the known ceiling" does not exist. The scalar B-mode remains 4th order (FF-type `~3-6e-5` of O0 in the paper's own numbers, plus K4), i.e. Krause-Hirata level, which is what an N-body ray-trace should show.
6. Side finding for the programme: rezeta's `FK_SLOT = 1` and F19/F28/D-RZ-44's "selection rule 2 => zeta_TPP = 0, needs a hexadecapole response" describe the kappa-kappa placement; for `xi_-` the placement is slot 0, the hard pair has m = 2 (partnered by the tidal response R_K), and the gamma^4 rise is the J_4 of the soft Psi leg. No new response is needed; `zeta_TPP` is fixed by `zeta_Bmod`.

## 8. Correct statement

For every term of C_l^{BB} in which one external shear leg is the linear response, in particular the FK diagram (the only Order-2 term linear in the three-point cumulant), `C_l^{BB} = 0` identically; `xi_+^FK` and `xi_-^FK` are the `d^l_{2,2}` and `d^l_{2,-2}` images of one cross-spectrum `C_l^{[Phi00 Psi0]_E, E}` (equivalently `zeta_Bmod`, `zeta_TPP` are the J_0, J_4 images of one hard-pair moment). The first non-vanishing scalar B-mode is `<gamma_B^(2) gamma_B^(2)>` = FF (Gaussian) + F^2 K4 (non-Gaussian); a zeta-linear B-mode first appears at Order 4 (`F^3 K3`, `~3e-5` of O0).