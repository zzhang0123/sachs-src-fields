# B-mode audit (2026-09-03)

Audit of the claim, in the revised *Statistical Field Theory for Weak
Gravitational Lensing* draft and the spin-off Letter, that the three-point
cumulant (FK diagram) sources a cosmic-shear B-mode with
C_BB/C_EE = 0.95 (l=60) -> 0.42 (l=1500).

**Read [`REPORT.md`](REPORT.md) first.** Verdict: the FK B-mode is zero
identically; the deployed number is a dropped spin phase in the vertex table.

| dir | agent | what |
|---|---|---|
| `T1/`, `T2/`, `T3/` | three adversarial refuters | the theorem C_BB^FK = 0 (harmonic/probability; Sachs-physics loopholes; the paper's own diagrammatics + vertex construction) |
| `n1_sachsray/` | N1 | direct test with the exact nonlinear Jacobi solver on the full sky (pure-E and skewed driving fields) |
| `n2_products/` | N2 | products-level E/B test on the paper's own O0/FF/FK sweeps; reproduces 0.951/0.420; pure-E consistency; robustness |
| `e1_takahashi/` | E1 | C_EE, C_BB, C_omega on the Takahashi 2017 full-sky ray-traced maps (z_s = 0.57, 1.03, 5.34) vs the paper's prediction and post-Born theory |
| `A1/` | A1 | audit of `prototype/fk_mc` (June 2026) and of `sachsray` as an N-body engine; convention, float32 and B-floor tests |
| `A2/` | A2 | audit of the SFT-WL-B programme repository (what is reusable, what rests on the claim) |
| `L1_literature/` | L1 | simulation codes that solve the Sachs / Jacobi equations, and public products |
| `L2_literature/` | L2 | lensing B-modes beyond Born and from non-Gaussianity: quantitative B/E ratios |

Everything outside this repository (the draft, its reproduction package,
SFT-WL-B, canoes, rezeta, SFT-Sachs) was read only.
