# Moments Cross-Check: sft-wick (Feynman diagrams) vs sachsfield (Monte Carlo ODE)

## Purpose

This notebook validates two completely independent ways of computing the statistical moments of an observable in a gravitational lensing scalar-field theory:

1. **sft-wick** — a symbolic Feynman diagram engine that enumerates diagrams order-by-order in perturbation theory, then numerically integrates each diagram using quasi-Monte Carlo (QMC).
2. **sachsfield** — a brute-force ODE solver that directly evolves the nonlinear fluctuation equations on the sphere, then estimates moments by averaging over many realizations and pixels.

If both are correct, they must agree (up to MC noise and perturbative truncation).

---

## The observable

The quantity being compared is:

$$\mu_n = \left\langle \left[\int_0^{\lambda_f} \xi_1(\lambda)\,d\lambda \right]^n \right\rangle$$

where $\xi_1$ is the first component of the fluctuation field (the direction-dependent deviation from the saddle-point solution). This is the integrated convergence fluctuation — the physical quantity that controls lensing magnification.

---

## Section 1 — Parameters (Cells 2–3)

A simplified toy model is set up:

- **Source spectrum**: $C_\ell = A/[\ell(\ell+1)]$ with $A = 10^{-15}$, the same for all 3 field components, independent of $\lambda$. This is a scale-invariant spectrum (equal power per log-$\ell$), chosen to be analytically tractable.
- **Lambda correlation**: $\rho(\Delta\lambda) = e^{-c\,\Delta\lambda^2}$ with $c = 0.01$, giving a correlation length $\sim 1/\sqrt{c} = 10$ in affine parameter. Sources at $\lambda$-separations $\gg 10$ are uncorrelated.
- **Background source**: constant $\bar{s}_1 = -10^{-6}$, $\bar{s}_2 = \bar{s}_3 = 0$. Only field 1 has a nonzero mean source.
- **F tensor**: the quadratic coupling, with $F_{111} = -1/2$, $F_{011} = F_{022} = -2$, $F_{101} = F_{202} = -1$. This encodes how the fields couple nonlinearly in the Sachs ODE $\dot{x}_a = F_{abc} x_b x_c + s_a$.
- **F_code = $-i \times$ F_physical**: the MSR (Martin-Siggia-Rose) convention used by sft-wick, where the response field $\psi$ carries a factor of $i$.
- **S0**: the monopole-subtracted noise variance $S_0 = A \sum_{\ell=1}^{95} (2\ell+1)/[4\pi\,\ell(\ell+1)]$. This is the pixel-space variance of the source field at a single $\lambda$ slice — it collapses the angular structure into a single number that sft-wick uses for the scalar propagator.
- **sft-wick fields**: `phi` (physical, 3 components) and `psi` (response, 3 components) are declared, with a single cubic vertex $\psi \phi \phi$ coupling.

**Key point**: The field theory has a cubic interaction, so odd moments ($\mu_1, \mu_3, \mu_5$) are nonzero — they arise from loop diagrams (1-loop, etc.), not at tree level.

---

## Section 2 — Saddle Point and Response Propagator (Cells 4–5)

The saddle-point equation is:

$$\frac{d\theta}{d\lambda} = -\frac{1}{2}\theta^2 + \bar{s}_1$$

with the singular boundary condition $\theta \sim 2/\lambda$ as $\lambda \to 0^+$ (this is the $\chi_1$ saddle, related by $\theta = \chi_1^{(\mathrm{sa})}$). The function `solve_singular_theta` uses a $u$-substitution to regularize the $1/\lambda$ singularity (as noted in CLAUDE.md's discussion of the rescaled variable).

**Why this matters**: The saddle point $\theta(\lambda)$ defines the **response propagator**:

$$R(\lambda, \lambda') = \exp\!\left[\Phi(\lambda') - \Phi(\lambda)\right], \quad \Phi(\lambda) = \int_0^\lambda \theta(\tau)\,d\tau$$

$R$ is the retarded Green's function of the linearized fluctuation equation. It tells you how a source perturbation at $\lambda'$ propagates forward to $\lambda$. Since $\theta > 0$ (focusing), $\Phi$ grows, and $R < 1$ for $\lambda > \lambda'$ — fluctuations are damped by the background convergence.

### The Phi correction trick

The raw numerical integral $\Phi = \int_0^\lambda \theta\,d\tau$ is badly approximated by the trapezoidal rule near $\lambda = 0$ because $\theta \sim 2/\lambda$ diverges. The notebook uses the analytic asymptotic $\Phi(\lambda) = 2\ln(\lambda/\epsilon)$ for $\lambda < 5$ and stitches it to the numerical integral beyond that. Without this, the spline of $\Phi$ would be corrupted by a ~400,000 error in the first trapezoidal interval, causing $R$ to overflow.

A second saddle solve from `lam_start = 1` is done for the MC side, avoiding the singular region entirely (the MC evolution starts at $\lambda = 1$, not 0).

---

## Section 3 — Correlation Propagator C (Cells 6–7)

The **C propagator** (the dressed two-point function of $\xi$) is:

$$C_{ab}(\lambda_1, \lambda_2) = \int_0^{\lambda_1} d\lambda' \int_0^{\lambda_2} d\lambda''\; \kappa^{(2)}_{ab}(\lambda', \lambda'')\; R(\lambda_1, \lambda')\; R(\lambda_2, \lambda'')$$

where $\kappa^{(2)}_{ab}(\lambda', \lambda'') = S_0 \cdot e^{-c(\lambda'-\lambda'')^2} \cdot \delta_{ab}$ is the source noise kernel.

This double integral is computed as a matrix product:

1. Define a quadrature grid $q_m$ with 1000 points.
2. Build the weight matrix $V_{km} = e^{\Phi(q_m) - \Phi(\lambda_k)} \cdot \Delta q$ for $q_m \le \lambda_k$ (zero otherwise). This is $R(\lambda_k, q_m) \cdot \Delta q$.
3. Build the Gaussian kernel matrix $G_{mm'} = e^{-c(q_m - q_{m'})^2}$.
4. Then $C = S_0 \cdot V \cdot G \cdot V^T$, which is an efficient matrix multiply replacing the double integral.

The result is a $400 \times 400$ symmetric table, wrapped in a `RectBivariateSpline` and injected into sft-wick's `PropagatorCache`. This avoids sft-wick having to compute expensive `dblquad` integrals for each grid point.

**Physical meaning**: $C(\lambda_1, \lambda_2)$ is the two-point correlation of the integrated fluctuation at two different affine parameters. sft-wick uses it as the "line" in Feynman diagrams — every pair of external $\phi$ legs connected by a line contributes a factor of $C$.

---

## Section 4 — sft-wick Diagram Computation (Cells 8–9)

For each moment order $n = 0, 1, \ldots, 5$:

1. **`compute_moment`** symbolically enumerates all Feynman diagrams contributing to $\langle \phi_1(x_0) \cdots \phi_1(x_{n-1}) \rangle$ up to perturbative order 3 (tree + 1-loop + 2-loop + 3-loop).
2. **`integrate_diagrams`** numerically evaluates each diagram by QMC integration over the internal $\lambda$ vertices, using the pre-computed $R$ and $C$ propagators.

The results are stored order-by-order in `breakdown_wick[n]`:

- **Order 0 (tree)**: Wick contractions only — all pairings of the $n$ external legs via $C$ propagators. Nonzero only for even $n$.
- **Order 1 (1-loop)**: One vertex insertion. Gives the leading contribution to odd moments ($\mu_1, \mu_3$).
- **Order 2+ (higher loops)**: Corrections to even moments, sub-leading contributions to odd moments.

$\mu_0 = 1$ by MSR normalization (vacuum diagrams cancel).

**Adaptive sampling**: Higher perturbative orders contribute smaller corrections, so fewer QMC samples are used ($2^{16}$ for tree/1-loop, $2^{14}$ for 2-loop, $2^{12}$ for 3-loop).

---

## Section 5 — sachsfield Monte Carlo (Cells 10–12)

This is the brute-force validation:

1. For each of 30 seeds, generate an independent full-sky Gaussian random source field $\delta s(\hat{n}, \lambda)$ on a HEALPix grid (nside=32, 12288 pixels) with the prescribed $C_\ell$ and $\lambda$-correlation.
2. Evolve the **nonlinear** fluctuation ODE:

$$\dot{\xi}_a = M_{ab}\,\xi_b + F_{abc}\,\xi_b\,\xi_c + \delta s_a$$

where $M_{ab} = 2F_{a1b}\,\chi_1^{(\mathrm{sa})} + F_{ab1}\,\chi_1^{(\mathrm{sa})}$ is the linearized coupling to the saddle. The `linear_only=False` flag keeps the $F_{abc}\xi_b\xi_c$ term, so this captures **all** perturbative orders nonlinearly.

3. Compute $X(\hat{n}) = \int_{\lambda_\mathrm{start}}^{\lambda_f} \xi_1(\hat{n}, \lambda)\,d\lambda$ per pixel via trapezoidal quadrature.

Each realization gives 12288 samples of $X$. With 30 seeds, there are $30 \times 12288 \approx 370{,}000$ total samples.

**Why start at $\lambda = 1$, not 0**: The saddle has a $2/\lambda$ singularity at the observer. By $\lambda = 1$, $\theta \approx 2$ and the ODE is well-conditioned. The contribution from $[0, 1]$ is negligible for the fluctuations (though the notebook notes this causes a ~3% discretization bias for $\mu_2$).

---

## Section 6 — Moment Estimation and Comparison (Cells 13–15)

All pixel values are pooled into a flat array, and raw moments $\mu_n = \langle X^n \rangle$ are computed. Errors are estimated as the standard error of the seed-by-seed means (treating each seed as an independent sample).

Two comparison tables are printed:

1. **MC vs sft-wick tree level**: For even moments ($\mu_2, \mu_4$), the tree-level (Gaussian Wick contraction) should dominate. Odd moments are zero at tree level.
2. **MC vs sft-wick full**: The full perturbative result (tree + loops) should match the nonlinear MC to within MC error bars, provided perturbation theory converges.

The "sigma" column shows $|\mu_\mathrm{wick} - \mu_\mathrm{MC}| / \sigma_\mathrm{MC}$ — agreement within 2–3 sigma is expected.

---

## Section 7 — Diagnostics (Cells 16–17)

Four plots:

1. **$R(\lambda, \lambda')$ heatmap** (zoomed to $[1, 100]$): Shows the retarded propagator. It is lower-triangular ($R = 0$ for $\lambda < \lambda'$) and decays away from the diagonal because $\theta > 0$ causes exponential damping.
2. **$C(\lambda, \lambda)$ diagonal**: The variance of the integrated fluctuation as a function of $\lambda$. Grows from zero (no sources yet) and saturates when $R$-damping balances source accumulation.
3. **$C(\lambda_1, \lambda_2)$ heatmap**: The full two-point function. Symmetric, concentrated near the diagonal (short correlation length from $e^{-c\Delta\lambda^2}$), with a fan shape reflecting the causal structure of $R$.
4. **MC histogram**: The PDF of $X$ across all pixels and seeds. Should be approximately Gaussian (since the source is Gaussian and the nonlinearity is weak at $A = 10^{-15}$), with small skewness from the cubic vertex.

---

## Section 8 — Summary (Cells 18–19)

Prints a final table with pass/fail status (OK if within 3 sigma, CHECK otherwise), and the order-by-order breakdown showing how loop corrections modify each moment relative to tree level.

---

## Why this notebook matters

This is a **foundational validation** of the sft-wick perturbative machinery against a known-good nonlinear solver. If the moments agree:

- The Feynman diagram enumeration in sft-wick is correct.
- The propagator ($R$, $C$) computation and injection are correct.
- The QMC integration of diagrams is converging to the right values.
- The sachsfield fluctuation solver is correctly capturing the nonlinear dynamics.

The toy model ($A = 10^{-15}$, constant $\bar{s}$, simple $C_\ell$) is deliberately chosen so that perturbation theory converges rapidly — loop corrections are tiny compared to tree level for even moments. This makes disagreement easy to diagnose: if tree-level already disagrees with MC, the bug is in the propagators; if tree matches but loops don't, the bug is in the vertex/diagram logic.
