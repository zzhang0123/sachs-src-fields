"""
Definitive numerical check: what field-1 source s1 does the OLD
sachsfield.SaddlePointSolver need to reproduce the physical angular-diameter
distance D = a*chi, and what does perFLRW actually feed it?

Ground truth (matter-dominated toy, a = tau^2, observer at tau=1):
    lam_phys = (1 - tau^5)/5 >= 0 ,   D_true = tau^2 (1 - tau) ,
    Phi00_phys = -6/tau^10  (< 0, focusing).

The old solver integrates chi1 = 2*theta in NEGATIVE lambda. We map
lam_code = -lam_phys, sweep candidate sources s1 = k * Phi00_phys, reconstruct
the saddle amplitude u = exp(0.5 * int chi1 dlam) (from chi1 = 2 u'/u), and see
which k reproduces D_true. Prediction (u'' = 0.5 s1 u): k = +2.
perFLRW feeds s1 = +|Phi00| = -Phi00_phys, i.e. k = -1.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

import sachsfield as sf


def toy(lam_phys):
    tau = (1.0 - 5.0 * lam_phys) ** 0.2
    return tau, tau**2 * (1.0 - tau), -6.0 / tau**10  # tau, D_true, Phi00_phys


def reconstruct_D(saddle, lam_code_grid):
    chi1 = np.array([saddle(lc)[0] for lc in lam_code_grid])  # = 2*theta
    # u = exp(0.5 * int chi1 dlam_code); |u| ∝ D
    integ = cumulative_trapezoid(chi1, lam_code_grid, initial=0.0)
    u = np.exp(0.5 * integ)
    return np.abs(u)


def run(k, label):
    eps, lam_max = 1e-3, 0.15
    # saddle solves in negative lambda: lam_code in (-eps, -lam_max)
    def sbar(lam_code):
        lam_phys = -lam_code
        _, _, phi = toy(lam_phys)
        return np.array([k * phi, 0.0, 0.0])

    sp = sf.SaddlePointSolver(
        sbar, (-eps, -lam_max), n_fields=3, y1_init=2.0,
        rtol=1e-10, atol=1e-12,
    )
    res = sp.solve()

    lam_code = -np.linspace(eps, lam_max, 300)
    D_rec = reconstruct_D(res, lam_code)
    tau, D_true, _ = toy(-lam_code)
    # normalise both to their value at the first point (shape comparison)
    D_rec_n = D_rec / D_rec[5]
    D_true_n = D_true / D_true[5]
    rel = np.max(np.abs(D_rec_n - D_true_n)) / np.max(np.abs(D_true_n))
    print(f"  s1 = {k:+g}*Phi00  ({label:18s}): shape rel.err vs D=a*chi = {rel:.3e}"
          f"   {'<<< MATCH' if rel < 1e-2 else ''}")
    return rel


def main():
    print("Old SaddlePointSolver: which s1 reproduces D = a*chi?")
    run(+2, "needed (2*Phi00)")
    run(-1, "perFLRW (+|Phi00|)")
    run(+1, "Phi00")
    run(-2, "-2*Phi00")
    print("\n  => saddle needs s1 = +2*Phi00 (Phi00<0). perFLRW feeds -1*Phi00 = +|Phi00|:")
    print("     wrong sign AND factor 2 (off by -2x). Fix: background_phi00 -> -2x for the")
    print("     OLD x1=2theta saddle, or use sachsray (theta convention, s1=Phi00).")


if __name__ == "__main__":
    main()
