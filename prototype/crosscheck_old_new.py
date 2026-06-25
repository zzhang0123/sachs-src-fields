"""
Cross-validate the existing sachsfield (x1 = 2*theta convention) against the
new sachsray physical (theta) convention, and quantify the field-1 source
normalisation.

Old coefficients.quadratic encodes F_111 = -1/2 etc., i.e. it integrates
x1 = 2*theta. With a field-1 source s1, the old expansion obeys

    d(x1)/dlam = -1/2 x1^2 + s1   ==>   dtheta/dlam = -theta^2 + s1/2   (x1=2theta)

so to reproduce the draft's physical  dtheta/dlam = -theta^2 + Phi00  the old
solver needs  s1 = 2*Phi00.  perFLRW injects s1 = Phi00 (cosmology.py:416),
i.e. HALF. This script demonstrates that numerically.
"""

import numpy as np
from scipy.integrate import solve_ivp

from sachsfield.coefficients import quadratic  # OLD, x1 = 2 theta


def old_rhs(lam, chi, s1):
    return quadratic(np.asarray(chi), 3) + np.array([s1, 0.0, 0.0])


def new_theta_rhs(lam, th, phi00):  # physical, shear-free
    return [-th[0] ** 2 + phi00]


def run(c, label):
    lam0, lam1, th0 = 0.5, 3.0, 0.8
    le = np.linspace(lam0, lam1, 60)
    old = solve_ivp(old_rhs, (lam0, lam1), [2 * th0, 0, 0], args=(c,),
                    t_eval=le, rtol=1e-11, atol=1e-13)
    theta_old = old.y[0] / 2.0  # x1 -> theta
    new_c = solve_ivp(new_theta_rhs, (lam0, lam1), [th0], args=(c,),
                      t_eval=le, rtol=1e-11, atol=1e-13).y[0]
    new_half = solve_ivp(new_theta_rhs, (lam0, lam1), [th0], args=(c / 2.0,),
                         t_eval=le, rtol=1e-11, atol=1e-13).y[0]
    e_c = np.max(np.abs(theta_old - new_c))
    e_h = np.max(np.abs(theta_old - new_half))
    print(f"  {label:22s}: |theta_old - new(Phi00=c)|={e_c:.2e}   "
          f"|theta_old - new(Phi00=c/2)|={e_h:.2e}")
    return e_c, e_h


def main():
    print("Old (x1=2theta, source s1) vs new physical theta, constant source s1=c:")
    run(0.0, "vacuum c=0")
    run(0.3, "c=0.3")
    run(0.8, "c=0.8")
    print("\n  Interpretation: theta_old tracks new with Phi00 = c/2, NOT c.")
    print("  => with perFLRW injecting s1=Phi00, the old background focusing is")
    print("     HALF the physical value. The fix is s1 = 2*Phi00 (or use sachsray,")
    print("     which uses theta directly with F_111=-1 and s1=Phi00).")


if __name__ == "__main__":
    main()
