"""Pure-E consistency at the level of the canoes HIGH-branch spin kernel.

Collapsed FK geometry: leg 0 (T) and leg 1 (P) coincident at the origin,
leg 2 (P or P*) offset by gamma along +x.  canoes' orientation-averaged
flat-sky kernel gives, for a Fourier pair l2 = u e^{i a}, l3 = v e^{i(a+phi)}:
   (0,0,0)  TTT : 2 pi J_0(v g)
   (0,2,-2) Bmod: 2 pi J_0(v g) cos 2phi      [post-fix a95c0d6]
   (0,2,-2) Bmod: 2 pi J_0(v g)               [deployed table: phase dropped]
   (0,2,2)  TPP : 2 pi J_4(v g) cos 2phi
So with C(v) := 2 pi v int u du dphi B(u,v,w) cos 2phi,
   Bmod_fixed(g) = int dv C(v) J_0(v g),   TPP(g) = int dv C(v) J_4(v g)
i.e. the d^l_{22} / d^l_{2,-2} transforms of ONE spectrum -> BB == 0.
The deployed Bmod is the J_0 transform of C'(v) (no cos 2phi) -> BB != 0.
This script checks the kernel forms numerically and evaluates the three
channels for a toy bispectrum, then inverts to compare the spectra.
"""
import sys, numpy as np
from scipy.special import jv
sys.path.insert(0, "/Users/zzhang/projects/angular_statistics/canoes/src")
from canoes.sachs.kappa3 import _kappa3_limber_alpha_kernel, _kappa3_flat_triangle_vertices

g = 3.0 / 60 * np.pi / 180          # 3 arcmin
r2, r3 = _kappa3_flat_triangle_vertices(0.0, g, g)   # legs 0,1 coincident; leg 2 offset
print("triangle vertices r2, r3 =", r2, r3)
u = np.array([300.0]); v = np.array([700.0]); phi = np.linspace(0, 2*np.pi, 9)[:-1]
U, V, PH = np.meshgrid(u, v, phi, indexing="ij")
W = np.sqrt(U**2 + V**2 + 2*U*V*np.cos(PH))
for spins, ref, name in (((0,0,0), 2*np.pi*jv(0, V*g)*np.ones_like(PH), "TTT  = 2pi J0"),
                         ((0,2,-2), 2*np.pi*jv(0, V*g)*np.cos(2*PH), "Bmod = 2pi J0 cos2phi (fixed)"),
                         ((0,2,2), 2*np.pi*jv(4, V*g)*np.cos(2*PH), "TPP  = 2pi J4 cos2phi")):
    k = _kappa3_limber_alpha_kernel(spins=spins, u=U, v=V, phi=PH, w=W, r2=r2, r3=r3)
    print(f"{name:32s} max|kernel-ref| = {np.max(np.abs(k-ref)):.2e}  (max|ref| {np.max(np.abs(ref)):.2e})")

# --- toy bispectrum channels and the inversion test -------------------------
def F2(ka, kb, c):
    return 5/7 + 0.5*c*(ka/kb + kb/ka) + 2/7*c*c
def P(k):      # toy power law with a turnover, chi=1 units, k=ell
    return k / (1 + (k/200.0)**3.6)
def Btree(l1, l2, l3):
    l1 = np.maximum(l1, 1e-6)      # closed-triangle corner l1 -> 0: B -> 0 there (P(0)=0), avoid 0/0
    c12 = (l3**2 - l1**2 - l2**2) / (2*l1*l2); c23 = (l1**2 - l2**2 - l3**2)/(2*l2*l3); c31 = (l2**2 - l3**2 - l1**2)/(2*l3*l1)
    c12, c23, c31 = (np.clip(c, -1, 1) for c in (c12, c23, c31))
    out = 2*(F2(l1,l2,c12)*P(l1)*P(l2) + F2(l2,l3,c23)*P(l2)*P(l3) + F2(l3,l1,c31)*P(l3)*P(l1))
    return np.nan_to_num(out)

nu, nv, nphi = 60, 150, 48
uu = np.linspace(1.0, 1500.0, nu); vv = np.linspace(1.0, 1500.0, nv); pp = np.linspace(0, 2*np.pi, nphi, endpoint=False)
U, V, PH = np.meshgrid(uu, vv, pp, indexing="ij"); W = np.sqrt(U**2 + V**2 + 2*U*V*np.cos(PH))
B = Btree(W, U, V)                     # l1 = -l2-l3 has modulus W
du, dv, dp = uu[1]-uu[0], vv[1]-vv[0], pp[1]-pp[0]
# spectra: C(v) with cos2phi (E-consistent), C'(v) without (deployed Bmod)
C_cos = 2*np.pi * vv * np.sum(B * U * np.cos(2*PH), axis=(0, 2)) * du * dp
C_one = 2*np.pi * vv * np.sum(B * U, axis=(0, 2)) * du * dp
gam = np.geomspace(0.5, 300, 16) / 60 * np.pi / 180
def direct(spins):
    out = np.empty(gam.size)
    for i, gg in enumerate(gam):
        r2, r3 = _kappa3_flat_triangle_vertices(0.0, gg, gg)
        k = _kappa3_limber_alpha_kernel(spins=spins, u=U, v=V, phi=PH, w=W, r2=r2, r3=r3)
        out[i] = np.sum(B * U * V * k) * du * dv * dp
    return out
TTT, Bmod, TPP = direct((0,0,0)), direct((0,2,-2)), direct((0,2,2))
pred_Bmod = np.array([np.sum(C_cos * jv(0, vv*gg)) * dv for gg in gam])
pred_TPP  = np.array([np.sum(C_cos * jv(4, vv*gg)) * dv for gg in gam])
pred_TTT  = np.array([np.sum(C_one * jv(0, vv*gg)) * dv for gg in gam])
print("\nchannel vs. single-spectrum prediction (max rel. err over gamma):")
print("  Bmod(fixed) vs J0[C_cos]  :", np.max(np.abs(Bmod - pred_Bmod) / np.max(np.abs(Bmod))))
print("  TPP         vs J4[C_cos]  :", np.max(np.abs(TPP - pred_TPP) / np.max(np.abs(TPP))))
print("  TTT(=deployed Bmod) vs J0[C_one]:", np.max(np.abs(TTT - pred_TTT) / np.max(np.abs(TTT))))
print("\n  -> fixed Bmod and TPP are J0/J4 transforms of the SAME C(v): BB = 0 identically.")
print("     deployed Bmod (=TTT) is the J0 transform of a DIFFERENT C'(v): BB = (C' - C)/2 != 0.")
print("\n  C_cos/C_one at v = 100, 300, 700, 1200:",
      [f"{C_cos[np.argmin(np.abs(vv-x))]/C_one[np.argmin(np.abs(vv-x))]:+.3f}" for x in (100, 300, 700, 1200)])
print("  implied BB/EE = (C_one - C_cos)/(C_one + C_cos):",
      [f"{(C_one[i]-C_cos[i])/(C_one[i]+C_cos[i]):+.3f}" for i in [np.argmin(np.abs(vv-x)) for x in (100,300,700,1200)]])
print("\n  gamma[']   TTT        Bmod_fixed   TPP     Bmod_fixed/TTT")
for i in range(0, gam.size, 3):
    print(f"  {gam[i]*180/np.pi*60:8.2f} {TTT[i]:+.3e} {Bmod[i]:+.3e} {TPP[i]:+.3e} {Bmod[i]/TTT[i]:+.3f}")
