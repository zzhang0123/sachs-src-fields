"""Post-Born (lens-lens coupling) rotation power spectrum for the Takahashi
2017 cosmology and source planes, flat-sky Limber, Gaussian planes, CAMB
halofit P(k). This is the Krause & Hirata 2010 / Cooray & Hu 2002 '22' term:

    C_ell^{omega omega} = int_0^{chi_s} dchi_2 int_0^{chi_2} dchi_1
        int d^2l_1/(2 pi)^2 sin^2(2 phi_12) P_kappa(l_1; chi_2 -> chi_s) P_kappa(l_2; chi_1 -> chi_2)

with l_2 = l - l_1, phi_12 the angle between l_1 and l_2, and
P_kappa(l; chi -> chi') = W(chi; chi')^2 / chi^2 P_delta(l / chi, z(chi)),
W(chi; chi') = (3/2) Omega_m (H0/c)^2 chi (chi' - chi) / (chi' a(chi)).
Derivation: A^(2) contains U_{2s} U_{12}; omega = antisym part; U(l) = 2 l_a l_b kappa(l)/l^2;
omega(l) = int d^2l_1/(2pi)^2 sin(2 phi_12) kappa_2(l_2) kappa_1(l_1).
On the sphere the shear B-mode of the same curl potential is C^BB = F_ell C^{omega omega},
F_ell = (l+2)(l-1)/(l(l+1)) (flat sky: C^BB = C^{omega omega}, Krause & Hirata 2010).
Also returns the first-order C_ell^{kappa kappa} (Limber, nu = l + 1/2) as a normalisation check.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
# Takahashi et al. 2017, Section 2: Omega_cdm 0.233, Omega_b 0.046, Omega_m 0.279, Omega_L 0.721, h 0.7, sigma_8 0.82, n_s 0.97
COSMO = dict(h=0.7, Om=0.279, Ob=0.046, s8=0.82, ns=0.97)
ZS = {"zs10": 0.574, "zs16": 1.033, "zs38": 5.342}
C_KMS = 299792.458


def camb_setup():
    import camb
    h = COSMO["h"]
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=100 * h, ombh2=COSMO["Ob"] * h * h, omch2=(COSMO["Om"] - COSMO["Ob"]) * h * h, mnu=0.0, omk=0.0, tau=0.06)
    pars.InitPower.set_params(ns=COSMO["ns"], As=2.0e-9)
    pars.set_matter_power(redshifts=[0.0], kmax=100.0)
    pars.NonLinear = camb.model.NonLinear_none
    res = camb.get_results(pars)
    s8 = float(res.get_sigma8_0())
    pars.InitPower.set_params(ns=COSMO["ns"], As=2.0e-9 * (COSMO["s8"] / s8) ** 2)
    pars.NonLinear = camb.model.NonLinear_both
    pars.NonLinearModel.set_params(halofit_version="takahashi")
    res = camb.get_results(pars)
    pk = camb.get_matter_power_interpolator(pars, nonlinear=True, hubble_units=False, k_hunit=False, kmax=300.0, zmax=6.0, var1="delta_tot", var2="delta_tot")
    return res, pk, float(res.get_sigma8_0())


def lens_weight(chi, chi_s, a):
    h0c = 100.0 * COSMO["h"] / C_KMS  # 1/Mpc
    return 1.5 * COSMO["Om"] * h0c ** 2 * chi * (chi_s - chi) / (chi_s * a)


def p_kappa_table(pk, chi, z, chi_s, lgrid):
    """P_kappa(l; chi -> chi_s) per unit chi on a grid of chi (rows) and l (cols)."""
    a = 1.0 / (1.0 + z)
    w = lens_weight(chi, chi_s, a)
    out = np.zeros((len(chi), len(lgrid)))
    for i in range(len(chi)):
        k = (lgrid + 0.5) / chi[i]
        out[i] = w[i] ** 2 / chi[i] ** 2 * pk.P(z[i], k, grid=False)
    return out


def compute(plane, res, pk, ells, n_chi=48, n_l1=96, n_phi=48):
    z_s = ZS[plane]
    chi_s = float(res.comoving_radial_distance(z_s))
    # chi_2 grid (outer), chi_1 grid nested (inner, uniform on (0, chi_2))
    x2 = (np.arange(n_chi) + 0.5) / n_chi
    chi2 = chi_s * x2
    z2 = res.redshift_at_comoving_radial_distance(chi2)
    lgrid = np.geomspace(1.0, 60000.0, 260)
    T_s = p_kappa_table(pk, chi2, z2, chi_s, lgrid)  # [chi2, l]
    ckk = np.array([np.sum(T_s[:, 0] * 0) for _ in ells])
    # first-order C_kk on the same chi2 grid
    ckk = np.zeros(len(ells))
    for j, l in enumerate(ells):
        vals = np.array([lens_weight(chi2[i], chi_s, 1 / (1 + z2[i])) ** 2 / chi2[i] ** 2 * pk.P(z2[i], (l + 0.5) / chi2[i], grid=False) for i in range(n_chi)])
        ckk[j] = np.sum(vals) * chi_s / n_chi
    # inner tables T_2[chi2 index][chi1, l] with chi1 on (0, chi2)
    n1 = 24
    T_2 = []
    chi1_all = []
    for i in range(n_chi):
        x1 = (np.arange(n1) + 0.5) / n1
        chi1 = chi2[i] * x1
        z1 = res.redshift_at_comoving_radial_distance(chi1)
        T_2.append(p_kappa_table(pk, chi1, z1, chi2[i], lgrid))
        chi1_all.append(chi1)
    l1 = np.geomspace(2.0, 40000.0, n_l1)
    dlnl1 = np.log(l1[1] / l1[0])
    phi1 = (np.arange(n_phi) + 0.5) * 2 * np.pi / n_phi
    dphi = 2 * np.pi / n_phi
    L1x = l1[:, None] * np.cos(phi1)[None, :]
    L1y = l1[:, None] * np.sin(phi1)[None, :]
    cww = np.zeros(len(ells))
    loglgrid = np.log(lgrid)
    for j, l in enumerate(ells):
        L2x = l - L1x
        L2y = -L1y
        l2 = np.hypot(L2x, L2y)
        dot = L1x * L2x + L1y * L2y
        cross = L1x * L2y - L1y * L2x
        s2 = (2.0 * dot * cross / (l1[:, None] ** 2 * l2 ** 2)) ** 2  # sin^2(2 phi_12)
        meas = s2 * (l1[:, None] ** 2) * dlnl1 * dphi / (2 * np.pi) ** 2  # l1 dl1 dphi
        logl2 = np.log(np.clip(l2, lgrid[0], lgrid[-1]))
        total = 0.0
        for i in range(n_chi):
            ts = np.interp(np.log(l1), loglgrid, T_s[i])  # [l1]
            inner = 0.0
            for m in range(n1):
                t2 = np.interp(logl2, loglgrid, T_2[i][m])  # [l1, phi]
                inner += np.sum(meas * ts[:, None] * t2)
            total += inner * (chi2[i] / n1)
        cww[j] = total * chi_s / n_chi
    return dict(ell=ells.tolist(), chi_s_Mpc=chi_s, C_kk_limber=ckk.tolist(), C_ww_postborn=cww.tolist())


def main():
    t0 = time.time()
    res, pk, s8 = camb_setup()
    print(f"CAMB ready ({time.time() - t0:.0f}s); sigma8 = {s8:.4f}")
    ells = np.array([20, 30, 50, 60, 80, 100, 150, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000, 4000], dtype=float)
    out = dict(cosmology=COSMO, sigma8_check=s8, note=__doc__)
    for plane in ("zs16", "zs38", "zs10"):
        t0 = time.time()
        out[plane] = compute(plane, res, pk, ells)
        print(plane, f"{time.time() - t0:.0f}s", json.dumps({k: (v if not isinstance(v, list) else [f"{x:.3e}" for x in v]) for k, v in out[plane].items()}))
    with open(os.path.join(OUT, "post_born_theory.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
