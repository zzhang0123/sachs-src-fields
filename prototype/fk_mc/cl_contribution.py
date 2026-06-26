"""Non-Gaussian contribution to the convergence angular power spectrum.

The convergence kappa is spin-0, so its angular power spectrum is the order-0
Hankel (flat-sky / small-angle) transform of the 2-point function,

    C_ell = 2 pi int_0^inf xi_kappa(theta) J_0(ell theta) theta dtheta,

valid for the MC's small-angle range (gamma <~ 32 deg) at ell >~ 1/theta_max.
The NON-GAUSSIAN contribution is C_ell^FK (the FK channel); O0 is the Gaussian
Order-0, FF the Gaussian nonlinear-propagation channel.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import j0

OUT = "/Users/zzhang/projects/SFT/src-field/prototype/fk_mc/outputs"


def hankel_cl(gamma_arcmin, xi, ells, n_theta=4000):
    """C_ell = 2 pi int xi(theta) J0(ell theta) theta dtheta over the measured range."""
    theta = np.deg2rad(np.asarray(gamma_arcmin) / 60.0)        # radians
    lt = np.log(theta)
    xi_of = interp1d(lt, np.asarray(xi), kind="linear", fill_value="extrapolate")
    tg = np.geomspace(theta.min(), theta.max(), n_theta)
    xg = xi_of(np.log(tg))
    return np.array([2.0 * np.pi * np.trapezoid(xg * j0(l * tg) * tg, tg) for l in ells])


def load(path=f"{OUT}/fk_channels.npz"):
    d = np.load(path)
    return {k: d[k] for k in d.files}


def make_figure(path=f"{OUT}/fk_channels.npz", save=f"{OUT}/fk_cl.pdf"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = load(path)
    g, o0, ff, fk, fk_se = d["gamma"], d["o0"], d["ff"], d["fk"], d["fk_se"]
    ells = np.geomspace(20, 3000, 60)
    cl_o0 = hankel_cl(g, o0, ells)
    cl_ff = hankel_cl(g, ff, ells)
    cl_fk = hankel_cl(g, fk, ells)
    norm = ells * (ells + 1) / (2 * np.pi)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    # left: xi_kappa channels (abs, log-log), FK dominance at large gamma
    ax1.loglog(g, np.abs(o0), "C0o-", ms=3, label=r"O0 (Gaussian)")
    ax1.loglog(g, np.abs(ff), "C2s-", ms=3, label=r"FF (Gaussian, nonlinear)")
    ax1.errorbar(g, np.abs(fk), yerr=fk_se, fmt="C3^-", ms=3, lw=1,
                 label=r"FK (non-Gaussian, VR)")
    ax1.axvline(200, color="k", ls=":", lw=0.8)
    ax1.text(210, ax1.get_ylim()[1] * 0.3, "FK dominates >200'", fontsize=8)
    ax1.set_xlabel(r"$\gamma$ [arcmin]")
    ax1.set_ylabel(r"$|\xi_\kappa(\gamma)|$  (channels)")
    ax1.set_title(r"convergence 2-point: channels")
    ax1.legend(fontsize=8)

    # right: C_ell contributions
    ax2.loglog(ells, np.abs(norm * cl_o0), "C0-", lw=2, label="O0")
    ax2.loglog(ells, np.abs(norm * cl_ff), "C2-", lw=2, label="FF")
    ax2.loglog(ells, np.abs(norm * cl_fk), "C3-", lw=2,
               label=r"$\Delta C_\ell^{\rm NG}$ = FK")
    ax2.set_xlabel(r"$\ell$")
    ax2.set_ylabel(r"$|\ell(\ell+1)C_\ell^{\kappa}/2\pi|$")
    ax2.set_title(r"non-Gaussian contribution to $C_\ell^\kappa$")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save)
    plt.close(fig)
    # quick text summary of the NG fraction in C_ell
    frac = np.abs(cl_fk) / (np.abs(cl_o0) + 1e-300)
    print(f"figure -> {save}")
    print(f"  C_ell^FK / C_ell^O0:  min={frac.min():.2f} max={frac.max():.2f} "
          f"(ell in [{ells[0]:.0f}, {ells[-1]:.0f}])")
    return ells, cl_o0, cl_ff, cl_fk


if __name__ == "__main__":
    make_figure()
