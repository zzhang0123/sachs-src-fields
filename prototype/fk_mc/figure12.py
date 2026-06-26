"""MC version of Figure 12 (draft label `fig: NLO 2pt FFFK cl`): the four
weak-lensing angular power spectra

    C_l^{kappa kappa},  C_l^{EE}+C_l^{BB},  C_l^{EE}-C_l^{BB},  C_l^{kappa E},

each split into Order-0 / O0+FF / O0+FF+FK, built from the Monte-Carlo (3,3)
two-point matrix xi_ab(gamma) of (kappa, gamma_+, gamma_x).

The 2PCF -> C_l transform is the paper's CURVED-SKY (reduced Wigner-d) operator,
ported VERBATIM from analyses/analysis3/plot_analysis3_cl_decomposition.py so the
conventions (kernels d^l_{00}/d^l_{22}/d^l_{2,-2}/d^l_{20}, the 2pi sin(theta)
measure, the DC-subtraction of the FF large-angle monopole) match exactly.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.special import eval_legendre

OUT = "/Users/zzhang/projects/SFT/src-field/prototype/fk_mc/outputs"

# observable -> (title, combos on the (3,3) xi_ab, Wigner (m,n)); from analysis3.
OBSERVABLE_SPECS = (
    ("kappa-kappa", r"$C_\ell^{\kappa\kappa}$", [((0, 0), +1.0)], (0, 0)),
    ("xi_plus", r"$C_\ell^{EE}+C_\ell^{BB}$", [((1, 1), +1.0), ((2, 2), +1.0)], (2, 2)),
    ("xi_minus", r"$C_\ell^{EE}-C_\ell^{BB}$", [((1, 1), +1.0), ((2, 2), -1.0)], (2, -2)),
    ("kappa_gamma_t", r"$C_\ell^{\kappa E}$", [((0, 1), -1.0)], (2, 0)),
)
ELL = np.array(sorted({int(round(v)) for v in np.geomspace(3.0, 1500.0, 30)}), float)


# --- curved-sky transform, ported verbatim from analysis3 -------------------
def wigner_d(ell_int, cos_theta, m, n):
    x = np.asarray(cos_theta, float)
    want = sorted({int(round(l)) for l in np.atleast_1d(ell_int)})
    out: dict[int, np.ndarray] = {}
    if (m, n) == (0, 0):
        return {L: eval_legendre(L, x) for L in want}
    s2sq = np.clip((1.0 - x) / 2.0, 0.0, None)
    c2sq = np.clip((1.0 + x) / 2.0, 0.0, None)
    if (m, n) == (2, 2):
        seed = c2sq * c2sq
    elif (m, n) == (2, -2):
        seed = s2sq * s2sq
    elif (m, n) == (2, 0):
        seed = math.sqrt(3.0 / 8.0) * (1.0 - x * x)
    else:
        raise ValueError(f"unsupported Wigner indices ({m},{n})")
    lmin, lmax = max(abs(m), abs(n)), max(want)
    dlm1 = np.zeros_like(x)
    dl = np.array(seed, dtype=float)
    if lmin in want:
        out[lmin] = dl.copy()
    for L in range(lmin, lmax):
        a1 = (2 * L + 1) * (L * (L + 1) * x - m * n)
        t = (L * L - m * m) * (L * L - n * n)
        a2 = (L + 1) * math.sqrt(t) if t > 0 else 0.0
        denom = L * math.sqrt(((L + 1) ** 2 - m * m) * ((L + 1) ** 2 - n * n))
        dlp1 = (a1 * dl - a2 * dlm1) / denom
        dlm1, dl = dl, dlp1
        if (L + 1) in want:
            out[L + 1] = dl.copy()
    return out


def selftest_wigner(tol=1e-9):
    nodes, w = np.polynomial.legendre.leggauss(210)
    for (m, n) in ((2, 2), (2, -2), (2, 0)):
        for L, arr in wigner_d([5, 17, 60, 200], nodes, m, n).items():
            assert abs(float(w @ (arr * arr)) / (2.0 / (2 * L + 1)) - 1.0) < tol
    return True


def build_curved_matrix(gamma_arcmin, ell_int, m, n, n_fine=20000):
    theta = gamma_arcmin * (math.pi / 180.0 / 60.0)
    lt = np.log(theta)
    lt_f = np.linspace(lt.min(), lt.max(), n_fine)
    theta_f = np.exp(lt_f)
    sth = np.sin(theta_f)
    dln = np.gradient(lt_f)
    dblock = wigner_d(ell_int, np.cos(theta_f), m, n)
    meas = 2.0 * np.pi * sth * theta_f * dln
    D = np.array([meas * dblock[int(round(L))] for L in ell_int])
    return lt, lt_f, D


def forward_curved(xi_theta, setup, dc_subtract=True):
    lt, lt_f, D = setup
    xi = np.asarray(xi_theta, float)
    if dc_subtract:
        xi = xi - xi[-1]
    return D @ PchipInterpolator(lt, xi)(lt_f)


def combine(xi33, combos):
    """Observable 2PCF from the (ng,3,3) xi_ab via the component combos."""
    out = None
    for (a, b), s in combos:
        term = s * xi33[:, a, b]
        out = term if out is None else out + term
    return out


def smooth_33(gamma, x33, se33=None, s_factor=1.0, gamma_cap=None):
    """Smooth each (a,b) component of a noisy MC (ng,3,3) channel on log-gamma.

    The physical 2PCF is smooth; the MC adds per-point noise that the high-ell
    curved-sky transform amplifies.  We fit a weighted smoothing spline (weights
    1/SE) per component and evaluate it back on the gamma grid, smoothing each
    xi_ab BEFORE combining -- so the modulus pair (11+22) keeps its signal and
    the cancellation channels (11-22, 01) keep their ~0 (smoothing 11 and 22
    equally leaves their difference ~0).
    """
    from scipy.interpolate import UnivariateSpline
    lg = np.log(np.asarray(gamma, float))
    out = np.array(x33, float)
    # robust bad-gamma mask: the VR estimator is numerically unstable at a few
    # large separations (near-singular node covariance -> Levy-Q blow-up, e.g.
    # gamma~647'). Flag points whose modulus xi_+ = x_11 + x_22 is a strong MAD
    # outlier (or whose SE blew up) and drop them from EVERY component's fit, so
    # one bad gamma cannot leak into the (small-difference) EE-BB / kE channels.
    xip = x33[:, 1, 1] + x33[:, 2, 2]
    med = np.median(xip)
    mad = np.median(np.abs(xip - med)) + 1e-300
    bad = np.abs(xip - med) > 8.0 * mad
    if se33 is not None:
        se_kk = se33[:, 0, 0]
        bad = bad | (se_kk > 8.0 * np.median(se_kk))
    if gamma_cap is not None:   # VR unreliable beyond this separation (blow-ups)
        bad = bad | (np.asarray(gamma, float) > gamma_cap)
    for a in range(3):
        for b in range(3):
            y = x33[:, a, b]
            if se33 is not None:
                w = 1.0 / np.clip(se33[:, a, b], np.nanmax(se33[:, a, b]) * 1e-3, None)
            else:
                w = np.ones_like(y)
            w = np.where(bad, w * 1e-6, w)   # exclude the unstable gammas
            sp = UnivariateSpline(lg, y, w=w, s=s_factor * len(lg))
            out[:, a, b] = sp(lg)
    return out


# --- signed-marker plotting (filled positive, hollow negative) ---------------
def _signed_markers(ax, ell, y, color, marker, label, line=False, **kw):
    y = np.asarray(y)
    ax.plot(ell, np.abs(y), color=color, lw=1.4 if line else 0,
            ls="-" if line else "none", zorder=kw.get("zorder", 2), label=label)
    pos, neg = y > 0, y <= 0
    ax.plot(ell[pos], np.abs(y[pos]), marker, color=color, mfc=color, ms=5,
            ls="none", zorder=kw.get("zorder", 2) + 0.1)
    ax.plot(ell[neg], np.abs(y[neg]), marker, color=color, mfc="none", ms=5,
            ls="none", zorder=kw.get("zorder", 2) + 0.1)


def make_figure(path=f"{OUT}/fk_channels.npz", save=f"{OUT}/figure12_mc.pdf", smooth=True):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assert selftest_wigner()
    d = np.load(path)
    g, o0, ff, fk = d["gamma"], d["o0"], d["ff"], d["fk"]
    if smooth:   # O0 is analytic (smooth); smooth the noisy MC channels FF, FK
        ff = smooth_33(g, ff, d["ff_se"])
        # FK: reject only the VR blow-up gammas (MAD outliers, e.g. ~461'/647');
        # the EE-BB residual that survives is the small-difference channel at the
        # MC floor (its xi_- ~ 0 is confirmed at small gamma; see fk_highstat).
        fk = smooth_33(g, fk, d["fk_se"])
    pref = ELL * (ELL + 1.0) / (2.0 * np.pi)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    print(f"MC Figure 12 (n_real={int(d['n_real'])}, {len(g)} gammas to {g[-1]:.0f}'):")
    for ax, (name, title, combos, mn) in zip(axes.ravel(), OBSERVABLE_SPECS):
        m, n = mn
        o0o, ffo, fko = (combine(c, combos) for c in (o0, ff, fk))
        setup = build_curved_matrix(g, ELL, m, n)
        Cl_o0 = forward_curved(o0o, setup)
        Cl_o0ff = forward_curved(o0o + ffo, setup)
        Cl_full = forward_curved(o0o + ffo + fko, setup)
        _signed_markers(ax, ELL, pref * Cl_full, "0.25", "s", "Full (O0+FF+FK)", line=True, zorder=4)
        _signed_markers(ax, ELL, pref * Cl_o0ff, "C1", "^", "O0+FF", zorder=3)
        _signed_markers(ax, ELL, pref * Cl_o0, "C0", "o", "Order-0", zorder=2)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(ELL.min(), ELL.max()); ax.set_ylim(1e-7, 1e-3)
        ax.set_title(title); ax.set_xlabel(r"$\ell$")
        ax.set_ylabel(r"$|\ell(\ell+1)C_\ell/2\pi|$")
        ng_frac = np.nanmax(np.abs((Cl_full - Cl_o0ff) / Cl_o0ff))
        print(f"  {name:14s} (d^l_{{{m},{n}}}): max |FK/(O0+FF)| over ell = {ng_frac:.2f}")
    axes[0, 0].legend(fontsize=9, loc="lower left")
    fig.suptitle("MC angular power spectra: O0 / O0+FF / O0+FF+FK  "
                 "(filled=+, hollow=-)", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(save)
    plt.close(fig)
    print(f"-> {save}")


if __name__ == "__main__":
    make_figure()
