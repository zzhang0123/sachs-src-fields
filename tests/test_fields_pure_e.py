"""Spin-2 convention gate for sachsray: a scalar-sourced driving field must give
a pure-E linear shear that reproduces the convergence.

This is the gate the 2026-09-03 B-mode audit pinned (prototype/bmode_audit/A1,
n1_sachsray). It guards the sign rule in ``driving_from_potential_alms`` and the
screen-basis convention of the solver: with the same sign on the spin-0 and the
spin-2 multipliers, ``kappa_E = +gamma_E`` and C_BB/C_EE is at the pixel-leakage
floor; with the sign of W2 flipped, B/E ~ 1.
"""
from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

hp = pytest.importorskip("healpy")

from sachsray import fields, raytrace  # noqa: E402

NSIDE, LMAX_IN, LMAX_OUT, N_LAM = 16, 24, 32, 12


def _potential_alms(seed: int, n_lam: int, lmax: int):
    """Independent Gaussian shells of a red potential, band-limited at lmax."""
    rng = np.random.default_rng(seed)
    ell = np.arange(lmax + 1, dtype=float)
    cl = np.zeros(lmax + 1)
    cl[2:] = 1.0 / (ell[2:] * (ell[2:] + 1.0)) ** 2
    alms = []
    for _ in range(n_lam):
        # healpy.synalm draws complex Gaussian alms with the requested C_l
        alms.append(hp.synalm(cl, lmax=lmax, new=True))
    return np.stack(alms)


def _spectra(gamma1, gamma2, kappa, lmax):
    alm_e, alm_b = hp.map2alm_spin([gamma1, gamma2], 2, lmax)
    alm_k = hp.map2alm(kappa, lmax=lmax, iter=3)
    cl_ee = hp.alm2cl(alm_e)
    cl_bb = hp.alm2cl(alm_b)
    cl_kk = hp.alm2cl(alm_k)
    cl_ke = hp.alm2cl(alm_k, alm_e)
    return cl_ee, cl_bb, cl_kk, cl_ke


def _trace(alms, scale):
    ts = np.linspace(0.0, 2.0, N_LAM)
    field = fields.driving_from_potential_alms(
        ts, alms, NSIDE, lmax=LMAX_IN, scale=scale, dtype=np.float64
    )
    obs = raytrace.trace_rays(field, 2.0, chunk=4096, rtol=1e-9, atol=1e-12)
    return {k: np.asarray(v) for k, v in obs.items()}


@pytest.fixture(scope="module")
def traced():
    alms = _potential_alms(seed=3, n_lam=N_LAM, lmax=LMAX_IN)
    # scale the potential so kappa_rms ~ 2e-3 (second-order B/E ~ kappa_rms^2)
    probe = _trace(alms, scale=1.0)
    scale = 2.0e-3 / float(np.std(probe["kappa"]))
    obs = _trace(alms, scale=scale)
    return obs, alms, scale


def test_linear_shear_is_pure_e(traced):
    obs, _, _ = traced
    cl_ee, cl_bb, _, _ = _spectra(obs["gamma1"], obs["gamma2"], obs["kappa"], LMAX_OUT)
    band = slice(2, LMAX_IN + 1)
    ratio = cl_bb[band].sum() / cl_ee[band].sum()
    # physical second-order B ~ kappa_rms^2 ~ 4e-6; pixel leakage far below
    assert ratio < 2.0e-5, f"B/E = {ratio:.2e}: driving field is not pure E"


def test_shear_e_mode_equals_convergence(traced):
    obs, _, _ = traced
    cl_ee, _, cl_kk, cl_ke = _spectra(obs["gamma1"], obs["gamma2"], obs["kappa"], LMAX_OUT)
    ell = np.arange(LMAX_OUT + 1, dtype=float)
    band = slice(2, LMAX_IN + 1)
    f_ell = np.ones_like(ell)
    f_ell[2:] = (ell[2:] + 2.0) * (ell[2:] - 1.0) / (ell[2:] * (ell[2:] + 1.0))
    ratio = cl_ee[band].sum() / (f_ell[band] * cl_kk[band]).sum()
    corr = cl_ke[band].sum() / np.sqrt(cl_ee[band].sum() * cl_kk[band].sum())
    assert abs(ratio - 1.0) < 0.05, f"C_EE / (F C_kk) = {ratio:.4f}"
    assert corr > 0.99, f"kappa-E correlation {corr:.4f} (sign rule violated if < 0)"


def test_wrong_sign_convention_is_detected(traced):
    obs, _, _ = traced
    cl_ee, cl_bb, _, _ = _spectra(obs["gamma1"], -obs["gamma2"], obs["kappa"], LMAX_OUT)
    band = slice(2, LMAX_IN + 1)
    ratio = cl_bb[band].sum() / cl_ee[band].sum()
    assert 0.3 < ratio < 3.0, f"flipped W2 should give B/E ~ 1, got {ratio:.3f}"
