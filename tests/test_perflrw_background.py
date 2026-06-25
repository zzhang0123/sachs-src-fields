"""
Settle the perFLRW background-Phi00 sign/normalisation against the STF_lensing
draft's first-principles derivation (cosmology.tex eq:driving bg).

Paper result (reproduced in the module docstring of the test below):

    Phi00^(0) = -(E/a)^2 (H^2 - H')          [conformal Hubble H = a'/a]
              = -4 pi G (rho + P) / a^2        [with E0 = 1, using H^2-H' = 4piG a^2 (rho+P)]

i.e. STRICTLY NEGATIVE for ordinary matter (focusing). perFLRW returns
+4 pi G (rho+P)(E/a)^2 -> a SIGN ERROR.

The pyccl-free tests use a matter-dominated toy (a ∝ tau^2) where the angular
diameter distance D = a*chi is closed-form, and verify:
  * the Friedmann identity H^2 - H' = 4 pi G a^2 (rho+P),
  * the sachsray Jacobi solver reproduces D = a*chi when fed Phi00 < 0,
  * feeding +|Phi00| (the perFLRW sign) does NOT reproduce D (sign matters).
A pyccl-gated test checks perFLRW.background_phi00 itself.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sachsray import solvers  # noqa: E402
import jax.numpy as jnp  # noqa: E402


# --------------------------------------------------------------------------
# Matter-dominated (EdS) toy, parametrised by conformal time tau, with the
# observer at tau = 1 (a = 1).  a = tau^2,  H = a'/a = 2/tau.
# Affine parameter (E0=1):  dtau/dlam = -1/a^2 = -1/tau^4  =>  lam = (1 - tau^5)/5.
# Comoving distance:        dchi/dlam = +1/a^2  =>  chi = 1 - tau.
# Jacobi amplitude:         D = a*chi = tau^2 (1 - tau).
# Background focusing:      Phi00 = -(H^2 - H')/a^4 = -6 / tau^10   (< 0).
# (Verified analytically: D'' (d/dlam) = Phi00 * D exactly.)
# --------------------------------------------------------------------------
def _tau_of_lam(lam: np.ndarray) -> np.ndarray:
    return (1.0 - 5.0 * lam) ** 0.2


def _toy(lam: np.ndarray):
    tau = _tau_of_lam(lam)
    D_true = tau**2 * (1.0 - tau)
    phi00 = -6.0 / tau**10
    return tau, D_true, phi00


def test_friedmann_identity_eds():
    """H^2 - H' = 4 pi G a^2 (rho+P) for matter domination (a ∝ tau^2)."""
    tau = np.linspace(0.4, 1.0, 50)
    H = 2.0 / tau            # conformal Hubble
    Hp = -2.0 / tau**2      # dH/dtau
    a = tau**2
    lhs = H**2 - Hp         # = 6/tau^2
    # EdS: H^2 = (8piG/3) a^2 rho  => 4piG a^2 rho = (3/2) H^2 ; P = 0
    rhs = 1.5 * H**2
    assert np.allclose(lhs, rhs, rtol=1e-12)
    assert np.allclose(lhs, 6.0 / tau**2, rtol=1e-12)


def test_jacobi_reproduces_angular_diameter_distance():
    """sachsray Jacobi with the PHYSICAL (negative) Phi00 reproduces D = a*chi.

    This is the ground-truth check against the draft's closed form
    (appendix.tex eq: D equals a chi), with a genuine cosmological a(tau).
    """
    lam_max = 0.19  # tau ~ 0.40 ; stay short of the a->0 singularity
    ts = jnp.linspace(0.0, lam_max, 800)
    _, _, phi00 = _toy(np.asarray(ts))
    le = jnp.linspace(1e-4, lam_max, 200)
    ys = solvers.solve_jacobi_scalar(ts, jnp.asarray(phi00), le, rtol=1e-11, atol=1e-13)
    D_ode = np.asarray(ys[:, 0])
    _, D_true, _ = _toy(np.asarray(le))
    rel = np.max(np.abs(D_ode - D_true)) / np.max(np.abs(D_true))
    assert rel < 1e-6, f"Jacobi did not reproduce a*chi: rel err {rel:.2e}"


def test_sign_matters():
    """The sign of Phi00 is physical: focusing (Phi00<0) gives D=sin(lam) which
    turns over (caustic); the perFLRW sign (Phi00>0) gives D=sinh(lam) which runs
    away exponentially -- an unphysical distance. (Const Phi00 = -+1.)"""
    ts = jnp.linspace(0.0, 3.0, 600)
    le = jnp.linspace(1e-3, 3.0, 200)
    lam = np.asarray(le)

    # correct sign: D'' = -D -> D = sin(lam) (focuses, turns over)
    D_focus = np.asarray(
        solvers.solve_jacobi_scalar(ts, jnp.full_like(ts, -1.0), le, rtol=1e-11, atol=1e-13)[:, 0]
    )
    assert np.max(np.abs(D_focus - np.sin(lam))) < 1e-8

    # perFLRW sign: D'' = +D -> D = sinh(lam) (runs away, never turns over)
    D_runaway = np.asarray(
        solvers.solve_jacobi_scalar(ts, jnp.full_like(ts, +1.0), le, rtol=1e-11, atol=1e-13)[:, 0]
    )
    assert np.max(np.abs(D_runaway - np.sinh(lam))) < 1e-7
    # at lam=3: sinh(3)=10.0 vs sin(3)=0.14  -> ~70x divergence
    assert D_runaway[-1] / D_focus[-1] > 50.0


def test_perflrw_background_phi00_sign():
    """perFLRW.background_phi00 must be NEGATIVE for ordinary matter (focusing).

    Regression test for the sign fix (draft eq:driving bg,
    Phi00^(0) = -(E/a)^2 (H^2-H') = -4 pi G (rho+P)/a^2 < 0). Run with pyccl
    installed to confirm; should PASS after the cosmology.py sign correction.
    """
    pytest.importorskip("pyccl")
    from perFLRW.cosmology import FLRWCosmology

    cosmo = FLRWCosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96, sigma8=0.8)
    for z in (0.1, 0.5, 1.0, 2.0):
        phi = float(cosmo.background_phi00(z))
        assert phi < 0.0, (
            f"background_phi00(z={z}) = {phi:.3e} > 0; draft eq:driving bg "
            f"requires Phi00 < 0 (focusing). perFLRW sign error."
        )


def test_phi00_fluctuation_prefactor_sign():
    """The delta_m -> Phi00^(1) prefactor must be NEGATIVE (focusing).

    First principles (draft eq:Phi00 scalar, Phi=Psi): the two-spatial-derivative
    terms reduce to Phi00 ~ -(E/a)^2 grad^2 Phi, and Poisson grad^2 Phi =
    4 pi G a^2 rho_m delta gives Phi00 ~ -4 pi G rho_m (E/a)^2 delta < 0. Matches
    Phi00 ~= -A delta_m. Run with pyccl; should PASS after the sign correction.
    """
    pytest.importorskip("pyccl")
    from perFLRW.cosmology import FLRWCosmology

    cosmo = FLRWCosmology(Omega_c=0.25, Omega_b=0.05, h=0.7, n_s=0.96, sigma8=0.8)
    for a, rho_m in [(1.0, 1.0), (0.5, 2.0), (0.25, 8.0)]:
        pref = float(cosmo._phi00_delta_prefactor(a, rho_m))
        assert pref < 0.0, f"phi00 conversion prefactor {pref} >= 0; must be <0 (focusing)"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
