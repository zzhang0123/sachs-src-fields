"""
End-to-end tests for the field-generation adapter: driving-field realisations
-> DrivingField -> trace_rays -> kappa/gamma/omega maps.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import sachsray as sr  # noqa: E402


def _synthetic_components(n_lam, npix, seed=0):
    """Smooth-in-lambda, band-limited-ish random fluctuation maps."""
    rng = np.random.default_rng(seed)
    lam = np.linspace(0.0, 2.0, n_lam)
    out = []
    for scale in (1.0, 0.5, 0.5):  # Phi00, RePsi, ImPsi
        ks = np.arange(1, 12)
        amp = scale * 0.02 / np.sqrt(ks)
        ph = rng.uniform(0, 2 * np.pi, (npix, ks.size))
        m = (amp[None, None, :] * np.sin(ks[None, None, :] * lam[None, :, None] + ph[:, None, :])).sum(-1)
        out.append(m.T)  # (n_lam, npix)
    return lam, out[0], out[1], out[2]


def test_driving_from_components_shapes_and_trace():
    n_lam, npix = 64, 96
    lam, dphi, dpre, dpim = _synthetic_components(n_lam, npix)
    phi00_bg = np.full(n_lam, -0.05)
    field = sr.driving_from_components(lam, dphi, dpre, dpim, phi00_bg, dtype=np.float64)

    assert field.maps.shape == (npix, n_lam, 3)
    assert field.npix == npix
    # total Phi00 = bg + fluctuation
    assert np.allclose(np.asarray(field.maps[:, :, 0]), phi00_bg[None, :] + dphi.T, atol=1e-6)

    out = sr.trace_rays(field, lam[-1], chunk=32, rtol=1e-7, atol=1e-9)
    for k in ("kappa", "gamma1", "gamma2", "omega"):
        assert out[k].shape == (npix,)
        assert np.all(np.isfinite(np.asarray(out[k])))


def test_driving_from_components_validation():
    lam = np.linspace(0, 1, 10)
    bad = np.zeros((5, 8))  # wrong n_lam
    with pytest.raises(ValueError):
        sr.driving_from_components(lam, bad, np.zeros((10, 8)), np.zeros((10, 8)))
    with pytest.raises(ValueError):
        sr.driving_from_components(lam, np.zeros((10, 8)), np.zeros((10, 8)),
                                   np.zeros((10, 8)), phi00_bg=np.zeros(9))


def test_driving_from_fullsky_source_end_to_end():
    """Generate a real correlated full-sky realisation (healpy) and ray-trace it."""
    hp = pytest.importorskip("healpy")
    sf = pytest.importorskip("sachsfield")

    nside = 4
    npix = hp.nside2npix(nside)  # 192
    lmax = 3 * nside - 1
    lam_samples = np.linspace(0.1, 2.0, 24)

    def cl_func(lam):
        ell = np.arange(lmax + 1)
        cl = np.zeros(lmax + 1)
        cl[1:] = 1e-4 / (ell[1:] * (ell[1:] + 1))
        return cl, cl.copy(), cl.copy()

    source = sf.FullSkySource(
        cl_func=cl_func, lam_samples=lam_samples, nside=nside,
        corr_func=lambda d: np.exp(-0.5 * d**2), seed=0, n_fields=3,
    )
    phi00_bg = np.full(len(lam_samples), -0.05)

    field = sr.driving_from_source(source, phi00_bg, dtype=np.float64)
    assert field.maps.shape == (npix, len(lam_samples), 3)

    out = sr.trace_rays(field, lam_samples[-1], chunk=64, rtol=1e-6, atol=1e-8)
    for k in ("kappa", "gamma1", "gamma2", "omega"):
        assert out[k].shape == (npix,)
        assert np.all(np.isfinite(np.asarray(out[k])))
    # a non-trivial shear field should produce some spread in convergence
    assert np.std(np.asarray(out["kappa"])) > 0.0


def test_born_convergence_limit():
    """Weak limit: kappa from the full Jacobi pipeline matches the analytic Born
    integral kappa = -int (lam_s-lam')lam'/lam_s * dPhi00 dlam' (vacuum bg, no
    shear). Validates the solver AND the observable sign end-to-end."""
    n_lam, npix, lam_s = 128, 64, 2.0
    lam = np.linspace(0.0, lam_s, n_lam)
    rng = np.random.default_rng(1)
    ks = np.arange(1, 8)
    amp = 1e-3 / np.sqrt(ks)  # small -> O(delta^2) corrections negligible
    ph = rng.uniform(0, 2 * np.pi, (npix, ks.size))
    dphi = (amp[None, None, :] * np.sin(
        ks[None, None, :] * lam[None, :, None] + ph[:, None, :])).sum(-1).T  # (n_lam, npix)
    zeros = np.zeros_like(dphi)

    field = sr.driving_from_components(lam, dphi, zeros, zeros, phi00_bg=None, dtype=np.float64)
    out = sr.trace_rays(field, lam_s, chunk=64, rtol=1e-11, atol=1e-13)
    kappa = np.asarray(out["kappa"])

    kernel = (lam_s - lam) * lam / lam_s              # standard lensing efficiency
    kappa_born = -np.trapezoid(kernel[:, None] * dphi, lam, axis=0)  # (npix,)

    rel = np.max(np.abs(kappa - kappa_born)) / np.std(kappa_born)
    assert rel < 0.05, f"kappa vs Born mismatch: rel = {rel:.3f}"
    # no shear sourced -> gamma ~ 0 at this order
    assert np.max(np.abs(np.asarray(out["gamma1"]))) < 1e-4


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
