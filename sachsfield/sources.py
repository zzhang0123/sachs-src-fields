"""
Source field generation with lambda-correlations.

Provides two geometry backends:
  - FullSkySource: healpy-based spherical harmonics
  - FlatSkySource: 2D FFT on a rectangular patch
"""

import numpy as np
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Full-sky (healpy) source
# ---------------------------------------------------------------------------

class FullSkySource:
    """Gaussian random source fields on the full sky, correlated in lambda.

    Parameters
    ----------
    cl_func : callable(lam) -> tuple of 3 ndarrays
        Returns (cl_11, cl_22, cl_33) each indexed by ell (starting at ell=0).
    lam_samples : ndarray
        Lambda values at which to generate source maps.
    nside : int
        Healpix nside parameter.
    corr_func : callable(dlam) -> float, optional
        Correlation coefficient rho(|dlam|) in [0,1].
        Default: no correlation (delta function).
    seed : int, optional
        Random seed for reproducibility.
    n_fields : int
        Number of fields (3 or 4). Default 3. Field 4 source is always 0.
    """

    def __init__(self, cl_func, lam_samples, nside, corr_func=None,
                 seed=None, n_fields=3):
        self.cl_func = cl_func
        self.lam_samples = np.asarray(lam_samples)
        self.nside = nside
        self.corr_func = corr_func
        self.seed = seed
        self.n_fields = n_fields
        self._generated = False
        self._maps = None       # (n_source_fields, n_lam, npix)
        self._sbar = None       # (n_fields, n_lam)
        self._delta_s = None    # (n_source_fields, n_lam, npix)

    @property
    def npix(self):
        import healpy as hp
        return hp.nside2npix(self.nside)

    @property
    def n_source_fields(self):
        """Number of fields with nonzero source (always 3: fields 1,2,3)."""
        return 3

    def generate(self):
        """Pre-generate all source maps with lambda correlations."""
        import healpy as hp

        rng = np.random.default_rng(self.seed)
        n_lam = len(self.lam_samples)
        npix = self.npix
        lmax = 3 * self.nside - 1
        n_alm = hp.Alm.getsize(lmax)

        # Evaluate Cl at all lambda samples
        # cls_all[field][i_lam] = cl array of length lmax+1
        cls_all = [[] for _ in range(3)]
        for lam in self.lam_samples:
            cl_11, cl_22, cl_33 = self.cl_func(lam)
            for f, cl in enumerate([cl_11, cl_22, cl_33]):
                # Pad or truncate to lmax+1
                cl_full = np.zeros(lmax + 1)
                n = min(len(cl), lmax + 1)
                cl_full[:n] = cl[:n]
                cls_all[f].append(cl_full)

        # Generate correlated alm for each field independently
        maps = np.zeros((3, n_lam, npix))

        # Pre-compute correlation matrix in lambda (shared across ell)
        if self.corr_func is not None:
            dlam = np.abs(self.lam_samples[:, None] - self.lam_samples[None, :])
            rho = self.corr_func(dlam)
        else:
            rho = np.eye(n_lam)

        for f in range(3):
            cls_f = np.array(cls_all[f])  # (n_lam, lmax+1)
            alm_arrays = [np.zeros(n_alm, dtype=complex) for _ in range(n_lam)]

            for ell in range(lmax + 1):
                cl_at_ell = cls_f[:, ell]
                sqrt_cl = np.sqrt(np.maximum(cl_at_ell, 0.0))

                K = np.outer(sqrt_cl, sqrt_cl) * rho
                K_diag_max = np.max(np.diag(K))
                if K_diag_max > 0:
                    K += 1e-12 * K_diag_max * np.eye(n_lam)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(K)
                    eigvals = np.maximum(eigvals, 0.0)
                    L = eigvecs * np.sqrt(eigvals)[None, :]

                for m in range(ell + 1):
                    idx = hp.Alm.getidx(lmax, ell, m)
                    if m == 0:
                        z = rng.standard_normal(n_lam)
                        vals = L @ z
                        for i_lam in range(n_lam):
                            alm_arrays[i_lam][idx] = vals[i_lam]
                    else:
                        z_re = rng.standard_normal(n_lam)
                        z_im = rng.standard_normal(n_lam)
                        vals = (L @ z_re + 1j * L @ z_im) / np.sqrt(2.0)
                        for i_lam in range(n_lam):
                            alm_arrays[i_lam][idx] = vals[i_lam]

            # Inverse SHT for each lambda
            for i_lam in range(n_lam):
                maps[f, i_lam, :] = hp.alm2map(alm_arrays[i_lam], self.nside)

        # Compute sky averages and fluctuations
        self._sbar = np.zeros((self.n_fields, n_lam))
        for f in range(3):
            self._sbar[f, :] = np.mean(maps[f], axis=1)

        self._delta_s = np.zeros((3, n_lam, npix))
        for f in range(3):
            self._delta_s[f] = maps[f] - self._sbar[f, :, None]

        self._maps = maps
        self._generated = True

    def sbar_interpolator(self):
        """Return callable sbar(lam) -> ndarray (n_fields,).

        Uses linear interpolation along lambda.
        """
        if not self._generated:
            self.generate()

        interps = []
        for a in range(self.n_fields):
            interps.append(interp1d(
                self.lam_samples, self._sbar[a],
                kind='linear', fill_value='extrapolate',
            ))

        def sbar(lam):
            return np.array([interps[a](lam) for a in range(self.n_fields)])

        return sbar

    def delta_s_interpolator(self):
        """Return callable delta_s(lam) -> ndarray (n_fields, npix).

        Uses linear interpolation along lambda.
        """
        if not self._generated:
            self.generate()

        npix = self.npix
        n_fields = self.n_fields

        # Build interpolators for each source field
        interps = []
        for f in range(3):
            interps.append(interp1d(
                self.lam_samples, self._delta_s[f],
                axis=0, kind='linear', fill_value='extrapolate',
            ))

        def delta_s(lam):
            result = np.zeros((n_fields, npix))
            for f in range(3):
                result[f] = interps[f](lam)
            return result

        return delta_s


# ---------------------------------------------------------------------------
# Flat-sky (2D FFT) source
# ---------------------------------------------------------------------------

class FlatSkySource:
    """Gaussian random source fields on a flat rectangular patch.

    Parameters
    ----------
    cl_func : callable(lam) -> tuple of 3 ndarrays
        Returns (cl_11, cl_22, cl_33) each indexed by ell (starting at ell=0).
    lam_samples : ndarray
        Lambda values at which to generate source maps.
    nx, ny : int
        Grid dimensions in pixels.
    dx : float
        Pixel angular size in radians.
    corr_func : callable(dlam) -> float, optional
        Correlation coefficient rho(|dlam|) in [0,1].
    seed : int, optional
        Random seed.
    n_fields : int
        Number of fields (3 or 4). Default 3.
    """

    def __init__(self, cl_func, lam_samples, nx, ny, dx,
                 corr_func=None, seed=None, n_fields=3):
        self.cl_func = cl_func
        self.lam_samples = np.asarray(lam_samples)
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.corr_func = corr_func
        self.seed = seed
        self.n_fields = n_fields
        self._generated = False
        self._maps = None
        self._sbar = None
        self._delta_s = None

    @property
    def npix(self):
        return self.nx * self.ny

    @property
    def shape(self):
        return (self.nx, self.ny)

    def _ell_grid(self):
        """Compute the ell value for each 2D Fourier mode."""
        kx = np.fft.fftfreq(self.nx, d=self.dx) * 2.0 * np.pi
        ky = np.fft.fftfreq(self.ny, d=self.dx) * 2.0 * np.pi
        kx2d, ky2d = np.meshgrid(kx, ky, indexing='ij')
        ell_grid = np.sqrt(kx2d**2 + ky2d**2)
        return ell_grid

    def _cl_to_power_grid(self, cl, ell_grid):
        """Interpolate C_l onto the 2D ell grid.

        Returns P(ell) = C_l / (dx^2) to get correct variance per pixel
        when using FFT normalization.
        """
        ells = np.arange(len(cl))
        # Interpolate C_l to continuous ell values
        cl_interp = interp1d(ells, cl, kind='linear',
                             bounds_error=False, fill_value=0.0)
        power = cl_interp(ell_grid) / self.dx**2
        return power

    def generate(self):
        """Pre-generate all source maps with lambda correlations."""
        rng = np.random.default_rng(self.seed)
        n_lam = len(self.lam_samples)
        nx, ny = self.nx, self.ny
        npix = self.npix
        ell_grid = self._ell_grid()

        # Flatten ell grid for vectorized operations
        ell_flat = ell_grid.ravel()
        n_modes = len(ell_flat)

        maps = np.zeros((3, n_lam, npix))

        for f in range(3):
            # Evaluate power at all (ell, lam) combinations
            # power_all[i_lam, i_mode]
            power_all = np.zeros((n_lam, n_modes))
            for i_lam, lam in enumerate(self.lam_samples):
                cls = self.cl_func(lam)
                cl = cls[f]
                ells = np.arange(len(cl))
                cl_interp = interp1d(ells, cl, kind='linear',
                                     bounds_error=False, fill_value=0.0)
                power_all[i_lam, :] = cl_interp(ell_flat) / self.dx**2

            # For each Fourier mode, generate correlated Gaussian across lambda
            sqrt_power = np.sqrt(np.maximum(power_all, 0.0))

            # Build per-mode correlation and generate
            fourier_maps = np.zeros((n_lam, n_modes), dtype=complex)

            if self.corr_func is not None:
                dlam = np.abs(self.lam_samples[:, None] - self.lam_samples[None, :])
                rho = self.corr_func(dlam)
            else:
                rho = np.eye(n_lam)

            # Group modes by similar power profile to batch Cholesky decompositions
            # For simplicity, iterate over unique ell bins
            ell_unique = np.unique(np.round(ell_flat, decimals=2))

            for ell_val in ell_unique:
                mask = np.abs(ell_flat - ell_val) < 0.015
                if not np.any(mask):
                    continue

                n_masked = np.sum(mask)
                # Use power at this ell (take first match, they're ~identical)
                idx_repr = np.where(mask)[0][0]
                sp = sqrt_power[:, idx_repr]  # (n_lam,)

                K = np.outer(sp, sp) * rho
                K_diag_max = np.max(np.diag(K))
                if K_diag_max > 0:
                    K += 1e-12 * K_diag_max * np.eye(n_lam)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(K)
                    eigvals = np.maximum(eigvals, 0.0)
                    L = eigvecs * np.sqrt(eigvals)[None, :]

                # Generate for all modes in this bin at once
                z_re = rng.standard_normal((n_lam, n_masked))
                z_im = rng.standard_normal((n_lam, n_masked))
                vals = (L @ z_re + 1j * L @ z_im) / np.sqrt(2.0)
                fourier_maps[:, mask] = vals

            # Inverse FFT to get real-space maps
            for i_lam in range(n_lam):
                fk = fourier_maps[i_lam].reshape(nx, ny)
                maps[f, i_lam, :] = np.real(np.fft.ifft2(fk)).ravel()

        # Compute sky averages and fluctuations
        self._sbar = np.zeros((self.n_fields, n_lam))
        for f in range(3):
            self._sbar[f, :] = np.mean(maps[f], axis=1)

        self._delta_s = np.zeros((3, n_lam, npix))
        for f in range(3):
            self._delta_s[f] = maps[f] - self._sbar[f, :, None]

        self._maps = maps
        self._generated = True

    def sbar_interpolator(self):
        """Return callable sbar(lam) -> ndarray (n_fields,)."""
        if not self._generated:
            self.generate()

        interps = []
        for a in range(self.n_fields):
            interps.append(interp1d(
                self.lam_samples, self._sbar[a],
                kind='linear', fill_value='extrapolate',
            ))

        def sbar(lam):
            return np.array([interps[a](lam) for a in range(self.n_fields)])

        return sbar

    def delta_s_interpolator(self):
        """Return callable delta_s(lam) -> ndarray (n_fields, npix)."""
        if not self._generated:
            self.generate()

        npix = self.npix
        n_fields = self.n_fields

        interps = []
        for f in range(3):
            interps.append(interp1d(
                self.lam_samples, self._delta_s[f],
                axis=0, kind='linear', fill_value='extrapolate',
            ))

        def delta_s(lam):
            result = np.zeros((n_fields, npix))
            for f in range(3):
                result[f] = interps[f](lam)
            return result

        return delta_s
