"""
Fluctuation ODE solver on pixel arrays (full-sky or flat-sky).

Solves:
  xi_dot_a(n) = M_{ab}(lam) xi_b(n) + F_{abc} xi_b(n) xi_c(n) + delta_s_a(lam, n)
with IC:  xi_a = 0.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .coefficients import quadratic, M_matrix


class FluctuationResult:
    """Result of fluctuation integration.

    Attributes
    ----------
    lam : ndarray, shape (N,)
        Output lambda values.
    xi : ndarray, shape (N, n_fields, npix)
        Fluctuation fields at output times.
    n_fields : int
    npix : int
    """

    def __init__(self, lam, xi, n_fields, npix):
        self.lam = lam
        self.xi = xi
        self.n_fields = n_fields
        self.npix = npix

    def get_maps(self, lam_index, field=None):
        """Get field map(s) at a given output time index.

        Parameters
        ----------
        lam_index : int
            Index into self.lam.
        field : int, optional
            If given, return map for this field only.

        Returns
        -------
        ndarray, shape (n_fields, npix) or (npix,)
        """
        if field is not None:
            return self.xi[lam_index, field]
        return self.xi[lam_index]

    def power_spectra_fullsky(self, lam_index, nside):
        """Compute auto-spectra C_l^{aa} at a given output time (full-sky).

        Parameters
        ----------
        lam_index : int
        nside : int

        Returns
        -------
        list of ndarray
            C_l for each field.
        """
        import healpy as hp
        cls = []
        for a in range(self.n_fields):
            cl = hp.anafast(self.xi[lam_index, a])
            cls.append(cl)
        return cls

    def power_spectra_flatsky(self, lam_index, nx, ny, dx):
        """Compute auto-spectra P(k) at a given output time (flat-sky).

        Parameters
        ----------
        lam_index : int
        nx, ny : int
        dx : float
            Pixel size in radians.

        Returns
        -------
        ells : ndarray
            Ell bin centers.
        cls : list of ndarray
            Binned power spectrum for each field.
        """
        kx = np.fft.fftfreq(nx, d=dx) * 2.0 * np.pi
        ky = np.fft.fftfreq(ny, d=dx) * 2.0 * np.pi
        kx2d, ky2d = np.meshgrid(kx, ky, indexing='ij')
        ell_grid = np.sqrt(kx2d**2 + ky2d**2)

        # Bin settings
        ell_max = np.max(ell_grid) / 2.0
        n_bins = 30
        ell_edges = np.linspace(0, ell_max, n_bins + 1)
        ell_centers = 0.5 * (ell_edges[:-1] + ell_edges[1:])

        cls_list = []
        for a in range(self.n_fields):
            map_2d = self.xi[lam_index, a].reshape(nx, ny)
            fk = np.fft.fft2(map_2d) * dx**2
            pk = np.abs(fk)**2 / (nx * ny * dx**2)

            # Bin the power spectrum
            cl_binned = np.zeros(n_bins)
            for b in range(n_bins):
                mask = (ell_grid >= ell_edges[b]) & (ell_grid < ell_edges[b + 1])
                if np.any(mask):
                    cl_binned[b] = np.mean(pk[mask])
            cls_list.append(cl_binned)

        return ell_centers, cls_list


class FluctuationSolver:
    """Solver for the fluctuation ODE on pixel arrays.

    Parameters
    ----------
    saddle_result : SaddlePointResult
        Solved saddle point providing chi(lam) and M(lam).
    delta_s_func : callable(lam) -> ndarray (n_fields, npix)
        Source fluctuation interpolator.
    npix : int
        Number of pixels.
    n_fields : int
        Number of fields (3 or 4). Default 3.
    linear_only : bool
        If True, drop the F_{abc} xi_b xi_c nonlinear term. Default False.
    lam_span : (float, float), optional
        Override integration interval. Default: use saddle_result.lam span.
    method : str
        ODE solver method. Default 'RK45'.
    **ivp_kwargs
        Additional arguments to solve_ivp (rtol, atol, max_step, etc.).
    """

    def __init__(self, saddle_result, delta_s_func, npix,
                 n_fields=3, linear_only=False, lam_span=None,
                 method='RK45', **ivp_kwargs):
        self.saddle_result = saddle_result
        self.delta_s_func = delta_s_func
        self.npix = npix
        self.n_fields = n_fields
        self.linear_only = linear_only
        self.lam_span = lam_span or (saddle_result.lam[0], saddle_result.lam[-1])
        self.method = method
        self.ivp_kwargs = ivp_kwargs

    def _rhs(self, lam, y_flat):
        """RHS of the fluctuation ODE, vectorized over pixels."""
        xi = y_flat.reshape(self.n_fields, self.npix)

        # Linear term: M(lam) @ xi
        M = self.saddle_result.M(lam)  # (n_fields, n_fields)
        result = M @ xi                # (n_fields, npix)

        # Nonlinear term
        if not self.linear_only:
            result += quadratic(xi, self.n_fields)

        # Source term
        result += self.delta_s_func(lam)

        return result.ravel()

    def solve(self, t_eval=None):
        """Run the integration.

        Parameters
        ----------
        t_eval : ndarray, optional
            Lambda values at which to store the solution.

        Returns
        -------
        FluctuationResult
        """
        y0 = np.zeros(self.n_fields * self.npix)

        sol = solve_ivp(
            self._rhs,
            self.lam_span,
            y0,
            method=self.method,
            t_eval=t_eval,
            **self.ivp_kwargs,
        )

        if not sol.success:
            raise RuntimeError(f"Fluctuation integration failed: {sol.message}")

        # Reshape solution: sol.y is (n_fields*npix, N_times)
        n_times = len(sol.t)
        xi = sol.y.T.reshape(n_times, self.n_fields, self.npix)

        return FluctuationResult(sol.t, xi, self.n_fields, self.npix)
