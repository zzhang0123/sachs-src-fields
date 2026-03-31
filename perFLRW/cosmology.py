"""
FLRW cosmology wrapper for Sachs optical equation sources.

Provides:
  - Affine parameter <-> redshift mapping
  - Background Phi_00 (Ricci focusing term)
  - Angular power spectra C_l^{Phi_00} and C_l^{Psi_0}
  - Convenience callables for sachsfield source functions
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

import pyccl as ccl


class FLRWCosmology:
    """FLRW cosmology interface for the Sachs optical equations.

    Wraps a PyCCL Cosmology to provide the affine parameter mapping,
    background Ricci focusing term, and angular power spectra needed
    by the sachsfield solver.

    Parameters
    ----------
    Omega_c : float
        Cold dark matter density fraction.
    Omega_b : float
        Baryon density fraction.
    h : float
        Dimensionless Hubble constant H0 / (100 km/s/Mpc).
    sigma8 : float
        RMS matter fluctuations in 8 Mpc/h spheres.
    n_s : float
        Scalar spectral index.
    E_co : float
        Comoving photon energy (constant along null geodesic), default 1.0.
        Appears in both the affine parameter mapping and Phi_00.
    z_max : float
        Maximum redshift for interpolation tables, default 10.0.
    n_table : int
        Number of points in interpolation tables, default 2000.
    **ccl_kwargs
        Extra keyword arguments passed to ``pyccl.Cosmology``
        (e.g., ``transfer_function``, ``w0``, ``wa``).
    """

    def __init__(self, Omega_c=0.25, Omega_b=0.05, h=0.7,
                 sigma8=0.8, n_s=0.96, E_co=1.0,
                 z_max=10.0, n_table=2000, **ccl_kwargs):
        self.cosmo = ccl.Cosmology(
            Omega_c=Omega_c, Omega_b=Omega_b, h=h,
            sigma8=sigma8, n_s=n_s, **ccl_kwargs,
        )
        self.E_co = E_co
        self.z_max = z_max

        # Physical constants derived from cosmology
        self.H0_Mpc = h * 100.0 / 299792.458  # H0 in 1/Mpc
        self._rho_crit_0 = ccl.background.rho_x(
            self.cosmo, 1.0, 'critical', is_comoving=False)

        # 4 pi G = (3/2) H0^2 / rho_crit_0  (in Mpc-based units)
        self._four_pi_G = 1.5 * self.H0_Mpc**2 / self._rho_crit_0

        # Dark energy EOS parameters (CPL: w = w0 + wa*(1-a))
        self._w0 = ccl_kwargs.get('w0', -1.0)
        self._wa = ccl_kwargs.get('wa', 0.0)

        # Build interpolation tables
        self._build_lambda_tables(z_max, n_table)

    @classmethod
    def from_ccl(cls, cosmo, E_co=1.0, z_max=10.0, n_table=2000):
        """Create from an existing pyccl.Cosmology object.

        Parameters
        ----------
        cosmo : pyccl.Cosmology
            Pre-configured cosmology.
        E_co : float
            Comoving photon energy.
        z_max : float
            Maximum redshift for tables.
        n_table : int
            Table resolution.
        """
        obj = cls.__new__(cls)
        obj.cosmo = cosmo
        obj.E_co = E_co
        obj.z_max = z_max

        h = cosmo['h']
        obj.H0_Mpc = h * 100.0 / 299792.458
        obj._rho_crit_0 = ccl.background.rho_x(
            cosmo, 1.0, 'critical', is_comoving=False)
        obj._four_pi_G = 1.5 * obj.H0_Mpc**2 / obj._rho_crit_0

        obj._w0 = cosmo['w0']
        obj._wa = cosmo['wa']

        obj._build_lambda_tables(z_max, n_table)
        return obj

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_lambda_tables(self, z_max, n_table):
        """Build dense interpolation tables for lambda(z) and z(lambda)."""
        z_tab = np.linspace(0.0, z_max, n_table)
        a_tab = 1.0 / (1.0 + z_tab)

        # H(z) in 1/Mpc
        H_tab = ccl.h_over_h0(self.cosmo, a_tab) * self.H0_Mpc

        # d lambda / dz = -1 / ((1+z)^2 * H(z) * E_co)
        dlam_dz = -1.0 / ((1.0 + z_tab)**2 * H_tab * self.E_co)

        # Integrate: lambda(z=0) = 0
        lam_tab = np.zeros(n_table)
        lam_tab[1:] = cumulative_trapezoid(dlam_dz, z_tab)

        self._z_tab = z_tab
        self._lam_tab = lam_tab

        # lambda(z) interpolator
        self._lam_of_z_interp = interp1d(
            z_tab, lam_tab, kind='cubic', bounds_error=True)

        # z(lambda) interpolator (lambda is monotonically decreasing)
        self._z_of_lam_interp = interp1d(
            lam_tab, z_tab, kind='cubic', bounds_error=True)

    def _w_de(self, a):
        """Dark energy equation of state w(a) = w0 + wa*(1-a)."""
        return self._w0 + self._wa * (1.0 - a)

    # ------------------------------------------------------------------
    # 1. Affine parameter <-> redshift
    # ------------------------------------------------------------------

    def lambda_of_z(self, z):
        """Affine parameter lambda as a function of redshift.

        Parameters
        ----------
        z : float or ndarray
            Redshift(s). Must be in [0, z_max].

        Returns
        -------
        lam : float or ndarray
            Affine parameter (negative for z > 0).
        """
        return self._lam_of_z_interp(z)

    def z_of_lambda(self, lam):
        """Redshift as a function of affine parameter.

        Parameters
        ----------
        lam : float or ndarray
            Affine parameter (must be <= 0).

        Returns
        -------
        z : float or ndarray
            Redshift.
        """
        return self._z_of_lam_interp(lam)

    # ------------------------------------------------------------------
    # 2. Background Phi_00
    # ------------------------------------------------------------------

    def background_phi00(self, z):
        """Background Ricci focusing term Phi_00(z).

        Computes Phi_00 = 4 pi G (rho + P)_total (E_co / a)^2
        using the standard Sachs convention.

        Parameters
        ----------
        z : float or ndarray
            Redshift(s).

        Returns
        -------
        phi00 : float or ndarray
            Background Phi_00 in units of 1/Mpc^2.
        """
        z = np.atleast_1d(np.asarray(z, dtype=float))
        a = 1.0 / (1.0 + z)

        # Component densities (physical, M_sun/Mpc^3)
        rho_m = np.array([
            ccl.background.rho_x(self.cosmo, ai, 'matter', is_comoving=False)
            for ai in a])
        rho_r = np.array([
            ccl.background.rho_x(self.cosmo, ai, 'radiation', is_comoving=False)
            for ai in a])
        rho_de = np.array([
            ccl.background.rho_x(self.cosmo, ai, 'dark_energy', is_comoving=False)
            for ai in a])

        # (rho + P) for each component
        # Matter: P = 0
        # Radiation: P = rho/3
        # Dark energy: P = w(a) rho
        w = self._w_de(a)
        sum_rho_p = rho_m + (4.0 / 3.0) * rho_r + (1.0 + w) * rho_de

        phi00 = self._four_pi_G * sum_rho_p * (self.E_co / a)**2

        return float(phi00[0]) if phi00.size == 1 else phi00

    def background_phi00_of_lambda(self, lam):
        """Background Phi_00 as a function of affine parameter.

        Parameters
        ----------
        lam : float or ndarray
            Affine parameter (must be <= 0).

        Returns
        -------
        phi00 : float or ndarray
        """
        z = self.z_of_lambda(lam)
        return self.background_phi00(z)

    # ------------------------------------------------------------------
    # 3. Angular power spectra
    # ------------------------------------------------------------------

    def sachs_cls(self, z, ell=None, lmax=2000, delta_z=0.005):
        """Angular power spectra of Phi_00 and Psi_0 at a given redshift.

        Uses a custom PyCCL tracer with a narrow Gaussian selection
        function around the target redshift.

        Parameters
        ----------
        z : float
            Target redshift.
        ell : ndarray, optional
            Multipole values. Default: np.arange(2, lmax+1).
        lmax : int
            Maximum multipole (used if ell is None).
        delta_z : float
            Width of the Gaussian selection function in redshift.

        Returns
        -------
        ell : ndarray
            Multipole values.
        cl_phi00 : ndarray
            C_ell for Phi_00 (Ricci focusing).
        cl_psi0 : ndarray
            C_ell for Psi_0 (Weyl shear).
        """
        if ell is None:
            ell = np.arange(2, lmax + 1, dtype=float)

        # Build tracer and compute auto-spectrum
        tracer_phi00, _ = self._make_phi00_tracer(z, delta_z=delta_z)
        cl_phi00 = ccl.angular_cl(self.cosmo, tracer_phi00, tracer_phi00, ell)

        # --- Psi_0 tracer (Weyl shear) ---
        # Psi_0 is the Weyl lensing shear sourced by matter.
        # At high ell (Limber), the spin-2 Weyl contribution differs
        # from the spin-0 Ricci by the geometric factor:
        #   C_l^{Psi_0} = C_l^{Phi_00} * l(l+1) / ((l+2)(l-1))
        # This comes from the Bessel function derivative for spin-2.
        cl_psi0 = cl_phi00 * ell * (ell + 1.0) / ((ell + 2.0) * (ell - 1.0))

        return ell, cl_phi00, cl_psi0

    def _make_phi00_tracer(self, z, delta_z=0.005):
        """Build a PyCCL custom tracer for Phi_00 at a given redshift shell.

        Parameters
        ----------
        z : float
            Central redshift of the narrow shell.
        delta_z : float
            Gaussian width of the shell.

        Returns
        -------
        tracer : ccl.Tracer
        chi_arr : ndarray
            Comoving distances for the shell.
        """
        z_arr = np.linspace(max(z - 5.0 * delta_z, 1e-6),
                            z + 5.0 * delta_z, 500)
        a_arr = 1.0 / (1.0 + z_arr)
        chi_arr = ccl.comoving_radial_distance(self.cosmo, a_arr)

        pz = np.exp(-0.5 * ((z_arr - z) / delta_z)**2)
        pz /= np.trapezoid(pz, z_arr)

        rho_m_arr = np.array([
            ccl.background.rho_x(self.cosmo, ai, 'matter', is_comoving=False)
            for ai in a_arr])

        conversion = self._four_pi_G * rho_m_arr * self.E_co**2 / a_arr**2

        H_arr = ccl.h_over_h0(self.cosmo, a_arr) * self.H0_Mpc
        dchi_dz = 1.0 / H_arr

        kernel = conversion * pz / dchi_dz

        tracer = ccl.Tracer()
        tracer.add_tracer(
            self.cosmo,
            kernel=(chi_arr, kernel),
            der_bessel=0, der_angles=0,
        )
        return tracer, chi_arr

    def sachs_cls_cross(self, z_array, ell=None, lmax=200,
                        delta_z=0.005, field='phi00'):
        """Cross-redshift angular power spectrum matrix C_ell(z_i, z_j).

        Computes the full z-z cross-spectrum at each ell by building
        narrow-shell tracers and calling ``ccl.angular_cl`` for all pairs.

        Parameters
        ----------
        z_array : ndarray, shape (N,)
            Redshift values for the grid.
        ell : ndarray, optional
            Multipole values. Default: np.arange(2, lmax+1).
        lmax : int
            Maximum multipole (used if ell is None).
        delta_z : float
            Width of each Gaussian shell.
        field : str
            'phi00' for Ricci, 'psi0' for Weyl shear.

        Returns
        -------
        ell : ndarray, shape (n_ell,)
            Multipole values.
        cl_matrix : ndarray, shape (N, N, n_ell)
            C_ell(z_i, z_j) for each ell.
        """
        z_array = np.atleast_1d(z_array)
        n_z = len(z_array)

        if ell is None:
            ell = np.arange(2, lmax + 1, dtype=float)
        n_ell = len(ell)

        # Build all tracers
        tracers = []
        for zi in z_array:
            tr, _ = self._make_phi00_tracer(zi, delta_z=delta_z)
            tracers.append(tr)

        # Compute cross-spectra for all (i, j) pairs with j >= i
        cl_matrix = np.zeros((n_z, n_z, n_ell))
        for i in range(n_z):
            for j in range(i, n_z):
                cl_ij = ccl.angular_cl(self.cosmo, tracers[i], tracers[j], ell)
                if field == 'psi0':
                    cl_ij = cl_ij * ell * (ell + 1.0) / ((ell + 2.0) * (ell - 1.0))
                cl_matrix[i, j, :] = cl_ij
                cl_matrix[j, i, :] = cl_ij  # symmetric

        return ell, cl_matrix

    def sachs_cls_of_lambda(self, lam, **kwargs):
        """Angular power spectra as a function of affine parameter.

        Parameters
        ----------
        lam : float
            Affine parameter (must be <= 0).
        **kwargs
            Passed to :meth:`sachs_cls`.

        Returns
        -------
        ell, cl_phi00, cl_psi0 : ndarrays
        """
        z = self.z_of_lambda(lam)
        return self.sachs_cls(float(z), **kwargs)

    # ------------------------------------------------------------------
    # 4. sachsfield-compatible callables
    # ------------------------------------------------------------------

    def sbar_func(self, n_fields=3):
        """Return a callable for the saddle-point source.

        The background source is s_bar_1(lambda) = Phi_00^{bg}(lambda),
        with all other components zero.

        Parameters
        ----------
        n_fields : int
            Number of fields (3 or 4).

        Returns
        -------
        sbar : callable
            sbar(lam) -> ndarray of shape (n_fields,).
        """
        def _sbar(lam):
            phi00 = self.background_phi00_of_lambda(lam)
            out = np.zeros(n_fields)
            out[0] = phi00
            return out
        return _sbar

    def cl_func(self, lam_array=None, z_array=None,
                lmax=96, delta_z=0.005, n_fields=3):
        """Return a callable for sachsfield source power spectra.

        Pre-computes C_ell at a grid of redshifts and returns an
        interpolating callable ``cl_func(lam) -> (cl_11, cl_22, cl_33)``.

        Parameters
        ----------
        lam_array : ndarray, optional
            Lambda values at which to pre-compute. Takes precedence.
        z_array : ndarray, optional
            Redshift values at which to pre-compute.
            One of lam_array or z_array must be given.
        lmax : int
            Maximum multipole.
        delta_z : float
            Width of Gaussian selection for each shell.
        n_fields : int
            Number of fields (3 or 4).

        Returns
        -------
        cl_func : callable
            cl_func(lam) -> tuple of (cl_11, cl_22, cl_33) arrays,
            each of length lmax+1 (indexed by ell).
        """
        if lam_array is not None:
            z_arr = self.z_of_lambda(lam_array)
            lam_arr = np.asarray(lam_array, dtype=float)
        elif z_array is not None:
            z_arr = np.asarray(z_array, dtype=float)
            lam_arr = self.lambda_of_z(z_arr)
        else:
            raise ValueError("Must provide either lam_array or z_array.")

        ell_out = np.arange(lmax + 1, dtype=float)
        ell_query = ell_out[2:]  # start from ell=2

        n_samples = len(z_arr)
        cl_phi00_table = np.zeros((n_samples, lmax + 1))
        cl_psi0_table = np.zeros((n_samples, lmax + 1))

        for i, zi in enumerate(z_arr):
            if zi < 1e-4:
                # At z~0, power spectra are ill-defined; set to zero
                continue
            _, cl_p, cl_s = self.sachs_cls(
                float(zi), ell=ell_query, lmax=lmax, delta_z=delta_z)
            cl_phi00_table[i, 2:] = cl_p
            cl_psi0_table[i, 2:] = cl_s

        # Interpolators for each ell, indexed by lambda
        # Shape: (n_ell, n_lam) -> interp along lambda axis
        _interp_phi00 = interp1d(
            lam_arr, cl_phi00_table, axis=0,
            kind='linear', bounds_error=False, fill_value=0.0)
        _interp_psi0 = interp1d(
            lam_arr, cl_psi0_table, axis=0,
            kind='linear', bounds_error=False, fill_value=0.0)

        def _cl_func(lam):
            cl_11 = _interp_phi00(lam)
            cl_22 = _interp_psi0(lam)
            cl_33 = cl_22.copy()  # statistical isotropy: both shear components
            cl_11 = np.maximum(cl_11, 0.0)
            cl_22 = np.maximum(cl_22, 0.0)
            cl_33 = np.maximum(cl_33, 0.0)
            if n_fields == 3:
                return cl_11, cl_22, cl_33
            else:
                return cl_11, cl_22, cl_33

        return _cl_func

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self):
        c = self.cosmo
        return (f"FLRWCosmology(Omega_c={c['Omega_c']:.4f}, "
                f"Omega_b={c['Omega_b']:.4f}, h={c['h']:.3f}, "
                f"sigma8={c['sigma8']:.3f}, n_s={c['n_s']:.3f}, "
                f"E_co={self.E_co})")
