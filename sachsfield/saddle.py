"""
Saddle-point (background) ODE solver.

Solves:  chi_dot_a = F_{abc} chi_b chi_c + sbar_a(lam)
with singular IC:  chi_1 -> y1_init/lam as lam -> 0^-,  chi_{rest} = 0.

For the standard F_{111} = -1/2, the self-consistent asymptotic is
chi_1 = 2/lam (i.e., y1_init = 2).

Internally uses a rescaled variable y_1 = lam * chi_1 (which stays O(1))
to avoid numerical stiffness near lam = 0.
"""

import numpy as np
from scipy.integrate import solve_ivp

from .coefficients import quadratic, M_matrix


class SaddlePointResult:
    """Result of saddle-point integration.

    Attributes
    ----------
    lam : ndarray, shape (N,)
        Lambda evaluation points.
    chi : ndarray, shape (n_fields, N)
        Field values at evaluation points (in original chi variables).
    n_fields : int
    """

    def __init__(self, lam, chi, sol_dense, n_fields):
        self.lam = lam
        self.chi = chi
        self.n_fields = n_fields
        self._sol_dense = sol_dense  # dense output in rescaled coords

    def __call__(self, lam):
        """Interpolate chi at arbitrary lambda value(s).

        Parameters
        ----------
        lam : float or ndarray

        Returns
        -------
        ndarray, shape (n_fields,) or (n_fields, N)
        """
        lam = np.asarray(lam, dtype=float)
        y = self._sol_dense(lam)
        return self._y_to_chi(y, lam)

    def _y_to_chi(self, y, lam):
        """Convert rescaled variables back to chi."""
        chi = y.copy()
        chi[0] = y[0] / lam  # chi_1 = y_1 / lam
        return chi

    def M(self, lam):
        """Linearization matrix at given lambda.

        Parameters
        ----------
        lam : float

        Returns
        -------
        ndarray, shape (n_fields, n_fields)
        """
        chi = self(lam)
        return M_matrix(chi, self.n_fields)


class SaddlePointSolver:
    """Solver for the saddle-point ODE.

    Internally integrates in rescaled coordinates y_1 = lam * chi_1
    (with y_{2,...} = chi_{2,...} unchanged) so the state vector stays O(1).

    The ODE in rescaled form (derived from chi_dot_1 = F_{1bc} chi_b chi_c + sbar_1
    with chi_1 = y_1/lam):
        dy_1/dlam = y_1(2 - y_1) / (2 lam) + lam * (-2 y_2^2 - 2 y_3^2 + sbar_1)
        dy_a/dlam = -y_1 y_a / lam + sbar_a   (for a >= 2)

    IC: y_1 = y1_init (default 2), y_{rest} = 0.
    y1_init=2 corresponds to the self-consistent asymptotic chi_1 = 2/lam
    for F_{111} = -1/2.

    Parameters
    ----------
    sbar_func : callable(lam) -> ndarray shape (n_fields,)
        Sky-averaged source term.
    lam_span : (float, float)
        (lam_start, lam_end) where lam_start is close to 0 (negative)
        and lam_end is more negative.
    n_fields : int
        Number of fields (3 or 4). Default 3.
    y1_init : float
        Initial value of y_1 = lam * chi_1. Default 2 (self-consistent
        with F_111=-1/2). Set to -1 if you use a different F_111.
    method : str
        ODE solver method. Default 'Radau' (implicit, handles mild stiffness).
    dense_output : bool
        Whether to compute dense output for interpolation. Default True.
    **ivp_kwargs
        Additional keyword arguments passed to solve_ivp (rtol, atol, etc.).
    """

    def __init__(self, sbar_func, lam_span, n_fields=3, y1_init=2.0,
                 method='Radau', dense_output=True, **ivp_kwargs):
        self.sbar_func = sbar_func
        self.lam_span = lam_span
        self.n_fields = n_fields
        self.y1_init = y1_init
        self.method = method
        self.dense_output = dense_output
        self.ivp_kwargs = ivp_kwargs

    def _rhs_rescaled(self, lam, y):
        """RHS in rescaled coordinates."""
        nf = self.n_fields
        s = self.sbar_func(lam)
        dy = np.zeros(nf)

        y1 = y[0]
        inv_lam = 1.0 / lam

        # dy_1/dlam: from F_111 chi_1^2 = F_111 y_1^2/lam^2, multiplied by lam
        # gives F_111 y_1^2/lam = -y_1^2/(2*lam), plus the y_1/lam from chain rule
        # => dy_1/dlam = y_1/lam - y_1^2/(2*lam) + lam*(rest + sbar_1)
        # = y_1(2 - y_1)/(2*lam) + lam*(rest + sbar_1)
        rest = -2.0 * y[1]**2 - 2.0 * y[2]**2
        if nf == 4:
            rest += 2.0 * y[3]**2

        dy[0] = y1 * (2.0 - y1) * 0.5 * inv_lam + lam * (rest + s[0])

        # dy_a/dlam = -y_1 * y_a / lam + sbar_a  (for a >= 2)
        for a in range(1, nf):
            dy[a] = -y1 * y[a] * inv_lam + s[a]

        return dy

    def _initial_conditions_rescaled(self):
        """IC in rescaled coords: y_1 = y1_init, y_{rest} = 0."""
        y0 = np.zeros(self.n_fields)
        y0[0] = self.y1_init
        return y0

    def solve(self, t_eval=None):
        """Run the integration.

        Parameters
        ----------
        t_eval : ndarray, optional
            Lambda values at which to store the solution.

        Returns
        -------
        SaddlePointResult
        """
        lam0, lam_end = self.lam_span
        y0 = self._initial_conditions_rescaled()

        sol = solve_ivp(
            self._rhs_rescaled,
            (lam0, lam_end),
            y0,
            method=self.method,
            dense_output=self.dense_output,
            t_eval=t_eval,
            **self.ivp_kwargs,
        )

        if not sol.success:
            raise RuntimeError(f"Saddle-point integration failed: {sol.message}")

        # Convert back to chi coordinates for storage
        lam_arr = sol.t
        y_arr = sol.y  # (n_fields, N_times)
        chi_arr = y_arr.copy()
        chi_arr[0] = y_arr[0] / lam_arr  # chi_1 = y_1 / lam

        # Dense output wrapper
        sol_dense = sol.sol if self.dense_output else None

        return SaddlePointResult(lam_arr, chi_arr, sol_dense, self.n_fields)




def solve_singular_theta(s, t_max, epsilon=1e-8, num_points=1000, method="Radau"):
    """
    Solve the singular Riccati-type ODE

        dtheta/dt = -0.5 * theta(t)^2 + s(t),   t in (0, t_max],

    with singular initial behavior

        theta(t) ~ 2 / t   as t -> 0+.

    Method
    ------
    Directly integrating theta is numerically unstable near t = 0 because of the
    singularity. Instead, use the substitution

        theta(t) = 2 u'(t) / u(t),

    which transforms the equation into the linear second-order ODE

        u''(t) = 0.5 * s(t) * u(t).

    To match theta(t) ~ 2/t near the origin, we require

        2 u'(t) / u(t) ~ 2 / t,

    so that u'(t) / u(t) ~ 1 / t, hence u(t) ~ C t. We may take C = 1, giving
    the initial data at a small epsilon > 0:

        u(epsilon)  = epsilon,
        u'(epsilon) = 1.

    Parameters
    ----------
    s : callable
        Source term s(t). Should accept a scalar float t and return a scalar.
        Vectorized callables are also supported.
    t_max : float
        Final integration time. Must satisfy t_max > epsilon > 0.
    epsilon : float, optional
        Small positive starting time used to avoid the singular point t = 0.
        Default is 1e-6.
    num_points : int, optional
        Number of output sample points in [epsilon, t_max]. Default is 1000.
    method : str, optional
        Integration method passed to scipy.integrate.solve_ivp. Default is "DOP853".

    Returns
    -------
    t : ndarray, shape (num_points,)
        Time samples in [epsilon, t_max].
    theta : ndarray, shape (num_points,)
        Reconstructed solution theta(t) = 2 u'(t) / u(t).

    Raises
    ------
    ValueError
        If input parameters are invalid.
    RuntimeError
        If the ODE solver fails.

    Notes
    -----
    Numerical stability measures:
    - Integration starts at t = epsilon, not at t = 0.
    - theta is reconstructed only where |u(t)| is safely above a small threshold.
    - Near any accidental zero crossing of u, theta is set to NaN to avoid
      division-by-zero or catastrophic blow-up.

    Examples
    --------
    Solve with s(t) = 0, for which the exact singular solution is theta(t) = 2/t:

    >>> t, theta = solve_singular_theta(lambda t: 0.0, t_max=1.0)

    Solve with constant forcing s(t) = 1:

    >>> t, theta = solve_singular_theta(lambda t: 1.0, t_max=1.0)
    """
    if not callable(s):
        raise ValueError("s must be a callable.")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive.")
    if t_max <= epsilon:
        raise ValueError("t_max must be greater than epsilon.")
    if num_points < 2:
        raise ValueError("num_points must be at least 2.")

    t_eval = np.linspace(epsilon, t_max, num_points)

    def _s_scalar(t):
        """
        Evaluate s(t) robustly for scalar t.

        Supports both scalar-only and vectorized user callables.
        """
        val = s(t)
        arr = np.asarray(val)
        if arr.ndim == 0:
            return float(arr)
        return float(arr.reshape(-1)[0])

    def rhs(t, y):
        """
        First-order system for y = [u, u']:

            u'  = v
            v'  = 0.5 * s(t) * u
        """
        u, up = y
        return np.array([up, 0.5 * _s_scalar(t) * u], dtype=float)

    # Initial data chosen so that u(t) ~ t near t = 0+,
    # which gives theta(t) = 2 u'(t)/u(t) ~ 2/t.
    u0 = float(epsilon)
    up0 = 1.0

    sol = solve_ivp(
        rhs,
        (epsilon, t_max),
        y0=np.array([u0, up0], dtype=float),
        t_eval=t_eval,
        method=method,
        rtol=1e-9,
        atol=1e-12,
    )

    if not sol.success:
        raise RuntimeError(f"ODE solve failed: {sol.message}")

    t = sol.t
    u = sol.y[0]
    up = sol.y[1]

    # Avoid division-by-zero: if u gets too small, mark theta as NaN.
    scale = max(1.0, np.max(np.abs(u)))
    u_floor = 1e-14 * scale

    theta = np.full_like(u, np.nan, dtype=float)
    safe = np.abs(u) > u_floor
    theta[safe] = 2.0 * up[safe] / u[safe]

    return t, theta
