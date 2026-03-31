"""
High-level solver combining saddle point + fluctuations.
"""

import numpy as np

from .saddle import SaddlePointSolver
from .fluctuations import FluctuationSolver


class FullResult:
    """Combined result of saddle point + fluctuation evolution.

    Attributes
    ----------
    saddle : SaddlePointResult
    fluctuation : FluctuationResult
    n_fields : int
    """

    def __init__(self, saddle, fluctuation, n_fields):
        self.saddle = saddle
        self.fluctuation = fluctuation
        self.n_fields = n_fields

    def total_field(self, lam_index):
        """Total field chi + xi at a given output time index.

        Parameters
        ----------
        lam_index : int
            Index into fluctuation.lam.

        Returns
        -------
        ndarray, shape (n_fields, npix)
        """
        lam = self.fluctuation.lam[lam_index]
        chi = self.saddle(lam)  # (n_fields,)
        xi = self.fluctuation.xi[lam_index]  # (n_fields, npix)
        return chi[:, None] + xi


class SachsFieldSolver:
    """High-level solver for the full Sachs scalar field system.

    Parameters
    ----------
    source : FullSkySource or FlatSkySource
        Source object (must have been generated or will auto-generate).
    lam_span : (float, float)
        (lam_start, lam_end). lam_start near 0 (negative), lam_end more negative.
    sbar_func : callable(lam) -> ndarray (n_fields,), optional
        Override for sky-averaged source. If None, uses source.sbar_interpolator().
    n_fields : int
        Number of fields (3 or 4). Default 3.
    linear_only : bool
        If True, drop nonlinear term in fluctuation equation. Default False.
    saddle_kwargs : dict, optional
        Extra kwargs for SaddlePointSolver (method, rtol, atol, etc.).
    fluct_kwargs : dict, optional
        Extra kwargs for FluctuationSolver (method, rtol, atol, max_step, etc.).
    """

    def __init__(self, source, lam_span, sbar_func=None,
                 n_fields=3, linear_only=False,
                 saddle_kwargs=None, fluct_kwargs=None):
        self.source = source
        self.lam_span = lam_span
        self.sbar_func = sbar_func
        self.n_fields = n_fields
        self.linear_only = linear_only
        self.saddle_kwargs = saddle_kwargs or {}
        self.fluct_kwargs = fluct_kwargs or {}

    def solve(self, t_eval=None):
        """Run the full evolution.

        Parameters
        ----------
        t_eval : ndarray, optional
            Lambda values at which to store the fluctuation solution.
            The saddle point uses dense output regardless.

        Returns
        -------
        FullResult
        """
        # 1. Generate sources if needed
        if not self.source._generated:
            self.source.generate()

        # 2. Get source interpolators
        sbar = self.sbar_func or self.source.sbar_interpolator()
        delta_s = self.source.delta_s_interpolator()

        # 3. Solve saddle point
        sp_solver = SaddlePointSolver(
            sbar, self.lam_span,
            n_fields=self.n_fields,
            **self.saddle_kwargs,
        )
        saddle_result = sp_solver.solve()

        # 4. Solve fluctuations
        fl_solver = FluctuationSolver(
            saddle_result, delta_s, self.source.npix,
            n_fields=self.n_fields,
            linear_only=self.linear_only,
            lam_span=self.lam_span,
            **self.fluct_kwargs,
        )
        fluct_result = fl_solver.solve(t_eval=t_eval)

        return FullResult(saddle_result, fluct_result, self.n_fields)



