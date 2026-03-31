"""sachsfield: Scalar field evolution for Sachs optical equations."""

from .coefficients import quadratic, M_matrix
from .saddle import SaddlePointSolver, SaddlePointResult
from .fluctuations import FluctuationSolver, FluctuationResult
from .sources import FullSkySource, FlatSkySource
from .solver import SachsFieldSolver, FullResult
from .utils import (
    animate_evolution,
    pixel_pdf,
    angular_power_spectrum,
    plot_power_spectra,
    summary_statistics,
    evolution_summary,
)

__all__ = [
    "quadratic",
    "M_matrix",
    "SaddlePointSolver",
    "SaddlePointResult",
    "FluctuationSolver",
    "FluctuationResult",
    "FullSkySource",
    "FlatSkySource",
    "SachsFieldSolver",
    "FullResult",
    "animate_evolution",
    "pixel_pdf",
    "angular_power_spectrum",
    "plot_power_spectra",
    "summary_statistics",
    "evolution_summary",
]
