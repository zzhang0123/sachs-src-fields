"""
sachsray -- a numerical full-sky Sachs ray-tracer.

Given realisations of the Ricci-focusing (Phi00) and Weyl-shear (Psi0) driving
fields along the line of sight, integrate the Sachs optical evolution per ray
and read off weak-lensing observables (kappa, gamma, rotation).

The core integrates the *linear Jacobi* form (caustic-robust, no vertex
rescaling) on a JAX/diffrax backend with vmap-over-rays and chunked streaming
for full-sky maps. Driving fields are supplied as given realisations (Gaussian
or non-Gaussian) and fed in as smooth cubic control paths.

See prototype/REDESIGN.md for the design rationale.
"""

from . import physics, solvers, raytrace, fields
from .physics import (
    riccati_rhs,
    tidal_matrix,
    deformation_rate,
    scalars_from_deformation,
    observables_from_jacobi,
)
from .solvers import (
    cubic_control,
    solve_jacobi_scalar,
    solve_jacobi_matrix,
    solve_riccati,
)
from .raytrace import DrivingField, background_distance, trace_rays, trace_rays_streaming
from .fields import driving_from_components, driving_from_source

__all__ = [
    "physics",
    "solvers",
    "raytrace",
    "fields",
    "riccati_rhs",
    "tidal_matrix",
    "deformation_rate",
    "scalars_from_deformation",
    "observables_from_jacobi",
    "cubic_control",
    "solve_jacobi_scalar",
    "solve_jacobi_matrix",
    "solve_riccati",
    "DrivingField",
    "background_distance",
    "trace_rays",
    "trace_rays_streaming",
    "driving_from_components",
    "driving_from_source",
]
