"""
Physics conventions for the full-sky Sachs ray-tracer.

Single source of truth for the optical-scalar / Jacobi-map conventions. Pure
functions (jnp only) so they compose inside ``jax.vmap`` / ``diffrax``.

Conventions (STF_lensing draft, sachs_dynamics.tex + appendix.tex)
------------------------------------------------------------------
Physical Sachs optical scalars and driving fields::

    chi = (theta, sigma_plus, sigma_cross)        # expansion + 2 shear pols
    s   = (Phi00,  W1,         W2)                 # Ricci focusing + Re/Im Psi0

Riccati (Sachs) system, twist-free (omega = 0)::

    dtheta/dlam     = -theta^2 - sp^2 - sc^2 + Phi00
    dsigma_plus/dlam = -2 theta sp + W1
    dsigma_cross/dlam= -2 theta sc + W2

NOTE on the existing package: ``sachsfield/coefficients.py`` uses
F_111 = -1/2 (etc.), i.e. it integrates x1 = 2*theta. That is a field
rescaling of the same physics; here we use the draft's published convention
(x1 = theta, F_111 = -1) so that theta -> 1/lam at the vertex and the Jacobi
amplitude D -> lam, with no factor-of-two source trap.

Jacobi (linear) form -- the robust object::

    Jddot(lam) = T(lam) @ J(lam),     J(0)=0, Jdot(0)=I

with the 2x2 optical tidal matrix whose ISOTROPIC part is Phi00 (so the
shear-free trace reduces to the draft's ``Ddot = Phi00 D``) and whose
trace-free part is the Weyl shear (W1, W2). The relative SIGN of the Weyl
coupling is pinned by the Riccati<->Jacobi cross-validation test
(tests/test_sachsray.py::test_jacobi_matches_riccati).
"""

from __future__ import annotations

import jax.numpy as jnp


# Draft Table tab:Fabc (physical theta-convention). Kept for reference / the
# Riccati RHS below encodes the same content explicitly.
F_NONZERO = {
    (0, 0, 0): -1.0,  # F_111  theta theta
    (0, 1, 1): -1.0,  # F_122  sp sp
    (0, 2, 2): -1.0,  # F_133  sc sc
    (1, 0, 1): -2.0,  # F_212  theta sp
    (2, 0, 2): -2.0,  # F_313  theta sc
}


def riccati_rhs(chi: jnp.ndarray, s: jnp.ndarray) -> jnp.ndarray:
    """RHS of the physical Sachs (Riccati) system.

    Parameters
    ----------
    chi : (3,) array (theta, sigma_plus, sigma_cross).
    s   : (3,) array (Phi00, W1, W2).

    Returns
    -------
    (3,) array dchi/dlam.
    """
    theta, sp, sc = chi[0], chi[1], chi[2]
    phi00, w1, w2 = s[0], s[1], s[2]
    return jnp.stack([
        -theta * theta - sp * sp - sc * sc + phi00,
        -2.0 * theta * sp + w1,
        -2.0 * theta * sc + w2,
    ])


def tidal_matrix(phi00: jnp.ndarray, w1: jnp.ndarray, w2: jnp.ndarray) -> jnp.ndarray:
    """2x2 optical tidal matrix T for the Jacobi equation ``Jddot = T J``.

    Isotropic part = Phi00 (reduces to ``Ddot = Phi00 D`` when W1=W2=0);
    trace-free part carries the Weyl shear. The Weyl sign convention here is
    pinned by the Riccati<->Jacobi cross-validation test.
    """
    return jnp.array(
        [[phi00 + w1, w2], [w2, phi00 - w1]],
        dtype=jnp.result_type(phi00, w1, w2),
    )


def deformation_rate(J: jnp.ndarray, Jdot: jnp.ndarray) -> jnp.ndarray:
    """S = Jdot @ inv(J): the 2x2 optical deformation-rate matrix."""
    return Jdot @ jnp.linalg.inv(J)


def scalars_from_deformation(S: jnp.ndarray) -> dict:
    """Extract optical scalars from S = Jdot inv(J).

    theta      = 1/2 tr(S)
    sigma_plus = 1/2 (S00 - S11)   (trace-free symmetric)
    sigma_cross= 1/2 (S01 + S10)
    omega      = 1/2 (S01 - S10)   (antisymmetric -> image rotation)
    """
    theta = 0.5 * (S[0, 0] + S[1, 1])
    sp = 0.5 * (S[0, 0] - S[1, 1])
    sc = 0.5 * (S[0, 1] + S[1, 0])
    om = 0.5 * (S[0, 1] - S[1, 0])
    return {"theta": theta, "sigma_plus": sp, "sigma_cross": sc, "omega": om}


def observables_from_jacobi(J: jnp.ndarray, D_bg: jnp.ndarray) -> dict:
    """Weak-lensing observables from the Jacobi map J and background D_bg.

    Distortion matrix A = J / D_bg (background A = I). Standard decomposition
    (Bartelmann & Schneider)::

        kappa  = 1 - 1/2 tr(A)
        gamma1 = -1/2 (A00 - A11)
        gamma2 = -1/2 (A01 + A10)
        omega  =  1/2 (A01 - A10)

    At first order this matches the draft's Eq. (convergence/shear),
    kappa = -1/2 dJ^A_A with dJ = I - A.
    """
    A = J / D_bg
    kappa = 1.0 - 0.5 * (A[0, 0] + A[1, 1])
    gamma1 = -0.5 * (A[0, 0] - A[1, 1])
    gamma2 = -0.5 * (A[0, 1] + A[1, 0])
    omega = 0.5 * (A[0, 1] - A[1, 0])
    return {"kappa": kappa, "gamma1": gamma1, "gamma2": gamma2, "omega": omega}
