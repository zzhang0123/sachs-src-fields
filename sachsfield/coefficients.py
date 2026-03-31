"""
Coupling coefficients for the Sachs scalar field system.

The dynamic equation is:  x_dot_a = F_{abc} x_b x_c + s_a
where a,b,c in {1,2,3,4} (internally 0-indexed as {0,1,2,3}).

Nonzero F entries (1-indexed):
    F_111 = -1/2,  F_122 = -2,  F_133 = -2,  F_144 = 2
    F_212 = -1,    F_313 = -1,  F_414 = -1
"""

import numpy as np


def quadratic(x, n_fields=4):
    """Compute F_{abc} x_b x_c for all a.

    Parameters
    ----------
    x : ndarray, shape (n_fields,) or (n_fields, npix)
    n_fields : int
        Number of fields (3 or 4).

    Returns
    -------
    ndarray, same shape as x
    """
    x1, x2, x3 = x[0], x[1], x[2]
    if n_fields == 4:
        x4 = x[3]
        return np.stack([
            -0.5 * x1 * x1 - 2.0 * x2 * x2 - 2.0 * x3 * x3 + 2.0 * x4 * x4,
            -x1 * x2,
            -x1 * x3,
            -x1 * x4,
        ])
    else:
        return np.stack([
            -0.5 * x1 * x1 - 2.0 * x2 * x2 - 2.0 * x3 * x3,
            -x1 * x2,
            -x1 * x3,
        ])


def M_matrix(chi, n_fields=4):
    """Linearization matrix M_{ab} = (F_{abc} + F_{acb}) chi_c.

    Parameters
    ----------
    chi : ndarray, shape (n_fields,)
    n_fields : int
        Number of fields (3 or 4).

    Returns
    -------
    ndarray, shape (n_fields, n_fields)
    """
    c1, c2, c3 = chi[0], chi[1], chi[2]
    if n_fields == 4:
        c4 = chi[3]
        return np.array([
            [-c1, -4.0 * c2, -4.0 * c3, 4.0 * c4],
            [-c2, -c1,        0.0,       0.0],
            [-c3,  0.0,      -c1,        0.0],
            [-c4,  0.0,       0.0,      -c1],
        ])
    else:
        return np.array([
            [-c1, -4.0 * c2, -4.0 * c3],
            [-c2, -c1,        0.0],
            [-c3,  0.0,      -c1],
        ])
