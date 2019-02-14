# -*- coding: utf-8 -*-
"""cosmology.py

"""

import numpy as np
import scipy.constants
import scipy.integrate


def d_c(z1, z2, omega_m, omega_l):
    """
    Calculates the comoving distance [Mpc/h].

    Args:
        z1: start redshift
        z2: end redshift
        omega_m: dark matter density parameter
        omega_l: dark energy density parameter

    Returns: comoving distance [Mpc/h]

    """
    return 1e-5 * scipy.constants.c * \
           scipy.integrate.quad(lambda x: 1 / np.sqrt(omega_m * (1 + x) ** 3 + omega_l), z1, z2)[0]


def E(z, omega_m, omega_l):
    """
    Calculates the ratio in between Hubble parameter at a given redshift and at the current time.

    Args:
        z: redshift
        omega_m: dark matter density parameter
        omega_l: dark energy density parameter

    Returns: E(z) = H(z)/H_0

    """
    return 1 / np.sqrt(omega_m * (1 + z) ** 3 + omega_l)


def n_s(source_type, source_parameters):
    """
    Returns a function that describes the selected source distribution with the given parameters.

    Args:
        source_type: distribution type (sheet, parametrized)
        source_parameters: distribution parameters

    Returns: distribution function

    """
    if source_type == "sheet":
        if not isinstance(float(source_parameters), float):
            params = float(source_parameters[0])

        return lambda z: np.exp(-(z - params) ** 2 * 1e4)
    elif source_type == "parametrized":
        return lambda z: z ** source_parameters[0] * np.exp(-(z / source_parameters[1]) ** source_parameters[2])
    else:
        raise NotImplementedError


def w_b(z1, z2, l1, l2, source_type, source_parameters, omega_m, omega_l):
    """
    Calculates the shell weight of a single shell spanning redshifts z1 to z2 (lightcone ranging from l1 to l2).
    See Teyssier et al. 2009 for an example and implementation.

    Args:
        z1: start redshift
        z2: end redshift
        source_type: distribution type (sheet, parametrized)
        source_parameters: distribution parameters
        omega_m: dark matter density parameter
        omega_l: dark energy density parameter

    Returns: shell weight

    """

    ns = n_s(source_type=source_type, source_parameters=source_parameters)

    f = lambda z: scipy.integrate.quad(
        lambda x: ns(x) * d_c(0, z, omega_m, omega_l) * d_c(z, x, omega_m, omega_l) / d_c(0, x, omega_m, omega_l)
                  * (1 + z) / E(z, omega_m, omega_l), z, l2)[0]
    g = lambda z: scipy.integrate.quad(lambda x: ns(x) / E(z, omega_m, omega_l), l1, l2)[0]

    return scipy.integrate.quad(f, z1, z2)[0] / scipy.integrate.quad(g, z1, z2)[0]
