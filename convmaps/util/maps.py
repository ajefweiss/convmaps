# -*- coding: utf-8 -*-
"""maps.py

"""

import numpy as np


def ang_dist(theta, phi, theta2=0, phi2=0):
    """
    Calculates angular distance between to points (RA/DEC system)

    Args:
        theta: target declination
        phi: target right-ascension
        theta2: base declination
        phi2: base right-ascension

    Returns: angle (radians)

    """
    if isinstance(theta, np.ndarray) and isinstance(phi, np.ndarray):
        if isinstance(float(theta2), float):
            theta2 = theta2 * np.ones_like(theta)

        if isinstance(float(phi2), float):
            phi2 = phi2 * np.ones_like(phi)

    return np.arccos(np.sin(theta) * np.sin(theta2) + np.cos(theta) * np.cos(theta2) * np.cos(phi - phi2))
