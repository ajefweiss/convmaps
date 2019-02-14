# -*- coding: utf-8 -*-
"""logging.py

"""

import logging
import sys


def configure_logger(debug=False, logfile=None, verbose=False):
    """
    Configures the built in python logger.

    Args:
        debug: set logging level to DEBUG
        logfile: log to file
        verbose: set logging level to INFO
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    stream = logging.StreamHandler(sys.stdout)
    if debug and verbose:
        stream.setLevel(logging.DEBUG)
    elif verbose:
        stream.setLevel(logging.INFO)
    else:
        stream.setLevel(logging.WARNING)

    stream.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    root.addHandler(stream)

    if logfile:
        file = logging.FileHandler(logfile, "a")
        if debug:
            file.setLevel(logging.DEBUG)
        else:
            file.setLevel(logging.INFO)

        file.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root.addHandler(file)
