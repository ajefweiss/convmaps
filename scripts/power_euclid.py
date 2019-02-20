# -*- coding: utf-8 -*-
"""power_euclid.py

"""

import argparse
import gc
import healpy as hp
import logging
import multiprocessing
import numpy as np
import time

from util import configure_logger
from util import ang_dist


def worker_power(map, lmax):
    logger = logging.getLogger(__name__)

    factor = 1 - np.sum(map == hp.UNSEEN) / len(map)

    logger.info("processing map ({:.0f} deg^2)".format(4*factor*np.pi*(180/np.pi)**2))

    return hp.anafast(map, lmax=lmax) / hp.pixwin(hp.npix2nside(len(map)))[:lmax + 1] ** 2 / factor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # File I/O
    parser.add_argument('-p', '--path', nargs='+', type=str, default=None, metavar="CONVMAPS",
                        help="convmaps")
    parser.add_argument('-o', '--out', nargs='?', type=str, default=None, metavar="FOLDER",
                        help="output folder")

    # Debugging/Logging
    parser.add_argument('-d', '--debug', action='store_true', help="enable DEBUG output")
    parser.add_argument('-l', '--logfile', nargs='?', default=None, metavar="LOGFILE", help="logfile path")
    parser.add_argument('-v', '--verbose', action='store_true', help="enable INFO output")

    # Power
    parser.add_argument('--lmax', nargs='?', type=int, default=3071)

    args = parser.parse_args()

    configure_logger(debug=args.debug, logfile=args.logfile, verbose=args.verbose)

    logger = logging.getLogger(__name__)

    # create worker pool
    logger.info("starting {}".format("power_euclid.py"))
    pool = multiprocessing.Pool(processes=2)

    cl_list = []

    for path in args.path:
        logger.info("using map {}".format(path))

        timer = time.time()

        k_map = np.array(np.load(path), dtype=np.float32)

        k_map1 = k_map
        k_map2 = np.array(k_map)

        # cut out 2 surveys
        angle = np.arccos(1 - 10000 * np.pi / 180**2)

        thetas, phis = hp.pix2ang(hp.npix2nside(len(k_map1)), range(0, len(k_map1)))
        thetas = np.pi / 2 - thetas

        k_map1[np.where(ang_dist(thetas, phis) > angle)] = hp.UNSEEN
        k_map2[np.where(ang_dist(thetas, phis, phi2=np.pi) > angle)] = hp.UNSEEN

        del thetas, phis
        gc.collect()

        result = pool.starmap(worker_power, [(m, args.lmax) for m in [k_map1, k_map2]])

        cl_list.extend(list(result))

        logger.info("processed {0} ({1:.2f}s)".format(path, time.time() - timer))

        del k_map, k_map1, k_map2
        gc.collect()

    ls = list(range(0, args.lmax + 1))
    cls = np.mean(cl_list, axis=0)
    err = np.std(cl_list, axis=0)

    np.savetxt(args.out, np.vstack((ls, cls, err)).T)
