# -*- coding: utf-8 -*-
"""peaks_euclid.py

"""

import argparse
import gc
import healpy as hp
import logging
import multiprocessing
import numpy as np
import os
import time

from util import configure_logger
from util import ang_dist


def worker_peaks(map, bin_count, logbin_count, bin_range):
    logger = logging.getLogger(__name__)

    factor = 1 - np.sum(map == hp.UNSEEN) / len(map)

    logger.info("processing map ({:.0f} deg^2)".format(4 * factor * np.pi * (180 / np.pi) ** 2))

    pixels = np.where(map != hp.UNSEEN)[0]

    pixel_neighbours = hp.get_all_neighbours(hp.npix2nside(len(map)), pixels)

    peaks = []

    for i in range(0, len(pixels)):
        pixel = pixels[i]
        nb = pixel_neighbours[:, i]

        if all(map[pixel] > map[nb]) and all(map[nb] != hp.UNSEEN):
            peaks.append(map[pixel])

    peaks = np.array(peaks)

    if len(bin_range) == 2:
        bins = np.linspace(bin_range[0], bin_range[1], bin_count + 1)
    elif len(bin_range) == 3:
        bins = np.append(np.linspace(bin_range[0], bin_range[1], bin_count + 1)[:-1],
                         np.logspace(np.log10(bin_range[1]), np.log10(bin_range[2]), logbin_count + 1))
    else:
        raise NotImplementedError

    counts = np.zeros(len(bins) - 1)

    for bin in range(0, len(bins) - 1):
        counts[bin] = np.sum((peaks > bins[bin]) & (peaks < bins[bin + 1]))

    counts = np.append(counts, np.sum((peaks > bins[-1])))
    bins = np.append(bins, [np.inf])

    return bins[:-1], bins[1:], counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # File I/O
    parser.add_argument('-p', '--path', nargs='+', type=str, default=None, metavar="CONVMAPS",
                        help="convmaps")
    parser.add_argument('-o', '--out', nargs='?', type=str, default=None, metavar="FOLDER",
                        help="output folder")
    parser.add_argument('-f', '--force', action='store_true', help="overwrite output (skip otherwise)")

    parser.add_argument('-c', '--combine', action='store_true', help="combine output files (no measurements)")

    # Debugging/Logging
    parser.add_argument('-d', '--debug', action='store_true', help="enable DEBUG output")
    parser.add_argument('-l', '--logfile', nargs='?', default=None, metavar="LOGFILE", help="logfile path")
    parser.add_argument('-v', '--verbose', action='store_true', help="enable INFO output")

    # Output
    parser.add_argument('-b', '--bins', nargs='?', type=int, default=20, help="peak bins")
    parser.add_argument('--logbins', nargs='?', type=int, default=10, help="logarithmic peak bins")
    parser.add_argument('-r', '--bin-range', nargs='+', type=float, default=[0, 0.1],
                        help="peak range, if two ranges are given second range is binned logarithmically")

    # Smoothing/Noise
    parser.add_argument('-n', '--galaxy-density', nargs='?', type=float, default=30, metavar="N/ARCMIN")
    parser.add_argument('-e', '--ellipticity', nargs='?', type=float, default=0.3)
    parser.add_argument('-s', '--seed', nargs='?', type=int, default=42)
    parser.add_argument('-k', '--smoothing', nargs='?', type=float, default=2, metavar="ARCMIN")

    args = parser.parse_args()

    configure_logger(debug=args.debug, logfile=args.logfile, verbose=args.verbose)

    logger = logging.getLogger(__name__)

    if args.combine:
        logger.info("starting {} (combining)".format("peaks_euclid.py"))

        peaks = np.array([np.loadtxt(path) for path in args.path])

        bins_l = peaks[0, :, 0]
        bins_r = peaks[0, :, 1]

        for i in range(1, len(args.path)):
            if list(bins_l) != list(peaks[i, :, 0]) or list(bins_r) != list(peaks[i, :, 1]):
                raise Exception("bin range does not match for all files")

        counts = []

        for i in range(0, len(args.path)):
            for j in range(0, 2):
                counts.append(peaks[i, :, 2 + j])

        counts = np.array(counts)

        mean = np.mean(counts, axis=0)
        error = np.std(counts, axis=0)

        logger.info("combined {0} files".format(len(args.path)))

        np.savetxt(args.out, np.vstack((bins_l, bins_r, mean, error)).T)
    else:
        # create worker pool
        logger.info("starting {}".format("peaks_euclid.py"))
        pool = multiprocessing.Pool(processes=2)

        cl_list = []

        for path in args.path:
            out_path = os.path.join(args.out, "{0}_n{1:.2f}_e{2:.2f}_s{3:.0f}_k{4:.2f}.npy".format(
                os.path.basename(path).split('.')[0], args.galaxy_density, args.ellipticity, args.seed, args.smoothing))

            if not args.force and os.path.isfile(out_path):
                logger.info("skipping map {}".format(path))
                continue
            else:
                logger.info("using map {}".format(path))

            # generate placeholder
            np.save(out_path, np.array([]))

            timer = time.time()

            k_map = np.array(np.load(path), dtype=np.float32)

            # set unique seed for each out file name
            np.random.seed(args.seed * int(sum([ord(c) for c in out_path])))

            logger.debug("generating noise")
            k_map += np.random.normal(0, args.ellipticity / np.sqrt(
                4 * 10800**2 / np.pi * args.galaxy_density / len(k_map)), len(k_map))

            if args.smoothing > 0:
                logger.debug("applying Gaussian beam ({} arcmin)".format(args.smoothing))
                k_map = hp.smoothing(k_map, sigma=args.smoothing * 0.000290888, iter=1, verbose=False)

            logger.debug("preparing maps")

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

            logger.debug("counting peaks")

            result = pool.starmap(worker_peaks,
                                  [(m, args.bins, args.logbins, args.bin_range) for m in [k_map1, k_map2]])

            logger.info("processed {0} ({1:.2f}s)".format(path, time.time() - timer))

            np.savetxt(out_path, np.vstack((result[0][0],
                                            result[0][1],
                                            result[0][2],
                                            result[1][2])).T)

            del k_map, k_map1, k_map2
            gc.collect()
