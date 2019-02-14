# -*- coding: utf-8 -*-
"""convpeaks.py

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


def worker_neighbours(pixel_slice, nside):
    return hp.get_all_neighbours(nside, list(range(pixel_slice.start, pixel_slice.stop)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # File I/O
    parser.add_argument('-p', '--path', nargs='+', type=str, default=None, metavar="CONVMAPS",
                        help="convmap file paths")
    parser.add_argument('-o', '--out', nargs='?', type=str, default=None, metavar="FOLDER",
                        help="output folder")

    parser.add_argument('-f', '--force', action='store_true', help="overwrite output (skip otherwise)")

    # Debugging/Logging
    parser.add_argument('-d', '--debug', action='store_true', help="enable DEBUG output")
    parser.add_argument('-l', '--logfile', nargs='?', default=None, metavar="LOGFILE", help="logfile path")
    parser.add_argument('-v', '--verbose', action='store_true', help="enable INFO output")

    # Output
    parser.add_argument('-b', '--bins', nargs='?', type=int, default=36, help="peak bins")
    parser.add_argument('-r', '--bin-range', nargs=2, type=float, default=[-1, 8], help="peak range (S/N)")
    parser.add_argument('-a', '--accumulate', action='store_true',
                        help="accumulate peaks above the given range in an extra bin")

    # Smoothing/Noise
    parser.add_argument('-n', '--galaxy-density', nargs='?', type=float, default=10, metavar="N/ARCMIN")
    parser.add_argument('-e', '--ellipticity', nargs='?', type=float, default=0.3)
    parser.add_argument('-s', '--seed', nargs='?', type=int, default=42)
    parser.add_argument('-k', '--smoothing', nargs='?', type=float, default=2, metavar="ARCMIN")

    # Parallelization/Replication
    parser.add_argument('--workers', nargs='?', type=int, default=4,
                        help="number of worker processes (should be smaller than the number of batches)")

    args = parser.parse_args()

    configure_logger(debug=args.debug, logfile=args.logfile, verbose=args.verbose)

    logger = logging.getLogger(__name__)

    # create worker pool
    logger.info("starting {}".format("mappeaks.py"))
    pool = multiprocessing.Pool(processes=args.workers)

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

        k_map = np.load(path)

        # set unique seed for each out file name
        np.random.seed(args.seed * int(sum([ord(c) for c in out_path])))

        logger.debug("generating noise")
        k_map += np.random.normal(0, args.ellipticity / np.sqrt(
            4 * (10800) ** 2 / np.pi * args.galaxy_density / len(k_map)), len(k_map))

        if args.smoothing > 0:
            logger.debug("applying Gaussian beam ({} arcmin)".format(args.smoothing))
            k_map = hp.smoothing(k_map, sigma=args.smoothing * 0.000290888, iter=1, verbose=False)

        # convert to S/N
        k_map = k_map * np.sqrt(4 * args.galaxy_density * np.pi * args.smoothing ** 2) / args.ellipticity

        logger.debug("counting peaks")

        slice_size = int(hp.npix2nside(len(k_map)) * hp.npix2nside(len(k_map)) / 2)
        slice_count = int(len(k_map) / slice_size)
        slices = [slice(i * slice_size, (i + 1) * slice_size) for i in range(0, slice_count)]

        result = pool.starmap(worker_neighbours, [(slices[slice_index], hp.npix2nside(len(k_map))) for slice_index in
                                                  range(0, slice_count)])

        neighbours = np.concatenate(list(result), axis=1)

        peaks = []

        for pixel in range(0, len(k_map)):
            nb = neighbours[:, pixel]

            if all(k_map[pixel] > k_map[nb]) and k_map[pixel] != hp.UNSEEN:
                peaks.append(k_map[pixel])

        peaks = np.array(peaks)

        bins = np.linspace(args.bin_range[0], args.bin_range[1], args.bins + 1)
        counts = np.zeros(args.bins)

        for bin in range(0, args.bins):
            counts[bin] = np.sum((peaks > bins[bin]) & (peaks < bins[bin + 1]))

        if args.accumulate:
            counts = np.append(counts, np.sum((peaks > bins[-1])))
            bins = np.append(bins, [np.inf])

        np.savetxt(out_path, np.vstack((bins[:-1], bins[1:], counts)).T)
