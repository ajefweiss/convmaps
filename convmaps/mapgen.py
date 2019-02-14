# -*- coding: utf-8 -*-
"""mapgen.py

"""

import argparse
import gc
import healpy as hp
import logging
import multiprocessing
import numpy as np
import os
import re
import scipy.constants
import scipy.integrate
import time

from util import d_c, w_b
from util import configure_logger
from util import TipsySnapshot
from util import worker_fullsky, worker_smallsky


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # File I/O
    parser.add_argument('-p', '--path', nargs='?', type=str, default=None, metavar="FOLDER",
                        help="simulation file path (folder or single file)")
    parser.add_argument('--path-filter', nargs='?', type=str, default=".*", metavar="REGEX",
                        help="regex filter")
    parser.add_argument('-o', '--out', nargs='?', type=str, default=None, metavar="FILENAME",
                        help="output path")

    # Debugging/Logging
    parser.add_argument('-d', '--debug', action='store_true', help="enable DEBUG output")
    parser.add_argument('-l', '--logfile', nargs='?', default=None, metavar="LOGFILE", help="logfile path")
    parser.add_argument('-v', '--verbose', action='store_true', help="enable INFO output")

    # Cosmology
    parser.add_argument('--sim-omegam', nargs='?', type=float, default=0.32, metavar="OMEGA_M",
                        help="dark matter density parameter (default: %(default)s)")
    parser.add_argument('--sim-omegal', nargs='?', type=float, default=0.68, metavar="OMEGA_L",
                        help="dark energy density parameter (default: %(default)s)")

    # Simulation
    parser.add_argument('--sim-format', nargs='?', type=str, const="tipsy", default="tipsy", choices=['tipsy'],
                        help='snapshot format: [tipsy] (default: %(default)s)')
    parser.add_argument('--sim-boxsize', nargs='?', type=float, default=1024, metavar="LENGTH (Mpc/h)",
                        help="simulation box size (default: %(default)s Mpc/h)")
    parser.add_argument('--sim-particles', nargs='?', type=int, default=1024, metavar="PARTICLES",
                        help="simulation particle count (default: %(default)s)")

    # Lightcone
    parser.add_argument('--lc-mode', nargs='?', type=str, const="simple", default="simple",
                        choices=['fullsky', 'smallsky', 'simple'],
                        help='lightcone mode: [fullsky, smallsky, simple] (default: %(default)s)')
    parser.add_argument('--lc-range', nargs=2, type=float, default=[0.1, 1.5], metavar="REDSHIFT",
                        help="lightcone redshift range (default: %(default)s)")
    parser.add_argument('--lc-shell-range', nargs=2, type=float, default=None, metavar="REDSHIFT",
                        help="lightcone shell redshift range (only required for single files)")
    parser.add_argument('--lc-source-type', nargs='?', type=str, const="sheet", default="sheet",
                        choices=['sheet', 'parametrized'],
                        help='lightcone source distribution: [sheet, parametrized] (default: %(default)s)')
    parser.add_argument('--lc-source-parameters', nargs='+', type=float, default=[1.5], metavar="PARAMETERS",
                        help="lightcone source distribution parameters (default: %(default)s)")

    # Parallelization/Replication
    parser.add_argument('--batches', nargs='?', type=int, default=5000,
                        help="particle batch count (higher number uses less memory)")
    parser.add_argument('--mpi', action='store_true', help="enable mpi parallelization using mpi4py")
    parser.add_argument('--randomize', action='store_true',
                        help="randomize replicated boxes (offset, flipping, rotation)")
    parser.add_argument('--seed', nargs='?', type=int, default=42, help="seed for RNG used for replication")
    parser.add_argument('--workers', nargs='?', type=int, default=4,
                        help="number of worker processes (should be smaller than the number of batches)")

    # HEALPix
    parser.add_argument('--hp-nside', nargs='?', type=int, default=4096, metavar="NSIDE",
                        help="HEALPix resolution for the generated convergence map")

    args = parser.parse_args()

    configure_logger(debug=args.debug, logfile=args.logfile, verbose=args.verbose)

    logger = logging.getLogger(__name__)

    # create worker pool, use mpi if mpi flag set (this requires the mpi4py package)
    if args.mpi:
        from mpi4py.futures import MPIPoolExecutor

        logger.info("starting {} (mpi enabled)".format("mapgen.py"))

        pool = MPIPoolExecutor(max_workers=args.workers)
    else:
        logger.info("starting {}".format("mapgen.py"))

        pool = multiprocessing.Pool(processes=args.workers)

    if args.seed < 0 or not args.randomize:
        logger.info("randomization disabled")

        # disable randomization if seed is negative
        args.randomize = False
    elif args.randomize:
        logger.info("randomization enabled")

    # empty convergence map
    k_map = np.zeros(hp.nside2npix(args.hp_nside), dtype=np.float32)

    if os.path.isfile(args.path):
        logger.info("using snapshot {}".format(args.path))

        if args.sim_format == "tipsy":
            snapshot = TipsySnapshot(args.path)
        else:
            raise NotImplementedError

        timer = time.time()

        if args.lc_mode == 'fullsky':
            pixels = pool.starmap(worker_fullsky, [
                (snapshot, args.lc_shell_range[0], args.lc_shell_range[1], batch_index, args.batches, args.sim_omegam,
                 args.sim_omegal, args.sim_boxsize, args.hp_nside, args.randomize, args.seed) for batch_index in
                range(0, args.batches)])
        elif args.lc_mode == 'smallsky':
            pixels = pool.starmap(worker_smallsky, [
                (snapshot, args.lc_shell_range[0], args.lc_shell_range[1], batch_index, args.batches, args.sim_omegam,
                 args.sim_omegal, args.sim_boxsize, args.hp_nside, args.randomize, args.seed) for batch_index in
                range(0, args.batches)])
        elif args.lc_mode == 'simple':
            raise NotImplementedError
        else:
            raise NotImplementedError

        pixels = np.concatenate(list(pixels), axis=0)

        counts = np.bincount(pixels, minlength=hp.nside2npix(args.hp_nside))

        shell_distance = d_c(0, (args.lc_shell_range[0] + args.lc_shell_range[1]) / 2, args.sim_omegam, args.sim_omegal)

        shell_weight = w_b(args.lc_shell_range[0], args.lc_shell_range[1], args.lc_range[0], args.lc_range[1],
                  args.lc_source_type, args.lc_source_parameters, args.sim_omegam, args.sim_omegal)

        k_map += 1.5e10 * args.sim_omegam * shell_weight / scipy.constants.c ** 2 * hp.nside2npix(args.hp_nside) \
              / 4 / np.pi * args.sim_boxsize ** 3 / args.sim_particles ** 3 / shell_distance ** 2 * counts
        k_map -= 1.5e10 * args.sim_omegam * shell_weight / scipy.constants.c ** 2 \
              * d_c(args.lc_shell_range[0], args.lc_shell_range[1], args.sim_omegam, args.sim_omegal)

        del pixels, counts
        gc.collect()

        logger.info(
            "processed snapshot z = {0:.3f} - {1:.3f} ({2:.2f}s)".format(args.lc_shell_range[0], args.lc_shell_range[1],
                                                                         time.time() - timer))
    else:
        # load all snapshot headers and sort by redshift
        files = []
        regex = re.compile(args.path_filter)

        for entry in [os.path.basename(entry) for entry in os.listdir(args.path)]:
            if not os.path.isdir(entry) and regex.match(entry):
                files.append(os.path.join(args.path, entry))

        if args.sim_format == "tipsy":
            snapshots = [TipsySnapshot(file) for file in files]
        else:
            raise NotImplementedError

        snapshots.sort(key=lambda x: x.z, reverse=True)

        # generate redshift bins
        snapshot_bins = [snapshots[0].z]
        snapshot_bins.extend([(snapshots[i - 1].z + snapshots[i].z) / 2 for i in range(1, len(snapshots))])
        snapshot_bins.append(0)

        # remove high redshift snapshots
        while snapshot_bins[0] > args.lc_range[1]:
            if snapshot_bins[1] > args.lc_range[1]:
                snapshot_bins = snapshot_bins[1:]
                snapshots = snapshots[1:]
            else:
                snapshot_bins[0] = args.lc_range[1]

        # remove low redshift snapshots
        while snapshot_bins[-1] < args.lc_range[0]:
            if snapshot_bins[-2] < args.lc_range[0]:
                snapshot_bins = snapshot_bins[:-1]
                snapshots = snapshots[:-1]
            else:
                snapshot_bins[-1] = args.lc_range[0]

        logger.debug(
            "generated redshift bins z = {0:.2f} - {1:.2f} ({2} snapshots)".format(snapshot_bins[-1], snapshot_bins[0],
                                                                                   len(snapshots)))

        for index in range(0, len(snapshots)):
            timer = time.time()

            if args.lc_mode == 'fullsky':
                pixels = pool.starmap(worker_fullsky, [
                    (snapshots[index], snapshot_bins[index], snapshot_bins[index + 1], batch_index, args.batches,
                     args.sim_omegam,
                     args.sim_omegal, args.sim_boxsize, args.hp_nside, args.randomize, args.seed) for batch_index in
                    range(0, args.batches)])
            elif args.lc_mode == 'smallsky':
                pixels = pool.starmap(worker_smallsky, [
                    (snapshots[index], snapshot_bins[index], snapshot_bins[index + 1], batch_index, args.batches,
                     args.sim_omegam,
                     args.sim_omegal, args.sim_boxsize, args.hp_nside, args.randomize, args.seed) for batch_index in
                    range(0, args.batches)])
            elif args.lc_mode == 'simple':
                raise NotImplementedError
            else:
                raise NotImplementedError

            pixels = np.concatenate(list(pixels), axis=0)

            counts = np.bincount(pixels, minlength=hp.nside2npix(args.hp_nside))

            shell_distance = d_c(0, (snapshot_bins[index] + snapshot_bins[index + 1]) / 2, args.sim_omegam,
                                 args.sim_omegal)

            shell_weight = w_b(snapshot_bins[index], snapshot_bins[index + 1], args.lc_range[0], args.lc_range[1],
                               args.lc_source_type, args.lc_source_parameters, args.sim_omegam, args.sim_omegal)

            k_map += 1.5e10 * args.sim_omegam * shell_weight / scipy.constants.c ** 2 * hp.nside2npix(args.hp_nside) \
                     / 4 / np.pi * args.sim_boxsize ** 3 / args.sim_particles ** 3 / shell_distance ** 2 * counts
            k_map -= 1.5e10 * args.sim_omegam * shell_weight / scipy.constants.c ** 2 \
                     * d_c(snapshot_bins[index], snapshot_bins[index + 1], args.sim_omegam, args.sim_omegal)

            del pixels, counts
            gc.collect()

            logger.info("processed snapshot z = {0:.3f} - {1:.3f} ({2:.2f}s)".format(snapshot_bins[index],
                                                                                     snapshot_bins[index + 1],
                                                                                     time.time() - timer))

    # mask map
    if args.lc_mode == "smallsky":
        angle = np.arcsin(args.sim_boxsize/ 2  / d_c(0, args.lc_range[1], args.sim_omegam, args.sim_omegal))

        logger.info("masking pixels out of lightcone ({} deg)".format(180 * angle / np.pi))

        thetas, phis = hp.pix2ang(hp.npix2nside(len(k_map)), range(0, len(k_map)))
        thetas = np.pi / 2 - thetas
        phis[np.where(phis > np.pi)] = 2 * np.pi - phis[np.where(phis > np.pi)]

        k_map[np.where(np.sqrt(thetas ** 2 + phis ** 2) > angle)] = hp.UNSEEN

    np.save(args.out, k_map)

    logger.info("generated {0} successfully".format(args.out))
