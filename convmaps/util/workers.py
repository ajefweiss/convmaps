# -*- coding: utf-8 -*-
"""workers.py

"""

import healpy as hp
import numpy as np

from . import d_c


def worker_fullsky(snapshot, z1, z2, batch_index, batches, omega_m, omega_l, boxsize, nside, randomize, seed):
    """
    Loads partial batch from snapshot, replicates (too fullsky), randomizes if necessary and returns the projected
    pixel positions of the particles within the given shell.

    Args:
        snapshot: Snapshot instance
        z1: inner shell redshift
        z2: outer shell redshift
        batch_index: batch index
        batches: total number of batches
        omega_m: dark matter density parameter
        omega_l: dark energy density parameter
        boxsize: snapshot boxsize
        nside: HEALPix NSIDE
        randomize: randomization flag
        seed: randomization seed

    Returns: list of particle positions within the given shell on a HEALPix map

    """
    particles = snapshot.batch_load(batch_index, batches)

    shell_min = d_c(0, z1, omega_m, omega_l) / boxsize
    shell_max = d_c(0, z2, omega_m, omega_l) / boxsize

    replications = int(np.ceil(shell_max))
    replications_boxes = (2 * replications) ** 3

    # randomization parameters
    if randomize:
        np.random.seed(seed)
        rand_f = [np.random.randint(2, size=3) for _ in range(0, replications_boxes)]
        rand_r = [np.random.randint(4, size=2) * np.pi / 2 for _ in range(0, replications_boxes)]
        rand_t = [np.random.rand(3) for _ in range(0, replications_boxes)]

    # replicated particle array
    _particles = np.zeros((replications_boxes * len(particles), 3), dtype=np.float32)

    box_index = 0
    for i in range(-replications, replications):
        for j in range(-replications, replications):
            for k in range(-replications, replications):
                box_slice = slice(box_index * len(particles), (box_index + 1) * len(particles))

                _particles[box_slice] = particles

                # randomize
                if randomize:
                    # transformation matrices for flipping/rotating
                    mat_f = np.matrix([
                        [1 - 2 * rand_f[box_index][0], 0, 0],
                        [0, 1 - 2 * rand_f[box_index][1], 0],
                        [0, 0, 1 - 2 * rand_f[box_index][2]]
                    ])

                    mat_rx = np.matrix([
                        [1, 0, 0],
                        [0, np.cos(rand_r[box_index][0]), -np.sin(rand_r[box_index][0])],
                        [0, np.sin(rand_r[box_index][0]), np.cos(rand_r[box_index][0])]
                    ])

                    mat_ry = np.matrix([
                        [np.cos(rand_r[box_index][1]), 0, np.sin(rand_r[box_index][1])],
                        [0, 1, 0],
                        [-np.sin(rand_r[box_index][1]), 0, np.cos(rand_r[box_index][1])]
                    ])

                    transform = np.dot(np.dot(mat_f, mat_rx), mat_ry)

                    _particles[box_slice] = np.dot(_particles[box_slice], transform)

                    _particles[box_slice] += rand_t[box_index]

                    # wrap particles into their boundaries
                    for v in range(0, 3):
                        _particles[box_slice][:, v][np.where(_particles[box_slice][:, v] < -0.5)] += 1
                        _particles[box_slice][:, v][np.where(_particles[box_slice][:, v] > 0.5)] -= 1

                # offset box position
                _particles[box_slice][:, 0] += i
                _particles[box_slice][:, 1] += j
                _particles[box_slice][:, 2] += k

                box_index += 1

    # select particles within shell
    observer = np.array([-0.5, -0.5, -0.5])
    particles = _particles - observer
    dist = np.sqrt(np.sum(particles**2, axis=1))
    shell = np.array(dist > shell_min) & np.array(dist < shell_max)

    # calculate and return the HEALPix positions for all particles within the shell
    theta = np.arccos(particles[shell][:, 2] / np.sqrt(np.sum((particles[shell]) ** 2, axis=1)), dtype=np.float32)
    phi = np.arctan2(particles[shell][:, 1], particles[shell][:, 0], dtype=np.float32)
    pixels = hp.ang2pix(nside, theta, phi, nest=False)

    return pixels
