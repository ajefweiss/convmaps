# -*- coding: utf-8 -*-
"""snapshot.py

"""

import logging
import numpy as np
import os
import struct


class Snapshot(object):
    _filename = None
    _properties = None

    def __init__(self):
        raise NotImplementedError

    @property
    def n(self):
        return self._properties['n']

    @property
    def z(self):
        if 'z' in self._properties:
            return self._properties['z']
        elif 'a' in self._properties:
            return 1 / np.round(self._properties['a'], 5) - 1
        else:
            raise KeyError

    def batch_load(self, batch_index, batch_count):
        raise NotImplementedError


class TipsySnapshot(Snapshot):
    """Snapshot implementation for the TIPSY format (http://faculty.washington.edu/trq/hpcc/tools/tipsy/tipsy.html).
    """
    def __init__(self, filename):
        f = open(filename, 'rb')

        logger = logging.getLogger(__name__)

        a, n, _, _, _, _ = struct.unpack(">diiiii", f.read(28))

        self._filename = filename
        self._properties = {
            'a': a,
            'n': n
        }

        f.close()

        logger.debug("loaded tipsy snapshot \"{0}\" (z={1:.2f})".format(os.path.basename(filename), self.z))

    def batch_load(self, batch_index, batch_count):
        """
        Loads the particle positions of the given batch from the snapshot file. All particles within the snapshot are
        divided into batches (the number of batches given by batch_count). The batch sizes may vary to accommodate all
        batches.

        Args:
            batch_index: batch index
            batch_count: total number of batches

        Returns: returns the particle positions within the specified batch

        """
        if batch_count <= batch_index:
            raise ValueError("batch_index must be strictly smaller than the given batch_count")

        batch_size = self.n / batch_count
        batch_size_r = int(np.floor(batch_size))
        batch_size_t = batch_size_r

        # procedure that divides n particles into more or less evenly sized batches
        b = 0
        r = 0.0

        for i in range(1, batch_index + 1):
            r += batch_size - batch_size_r

            batch_size_t = batch_size_r

            if r >= 1:
                batch_size_t += 1
                r -= 1

            b += batch_size_t

        if r > 0 and batch_index == batch_count - 1:
            batch_size_t += 1

        f = open(self._filename, 'rb')
        f.seek(32 + b * 36)

        return np.array(list(struct.iter_unpack(">4x3f20x", f.read(36 * batch_size_t))), dtype=np.float32)
