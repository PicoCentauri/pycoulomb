# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Energy calculation based on Ewald method."""
import logging

import numpy as np
from MDAnalysis.lib.distances import capped_distance


logger = logging.getLogger(__name__)


class Direct:
    def __init__(self, positions, charges, L, r_cutoff=None):
        """
        Calculate energy of point charge distribution based on a direct sum.

        Only charges in the first unit cell are taken into account and
        no preiodic images. Calculations are preformed in the CGS system.

        Paramters
        ---------
        positions : np.ndarray
            positions of point charges
        charges : np.ndarray
            charges
        L : float
            length of the cubic unit cell
        r_cutoff : float
            cutoff for real space part. If `None` take half of `L`

        Attributes
        ----------
        energy : float
            total energy
        """
        self.positions = positions
        self.charges = charges
        self.L = L
        self._r_cutoff = r_cutoff

    def _calculate_energy_real(self):
        """Calculate energy."""
        pairs, r_ij = capped_distance(
            self.positions,
            self.positions,
            min_cutoff=0,
            max_cutoff=self.r_cutoff,
            box=self.dimensions,
            return_distances=True,
        )

        q_i_q_j = np.zeros(len(pairs))
        for i, pair in enumerate(pairs):
            q_i_q_j[i] = self.charges[pair[0]] * self.charges[pair[1]]

        self.energy_real = np.sum(q_i_q_j / r_ij) / 2

        logger.debug(f"real = {self.energy_real:.2e}")

    def _prepare(self):
        """Prepartions for calculations"""
        self.n_particles = len(self.positions)
        self.dimensions = np.array([self.L, self.L, self.L, 90, 90, 90])

        if self._r_cutoff is None:
            self.r_cutoff = np.sqrt(3) / 2 * self.L
        else:
            self.r_cutoff = self._r_cutoff

    def calculate_energy(self):
        """Perform the energy calculation."""
        self._prepare()

        logger.info("Calculate energy")
        self._calculate_energy_real()
        self.energy = self.energy_real
