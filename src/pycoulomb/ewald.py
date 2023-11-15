# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Energy calculation based on Ewald method."""
import logging

import numpy as np
from MDAnalysis.lib.distances import capped_distance
from scipy.special import erfc


logger = logging.getLogger(__name__)


class Ewald:
    def __init__(
        self, positions, charges, L, r_cutoff=None, alpha=10, n_kvecs=20, epsilon=None
    ):
        """
        Calculate total energy of point charge distribution based on Ewald sum.

        Calculations are preformed in the CGS system.

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
        alpha : float
            Ewald parameter, tunes the relative weight of the real space
            and the reciprocal space contribution
        n_kvecs : int
            Numbre of k vectors for calculating reciprocal space contribution
        epsilon : float
            For `None` treat as epsilon is infinity (metallic)

        Attributes
        ----------
        energy : float
            total energy
        energy_real : float
            energy from the real space contribtion
        energy_reciprocal : float
            energy from the reciprocal space contribtion
        energy_self : float
            correction energy due to self-energy
        energy_dipole : float
            dipole correction. Only if `epsilon` is not `None`
        energy_neutralization : float
            Neutralization energy due to a homogeneous background for
            finite charged systems.  Only if total charge > 1e-8.
        """
        self.positions = positions
        self.charges = charges
        self.L = L
        self._r_cutoff = r_cutoff
        self.alpha = alpha
        self.n_kvecs = n_kvecs
        self.epsilon = epsilon

    def _rho_reciprocal_sq(self, kvec):
        kpos = np.atleast_2d(kvec @ self.positions.T)

        reals = self.charges * np.cos(kpos)
        imags = self.charges * np.sin(kpos)
        return np.sum(reals, axis=1) ** 2 + np.sum(imags, axis=1) ** 2

    def _calculate_energy_real(self):
        """Real part of the ewald energy."""
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

        self.energy_real = np.sum(q_i_q_j * erfc(self.alpha * r_ij) / r_ij) / 2

        logger.debug(f"real = {self.energy_real:.2e}")

    def _calculate_energy_reciprocal(self):
        """Reciprocal part of the ewald energy."""
        r_n = np.arange(-self.n_kvecs, self.n_kvecs + 1)
        kvecs = np.array(np.meshgrid(r_n, r_n, r_n), dtype=float).T.reshape(-1, 3)
        # Remove k = (0, 0, 0) vector
        kvecs = np.array([i for i in kvecs if np.any(i != (0, 0, 0))])
        kvecs *= 2 * np.pi / self.L

        logger.debug(f"kvector shape = {kvecs.shape}")

        k_sq = np.linalg.norm(kvecs, axis=1) ** 2

        energy_temp = np.exp(-k_sq / (4 * self.alpha**2)) / k_sq
        energy_temp *= self._rho_reciprocal_sq(kvecs)

        self.energy_reciprocal = np.sum(energy_temp)
        self.energy_reciprocal *= 2 * np.pi / self.L**3
        logger.debug(f"reciprocal = {self.energy_reciprocal:.2e}")

    def _calculate_energy_self(self):
        """Self part of the ewald energy"""
        self.energy_self = -self.alpha / np.sqrt(np.pi)
        self.energy_self *= np.sum(self.charges**2)

        logger.debug(f"self = {self.energy_self:.2e}")

    def _calculate_energy_dipole(self):
        """Dipole correction to the Ewald energy."""
        qpos = np.linalg.norm(self.positions * self.charges[:, None]) ** 2
        self.energy_dipole = 2 * np.pi / (1 + self.epsilon) / self.L**3
        self.energy_dipole *= qpos

        logger.debug(f"dipole = {self.energy_dipole:.2e}")

    def _calculate_energy_neutralization(self):
        """Correction if system has a finite net charge."""
        self.energy_neutralization = -np.pi / (2 * self.alpha**2 * self.L**3)
        self.energy_neutralization *= self.total_charge**2

        logger.debug(f"self = {self.energy_neutralization:.2e}")

    def _prepare(self):
        """Prepartions for calculations"""
        self.n_particles = len(self.positions)
        self.total_charge = np.sum(self.charges)
        self.dimensions = np.array([self.L, self.L, self.L, 90, 90, 90])

        if self._r_cutoff is None:
            self.r_cutoff = np.sqrt(3) / 2 * self.L
        else:
            self.r_cutoff = self._r_cutoff

    def calculate_energy(self):
        """Perform the energy calculation."""
        self._prepare()

        logger.info("Calculate real part")
        self._calculate_energy_real()
        self.energy = self.energy_real

        logger.info("Calculate reciporcal part")
        self._calculate_energy_reciprocal()
        self.energy += self.energy_reciprocal

        logger.info("Calculate self correction")
        self._calculate_energy_self()
        self.energy += self.energy_self

        if self.epsilon is not None:
            logger.info("Calculate dipole part")
            self._calculate_energy_dipole()
            self.energy += self.energy_dipole

        if self.total_charge > 1e-8:
            logger.info("Calculate neutralization part")
            self._calculate_energy_neutralization()
            self.energy += self.energy_neutralization
