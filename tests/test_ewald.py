# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Test Energy calculation based on Ewald method."""
from pycoulomb import Ewald
import numpy as np
from numpy.testing import assert_allclose
import pytest

# NaCL
positions_nacl = np.array([[0., 0., 0.], [.5, .5, .0], [.5, .0, .5],
                           [.0, .5, .5], [.5, 0., 0.], [.0, .5, .0],
                           [.0, .0, .5], [.5, .5, .5]])
charges_nacl = np.array([1, 1, 1, 1, -1, -1, -1, -1])
madelung_nacl = 1.7476

# CsCl
positions_cscl = np.array([[0., 0., 0.], [.5, .5, .5]])
charges_cscl = np.array([1, -1])
madelung_cscl = 1.7626 / np.sqrt(3)

# 3D
positions_3d = np.array([[0., 0., 0.]])
charges_3d = np.array([1])
madelung_3d = 2.837297 / 2


@pytest.mark.parametrize("positions, charges, madelung",
                         [(positions_nacl, charges_nacl, madelung_nacl),
                          (positions_cscl, charges_cscl, madelung_cscl),
                          (positions_3d, charges_3d, madelung_3d)])
def test_madelungt(positions, charges, madelung):
    """Test Ewald summation for calculating Madelung constant."""

    ewald = Ewald(positions=positions,
                  charges=charges,
                  L=1,
                  alpha=10,
                  n_kvecs=20,
                  r_cutoff=1,
                  epsilon=None)
    ewald.calculate_energy()

    assert_allclose(-ewald.energy / len(positions), madelung, rtol=1e-3)
