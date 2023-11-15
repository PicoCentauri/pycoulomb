# -*- Mode: python3; tab-width: 4; indent-tabs-mode:nil; coding:utf-8 -*-
# vim: tabstop=4 expandtab shiftwidth=4 softtabstop=4 fileencoding=utf-8
"""Test Energy calculation based on direct summation."""
import numpy as np
from numpy.testing import assert_allclose

from pycoulomb import Direct


def test_two_charges():
    """Test direct summation for calculating two particles."""
    r = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    q = np.array([1, -1])

    ewald = Direct(positions=r, charges=q, L=1)
    ewald.calculate_energy()

    assert_allclose(ewald.energy, -1 / np.linalg.norm(r[0] - r[1]), rtol=1e-3)
