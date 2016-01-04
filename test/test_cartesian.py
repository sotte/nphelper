from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import TestCase, assert_array_equal, assert_raises

from nphelper import cartesian_product


class TestCartesianProduct(TestCase):
    def test_too_few_arrays(self):
        x = np.array([0, 1])
        assert_raises(ValueError, cartesian_product, [])
        assert_raises(ValueError, cartesian_product, [x])

    def test_two_arrays(self):
        x = np.array([0, 1])
        out = cartesian_product([x, x])
        expected = np.array([[0,  0], [0,  1], [1,  0], [1, 1]])
        assert_array_equal(out, expected)

    def test_three_arrays(self):
        x = np.array([0, 1])
        out = cartesian_product([x, x, x])
        expected = np.array([[0,  0,  0],
                             [0,  0,  1],
                             [0,  1,  0],
                             [0,  1,  1],
                             [1,  0,  0],
                             [1,  0,  1],
                             [1,  1,  0],
                             [1,  1,  1]])
        assert_array_equal(out, expected)

    def test_arrays_as_lists(self):
        x = [0, 1]
        out = cartesian_product([x, x])
        expected = np.array([[0,  0], [0,  1], [1,  0], [1, 1]])
        assert_array_equal(out, expected)

    def test_refuse_2D_arrays(self):
        x = np.array([[0, 1], [2, 3]])
        assert_raises(ValueError,  cartesian_product, [x, x])

    def test_valid_out_parameter(self):
        x = np.array([0, 1])
        out = np.empty((4, 2))
        cartesian_product([x, x], out=out)
        expected = np.array([[0,  0], [0,  1], [1,  0], [1,  1]])
        assert_array_equal(out, expected)

    def test_invalid_out_parameter(self):
        x = np.array([0, 1])
        out = np.empty((1, 1))  # not the correct shape
        assert_raises(ValueError,  cartesian_product, [x, x], out)
