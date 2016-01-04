from __future__ import division, absolute_import, print_function

import numpy as np
from numpy.testing import TestCase, assert_almost_equal

from nphelper import block


class TestBlock(TestCase):
    def test_block_row_wise(self):
        A = np.ones((2, 2))
        B = 2 * A
        assert_almost_equal(block([A, B]),
                            np.array([[1, 1, 2, 2],
                                      [1, 1, 2, 2]]))
        # tuple notation
        assert_almost_equal(block((A, B)),
                            np.array([[1, 1, 2, 2],
                                      [1, 1, 2, 2]]))

    def test_block_column_wise(self):
        A = np.ones((2, 2))
        B = 2 * A
        assert_almost_equal(block([[A], [B]]),
                            np.array([[1, 1],
                                      [1, 1],
                                      [2, 2],
                                      [2, 2]]))
        # tuple notation with only one element per tuple does not make much
        # sense. test it anyway just to make sure
        assert_almost_equal(block(((A, ), (B, ))),
                            np.array([[1, 1],
                                      [1, 1],
                                      [2, 2],
                                      [2, 2]]))

    def test_block_complex(self):
        # # # a bit more complex
        One = np.array([[1, 1, 1]])
        Two = np.array([[2, 2, 2]])
        Three = np.array([[3, 3, 3, 3, 3, 3]])
        four = np.array([4, 4, 4, 4, 4, 4])
        five = np.array([5])
        six = np.array([6, 6, 6, 6, 6])
        Zeros = np.zeros((2, 6))
        result = block([[One, Two],
                        [Three],
                        [four],
                        [five, six],
                        [Zeros]])
        expected = np.array([[1, 1, 1, 2, 2, 2],
                             [3, 3, 3, 3, 3, 3],
                             [4, 4, 4, 4, 4, 4],
                             [5, 6, 6, 6, 6, 6],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])
        assert_almost_equal(result, expected)

    def test_block_with_1d_arrays(self):
        # # # 1-D vectors are treated as row arrays
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        assert_almost_equal(block([a, b]),
                            np.array([[1, 2, 3, 2, 3, 4]]))
        assert_almost_equal(block([[a], [b]]),
                            np.array([[1, 2, 3],
                                      [2, 3, 4]]))
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        assert_almost_equal(block([[a, b], [a, b]]),
                            np.array([[1, 2, 3, 2, 3, 4],
                                      [1, 2, 3, 2, 3, 4]]))
