import unittest
import numpy as np

from dbn import DBN
import utils

__author__ = 'blackecho'


class DBNTest(unittest.TestCase):

    def test_forward_func(self):
        """The forward function should return an array of shape:
        (number of samples in data (1 in this case), last layer (8 in this case))
        """
        deep_net = DBN([10, 5, 8])
        data = np.array([[0, 1, 0, 1, 1, 1, 0, 1, 0, 0]])
        middle_out, out = deep_net.forward(data)
        self.assertEqual(middle_out[0].shape[0], 5)
        self.assertEqual(middle_out[1].shape[0], 8)
        self.assertEqual(out.shape, (1, 8))

    def test_backward_func(self):
        """The backward function should return an array of shape:
        (number of samples in data (1 in this case), first layer(10 in this case))
        """
        deep_net = DBN([13, 5, 7])
        data = np.array([[0, 0, 1, 1, 1, 0, 1]])
        out = deep_net.backward(data)
        self.assertEqual(out.shape, (1, 13))


class UtilsTest(unittest.TestCase):

    def test_filter_dropout(self):
        """Test the filter_dropout function in the utils.
        """
        w = np.array([np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]) for _ in range(10)])
        dp = [0, 1, 0, 1, 1, 0, 0, 0, 0, 0]
        filtered_w = utils.filter_dropout(w, dp)
        for row in filtered_w:
            self.assertEqual(row, [2, 4, 5])


if __name__ == '__main__':
    unittest.main()
