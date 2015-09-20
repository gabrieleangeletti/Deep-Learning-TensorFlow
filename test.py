import unittest
import numpy as np

from dbn import DBN

__author__ = 'blackecho'


class DBNTest(unittest.TestCase):

    def test_forward_func(self):
        """The forward function should return an array of shape:
        (number of samples in data (1 in this case), last layer (8 in this case))
        """
        deep_net = DBN([10, 5, 8])
        data = np.array([[0, 1, 0, 1, 1, 1, 0, 1, 0, 0]])
        out = deep_net.forward(data)
        self.assertEqual(out.shape, (1, 8))

    def test_backward_func(self):
        """The backward function should return an array of shape:
        (number of samples in data (1 in this case), first layer(10 in this case))
        """
        deep_net = DBN([13, 5, 7])
        data = np.array([[0, 0, 1, 1, 1, 0, 1]])
        out = deep_net.backward(data)
        self.assertEqual(out.shape, (1, 13))


if __name__ == '__main__':
    unittest.main()

