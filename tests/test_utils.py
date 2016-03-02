import numpy as np
import unittest

import utils


class TestUtilsMethods(unittest.TestCase):
    """ Test the utils method in the utils module.
    """

    def setUp(self):
        """ Setup values for testing.
        """
        self.v = 13
        self.x = np.random.rand(39, 58)

    def test_masking_noise(self):
        """ test masking noise function.
        """
        x_noise = utils.masking_noise(self.x, self.v)

        for sample in x_noise:
            self.assertEqual(sum([i == 0 for i in sample]), self.v)

    def test_salt_and_pepper_noise_with_min_max(self):
        """ test the salt and pepper function with specified min and max values.
        """
        x_sp = utils.salt_and_pepper_noise(self.x, self.v, 0, 1)

        for sample in x_sp:
            salted_elems = sum([i == 0 or i == 1 for i in sample])
            self.assertEqual(salted_elems, self.v)

    def test_salt_and_pepper_noise_without_min_max(self):
        """ test the salt and pepper function without specified min and max values.
        """
        x_sp = utils.salt_and_pepper_noise(self.x, self.v)

        mn = self.x.min()
        mx = self.x.max()

        for sample in x_sp:
            salted_elements = sum([i == mn or i == mx for i in sample])
            self.assertAlmostEqual(salted_elements, self.v, delta=2)


if __name__ == '__main__':
    unittest.main()
