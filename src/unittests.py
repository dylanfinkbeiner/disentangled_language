import unittest
import random
import numpy as np

import data_utils

class TestDataUtils(unittest.TestCase):

    def test_get_cutoffs(self):
        test_sentences = [np.random.randn(random.randint(1,10)) for _ in range(20)]

        test_sentences = sorted(test_sentences, key = lambda x: x.shape[0])

        i2c = data_utils.get_cutoffs(test_sentences)

        for index, cutoffs in i2c.items():
            self.assertTrue(len(cutoffs) == 2)
            self.assertTrue(index >= cutoffs[0])
            self.assertTrue(index < cutoffs[1])

        self.assertEqual(len(test_sentences), len(i2c))


if __name__ == '__main__':
    unittest.main()

