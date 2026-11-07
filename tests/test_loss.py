import unittest

import torch

from loss import gram_matrix


class GramMatrixTest(unittest.TestCase):
    def test_noncontiguous_feature_tensor(self):
        source = torch.arange(
            2 * 3 * 4 * 5,
            dtype=torch.float32,
        ).reshape(2, 3, 4, 5)
        feature = source.transpose(-2, -1)
        self.assertFalse(feature.is_contiguous())

        flattened = feature.reshape(2 * 3, 5 * 4)
        expected = flattened @ flattened.t()

        actual = gram_matrix(feature)

        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
