import torch

import unittest
from ortho.utils import get_condition_number


class TestUtils(unittest.TestCase):
    def setUp(self):
        return

    def test_condition_number(self):
        condition = get_condition_number(torch.eye(10))
        self.assertEqual(condition, torch.Tensor([1.0]))
