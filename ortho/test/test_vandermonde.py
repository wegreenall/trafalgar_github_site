import torch

import math
import unittest

import matplotlib.pyplot as plt
from ortho.vandermonde import Bjorck_Pereyra


# def nchoosek(n, k):
# return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def nchoosek(n, k):
    return torch.exp(
        torch.lgamma(torch.tensor(n + 1))
        - (
            torch.lgamma(torch.tensor(k + 1))
            + torch.lgamma(torch.tensor(n - k + 1))
        )
    )


"""
Tests for the case as described in Bjorck and Pereyra:
    alpha_i = 1 / (i+3)
    b_i = 1/2^i  i = 0,1,...,n

"The exact solution can be shown to be:
    x_i = (-1)**i (n+1)Choose(i+1) (1 + (i+1)/2)^n
"""


class TestBjorckPereyra(unittest.TestCase):
    def setUp(self):
        # build the terms for the test
        n = 10
        lb = -7
        ub = 7
        self.alpha = torch.linspace(
            lb, ub, n
        )  # points at which we observe the fn
        self.a = torch.distributions.Exponential(1 / 10).sample([n])
        function_points = torch.zeros(n)
        # alpha are observation locations
        # a are the true (random!) coefficients
        for i in range(n):
            function_points += self.a[i] * self.alpha ** i  # alpha are
        self.function_points = function_points

        pass

    def test_vandermonde(self):
        eps = 0.0001
        result = Bjorck_Pereyra(self.alpha, self.function_points)
        self.assertTrue((torch.abs(self.a - result) < eps).all())


if __name__ == "__main__":
    unittest.main()
