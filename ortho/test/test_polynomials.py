import unittest
import torch
from ortho.polynomials import (
    LaguerrePolynomial,
    HermitePolynomial,
    ProbabilistsHermitePolynomial,
    chebyshev_first,
    chebyshev_second,
)


# @unittest.skip("Not Implemented Yet")
class TestChebyshevPolynomials(unittest.TestCase):
    def setUp(self):
        self.x = torch.linspace(-1 + 0.001, 1 - 0.001, 1000)
        self.chebyshev_polynomials_first = [
            lambda x: torch.ones(x.shape),
            lambda x: x,
            lambda x: 2 * x**2 - 1,
            lambda x: 4 * x**3 - 3 * x,
            lambda x: 8 * x**4 - 8 * x**2 + 1,
            lambda x: 16 * x**5 - 20 * x**3 + 5 * x,
        ]

        self.chebyshev_polynomials_second = [
            lambda x: torch.ones(x.shape),
            lambda x: 2 * x,
            lambda x: 4 * x**2 - 1,
            lambda x: 8 * x**3 - 4 * x,
            lambda x: 16 * x**4 - 12 * x**2 + 1,
            lambda x: 32 * x**5 - 32 * x**3 + 6 * x,
        ]

    def test_chebyshev_first(self):
        # tests whether calculation of the chebyshev bases gets right answer
        for i in range(6):
            with self.subTest(i=i):
                values = chebyshev_first(self.x, i)
                test_values = torch.Tensor(
                    self.chebyshev_polynomials_first[i](self.x)
                )
                self.assertTrue(torch.allclose(values, test_values))

    def test_chebyshev_second(self):
        # tests whether calculation of the chebyshev bases gets right answer
        for i in range(6):
            with self.subTest(i=i):
                values = chebyshev_second(self.x, i)
                test_values = torch.Tensor(
                    self.chebyshev_polynomials_second[i](self.x)
                )
                self.assertTrue(torch.allclose(values, test_values))


class TestLaguerrePolynomials(unittest.TestCase):
    def setUp(self):
        # setup in here
        self.params = {"alpha": 0}
        self.x = torch.linspace(0 + 0.001, 5 - 0.001, 1000)
        self.testable_laguerre = LaguerrePolynomial(6)

        self.laguerre_polynomials = [
            lambda x: torch.ones(x.shape),  # 1
            lambda x: -x + 1,  # 1 - x
            lambda x: (1 / 2) * (x**2 - 4 * x + 2),
            lambda x: (1 / 6) * (-(x**3) + 9 * x**2 - 18 * x + 6),
            lambda x: (1 / 24)
            * (x**4 - 16 * x**3 + 72 * x**2 - 96 * x + 24),
            lambda x: (1 / 120)
            * (
                -(x**5)
                + 25 * x**4
                - 200 * x**3
                + 600 * x**2
                - 600 * x
                + 120
            ),
        ]

    def test_laguerre_polynomials(self):
        for i in range(6):
            with self.subTest(i=i):
                values = self.testable_laguerre(self.x, i, self.params)
                test_values = torch.Tensor(
                    self.laguerre_polynomials[i](self.x)
                )
                self.assertTrue(abs(values[10] - test_values[10]) < 1e-10)
            return
        return


class TestHermitePolynomials(unittest.TestCase):
    def setUp(self):
        self.inputs = torch.linspace(0, 10, 1000)
        self.subtest_count = 6
        self.testable_hermite = HermitePolynomial(6)
        self.testable_probabilists_hermite = ProbabilistsHermitePolynomial(6)

        self.prob_polynomials = [
            lambda x: torch.ones(x.shape),
            lambda x: x,
            lambda x: x**2 - 1,
            lambda x: x**3 - 3 * x,
            lambda x: x**4 - 6 * x**2 + 3,
            lambda x: x**5 - 10 * x**3 + 15 * x,
        ]
        self.phys_polynomials = [
            lambda x: torch.ones(x.shape),
            lambda x: 2 * x,
            lambda x: 4 * x**2 - 2,
            lambda x: 8 * x**3 - 12 * x,
            lambda x: 16 * x**4 - 48 * x**2 + 12,
            lambda x: 32 * x**5 - 160 * x**3 + 120 * x,
        ]

    def test_probabilists_polynomials(self):
        # compares the correct polynomials with the constructed hermite poly.
        params = dict()
        for i in range(self.subtest_count):
            with self.subTest(i=i):
                # zeros_vector = torch.zeros([i])
                values = self.testable_probabilists_hermite(
                    self.inputs, i, params
                )
                test_values = self.prob_polynomials[i](self.inputs)
                self.assertTrue(abs(values[10] - test_values[10]) < 1e-14)

    def test_physicists_polynomials(self):
        # compares the correct polynomials with the constructed hermite poly.
        params = dict()
        for i in range(self.subtest_count):
            with self.subTest(i=i):
                values = self.testable_hermite(self.inputs, i, params)
                test_values = self.phys_polynomials[i](self.inputs)
                self.assertTrue(torch.abs(values[10] - test_values[10]) < 1e-7)


if __name__ == "__main__":
    unittest.main()
