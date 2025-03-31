import torch
import math
from ortho.measure import MaximalEntropyDensity, CatNet, Polynomial
from ortho.utils import integrate_function, sample_from_function
import unittest
import matplotlib.pyplot as plt


class TestPolynomial(unittest.TestCase):
    def setUp(self):
        self.lambdas = torch.Tensor([1.0, 2.0, 3.0, 4.0])
        self.polynomial = Polynomial(self.lambdas)
        self.fineness = 1000
        self.inputs = torch.linspace(-5, 5, self.fineness)

    def test_order(self):
        self.assertEqual(self.polynomial.get_order(), 4)

    def test_call_shape(self):
        self.assertEqual(
            self.polynomial(self.inputs).shape, torch.Size([self.fineness])
        )

    def test_polynomial_evaluation(self):
        polyvals = torch.zeros(self.fineness)
        for i, coeffic in enumerate(self.lambdas):
            x = self.inputs ** i
            polyvals += coeffic * x
        self.assertTrue(torch.allclose(polyvals, self.polynomial(self.inputs)))


class TestMaximalEntropyDensity(unittest.TestCase):
    def setUp(self):
        self.catalan_numbers = torch.Tensor(
            [
                1,
                0,
                1,
                0,
                2,
                0,
                5,
                0,
                14,
                0,
                42,
                0,
                132,
                0,
                429,
                0,
                1430,
                0,
                4862,
                0,
                16796,
                0,
                58786,
                0,
                208012,
                0,
                742900,
                0,
                2674440,
                0,
                9694845,
                0,
                35357670,
                0,
                129644790,
                0,
                477638700,
                0,
                1767263190,
                0,
                6564120420,
                0,
            ]
        )

        self.end_point = 10
        self.input_length = 1000

        self.order = 8
        self.sample_size = 1000

        self.betas = torch.zeros(2 * self.order)
        self.gammas = torch.ones(2 * self.order)

        self.med = MaximalEntropyDensity(self.order, self.betas, self.gammas)

        self.input_points = torch.linspace(
            -self.end_point, self.end_point, self.input_length
        )
        pass

    def test_get_lambdas(self):
        lambdas = self.med._get_lambdas()
        self.assertEqual(lambdas.shape, torch.Size([self.order + 1]))  # 1

    def test_moments(self):
        moments, _ = self.med._get_moments()
        self.assertTrue(
            torch.allclose(moments, self.catalan_numbers[1 : self.order + 1])
        )

    @unittest.skip("Not relevant.")
    def test_actual_moments(self):
        maximum = torch.max(self.med(self.input_points))
        sample = sample_from_function(self.med, self.end_point, maximum)
        # calculated_moments = torch.zeros(self.order)
        for i in range(1, self.order):
            calculated_moment = torch.mean(sample ** i)
            with self.subTest(i=i):
                breakpoint()
                self.assertTrue(
                    torch.allclose(
                        torch.tensor(calculated_moment),
                        self.catalan_numbers[i],
                        atol=0.01,
                    )
                )

    def test_call_shape(self):
        output = self.med(self.input_points)
        self.assertEqual(output.shape, torch.Size([self.input_length]))

    def test_call_sign(self):
        output = self.med(self.input_points)
        self.assertTrue((output >= 0).all())

    def test_call_for_nans(self):
        output = self.med(self.input_points)
        self.assertTrue((output == output).all())

    def test_integral(self):
        med = MaximalEntropyDensity(self.order, self.betas, self.gammas)
        x = torch.linspace(-self.end_point, self.end_point, 1000)
        maximum = torch.max(self.med(x))
        medintegral = integrate_function(
            self.med, torch.Tensor([4.0]), maximum
        )

        self.assertTrue(
            torch.allclose(medintegral, torch.Tensor([1.0]), 1e-02)
        )


class TestCatNet(unittest.TestCase):
    def setUp(self):
        self.order = 10
        gammas = torch.ones(2 * self.order)
        betas = torch.zeros(2 * self.order)
        self.cat_net = CatNet(self.order, betas, gammas)
        self.catalan_numbers = torch.Tensor(
            [
                1,
                0,
                1,
                0,
                2,
                0,
                5,
                0,
                14,
                0,
                42,
                0,
                132,
                0,
                429,
                0,
                1430,
                0,
                4862,
                0,
                16796,
                0,
                58786,
                0,
                208012,
                0,
                742900,
                0,
                2674440,
                0,
                9694845,
                0,
                35357670,
                0,
                129644790,
                0,
                477638700,
                0,
                1767263190,
                0,
                6564120420,
                0,
            ]
        )

    @unittest.skip("Not Implemented Yet")
    def test_call_shape(self):
        result = self.cat_net(torch.Tensor([1.0]))
        self.assertEqual(result.shape, torch.Size([self.order]))

    @unittest.skip("Not Implemented Yet")
    def test_correctness(self):
        result = self.cat_net(torch.Tensor([1.0]))
        self.assertEqual(result, self.catalan_numbers[: self.order])


if __name__ == "__main__":
    unittest.main()
