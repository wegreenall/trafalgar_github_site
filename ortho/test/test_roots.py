import torch
import torch.distributions as D
from ortho.roots import (
    get_polynomial_maximiser,
    get_polynomial_max,
    get_second_deriv_at_root,
    get_second_deriv_at_max,
    exp_of_poly,
)

# from ortho.builders import (
# integrate_function,
# )
import math

# from ortho.measure import coeffics_list
import unittest
import matplotlib.pyplot as plt

torch.set_default_tensor_type(torch.DoubleTensor)

torch.set_printoptions(precision=10)


class TestRootFinding(unittest.TestCase):
    def setUp(self):
        self.order = 5
        self.coeffics = torch.Tensor([0.0, 0.5, 3.5, -4, -20])
        self.roots = torch.Tensor([-0.4617, -0.1361, 0, 0.3978])
        self.peaks = torch.Tensor([-0.3493, -0.0672, 0.2664])
        self.integrand = lambda x: (
            (1 / math.sqrt(2 * math.pi)) * torch.exp(-(x ** 2) / 2)
        ).squeeze()

    def test_get_maximiser(self):
        poly_maximiser = get_polynomial_maximiser(self.coeffics, self.order)
        # breakpoint()
        self.assertTrue(
            torch.allclose(poly_maximiser, torch.tensor(0.2664), 1e-03)
        )

    def test_get_max(self):
        poly_max = get_polynomial_max(self.coeffics, self.order)
        self.assertTrue(torch.allclose(poly_max, torch.tensor(0.2052), 1e-03))

    def test_get_second_deriv_at_max(self):
        poly_max = get_second_deriv_at_max(self.coeffics, self.order)
        # print(test_second_derivs[i])
        self.assertTrue(
            torch.allclose(poly_max, torch.tensor(-16.432455248097114))
        )

    def test_get_second_deriv_at_root(self):
        test_second_derivs = torch.Tensor(
            [-33.0792536, 5.8208296, 7, -40.5259616]
        )
        for i, root in enumerate(self.roots):
            poly_max = get_second_deriv_at_root(
                self.coeffics, self.order, root
            )
            self.assertTrue(torch.allclose(poly_max, test_second_derivs[i]))

    def test_exp_of_poly(self):
        x = torch.linspace(-4, 4, 100)
        result = exp_of_poly(x, self.coeffics, self.order)
        self.assertEqual(result.shape, torch.Size([100]))


if __name__ == "__main__":
    unittest.main()
