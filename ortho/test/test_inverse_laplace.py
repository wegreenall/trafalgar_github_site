import unittest
import torch
from ortho.inverse_laplace import mgf, build_inverse_laplace


class TestInverseLaplaceL(unittest.TestCase):
    def setUp(self):
        self.order = 25
        self.x = torch.linspace(0, 1, 100)
        self.eps = 1e-6
        pass

    def test_mgf(self):
        moments = torch.ones(self.order)
        mgf_vals = mgf(self.x, moments)
        exp_vals = torch.exp(self.x)
        # print(torch.abs(exp_vals - mgf_vals))
        # breakpoint()

        self.assertTrue((torch.abs(mgf_vals - exp_vals) < self.eps).all())
        pass

    @unittest.skip("Too long!")
    def test_inverse_laplace(self):
        # test the sin function inverse laplace transform
        t = 2
        F = lambda x: t / (x ** 2 + t ** 2)
        b = 1
        s = 1
        my_sin = build_inverse_laplace(F, b, s, self.order)
        inverse_laplace_vals = my_sin(self.x)
        sin_vals = torch.sin(t * self.x)
        # print(torch.abs(sin_vals - inverse_laplace_vals))
        self.assertTrue(
            (torch.abs(sin_vals - inverse_laplace_vals) < self.eps).all()
        )
        pass


if __name__ == "__main__":
    unittest.main()
