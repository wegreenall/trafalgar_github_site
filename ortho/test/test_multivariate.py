import unittest
import torch
from multivariate import MultivariateStieltjes


# d = 2 | matrix testing
class TestMatrixFunction2d(unittest.TestCase):
    def setUp(self):
        self.n = 4
        self.d = 3
        self.i = 1
        self.j = 2
        self.sample_size = 10000 
        self.sample = torch.randn(self.sample_size, self.d)
        self.MSM = MultivariateStieltjes(n=self.n, d=self.d, sample=self.sample)

    def test_r_delta(self):
        self.assertEqual(self.MSM.r_delta(self.n, self.d), self.MSM.r(self.n, self.d) - self.MSM.r(self.n-1, self.d))

    def test_r(self):
        self.assertEqual(self.MSM.r(self.n, self.d), 15)

    def test_R(self):
        self.assertEqual(self.MSM.R(self.n, self.d), 35)

    def test_S(self):
        self.MSM.calculate()
        result = self.MSM.S(self.n, self.d)
        self.assertEqual(result.shape, torch.Size((r(self.n, self.d), r(self.n, self.d))))

    @unittest.skip("Not implemented")
    def test_T(self):
        result = self.MSM.T(self.n, self.i, self.j, self.d)
        self.assertEqual(result.shape, torch.Size((r(self.n, self.d), r(self.n, self.d))))

    def test_A(self):
        self.MSM.calculate()
        result = self.MSM.A(self.n, self.d)
        self.assertEqual(result.shape, torch.Size((r(self.n-1, self.d), r(self.n-1, self.d))))

    def test_B(self):
        result = self.MSM.B(self.n, self.d)
        self.assertEqual(result.shape, torch.Size((r(self.n-1, self.d), r(self.n-1, self.d))))

    @unittest.skip("Not implemented")
    def test_Sigma(self):
        result = self.MSM.Sigma(self.n, self.d)
        self.assertEqual(result.shape, torch.Size((r(self.n-1, self.d), r(self.n-1, self.d))))

    @unittest.skip("Not implemented")
    def test_U(self):
        result = self.MSM.U(self.n, self.i, self.d)
        self.assertEqual(result.shape, torch.Size((r(self.n-1, self.d), r(self.n-1, self.d))))

    @unittest.skip("Not implemented")
    def test_V_hat(self):
        result = self.MSM.V_hat(self.n, self.i, self.d)
        self.assertEqual(result.shape, torch.Size((r(self.n-1, self.d), r(self.n-1, self.d))))

    @unittest.skip("Not implemented")
    def test_V_tilde(self):
        result = self.MSM.V_tilde(self.n, self.i, self.d)
        self.assertEqual(result.shape, torch.Size((r_delta(self.n, self.d), r(self.n-1, self.d))))

    def test_ops_zero(self):
        inputs = torch.zeros(self.sample_size, self.d)
        result = self.MSM.ops(self.n, inputs)
        self.assertEqual(result.shape, torch.Size((r(self.n, self.d), )))

    def test_calculate_polynomials_As(self):
        self.MSM.calculate()
        self.assertEqual(len(self.MSM.As), self.n)

    @unittest.skip("Not implemented")
    def test_calculate_polynomials_Bs(self):
        self.MSM.calculate()
        self.assertEqual(len(self.MSM.Bs), self.n)


if __name__ == "__main__":
    unittest.main()
